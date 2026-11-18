#include "compile/pcfb_writer.h"

#include "compile/sha256.h"

#include <libxml/parser.h>
#include <libxml/xmlversion.h>

#include "initializer.h"
#include "model.h"
#include "pdag.h"
#include "preprocessor.h"
#include "settings.h"
#include "version.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fuzzybits::compile {
namespace {

pcfb::Error compile_error(std::string message) {
  return pcfb::Error{pcfb::Error::Code::BuilderError, std::move(message)};
}

void initialize_xml_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    LIBXML_TEST_VERSION
    xmlInitParser();
  });
}

const scram::mef::Gate *find_gate(const scram::mef::Model &model, std::string_view target) {
  for (const scram::mef::Gate &gate : model.gates()) {
    if (gate.id() == target || gate.name() == target || gate.full_path() == target) {
      return &gate;
    }
  }
  return nullptr;
}

const scram::mef::Gate *first_top_gate(const scram::mef::Model &model) {
  for (const scram::mef::FaultTree &fault_tree : model.fault_trees()) {
    if (!fault_tree.top_events().empty()) {
      return fault_tree.top_events().front();
    }
  }
  for (const scram::mef::Gate &gate : model.gates()) {
    return &gate;
  }
  return nullptr;
}

const scram::mef::BasicEvent *basic_event_for_variable(const scram::core::Pdag &graph, int index) {
  const auto &basic_events = graph.basic_events();
  const int offset = index - scram::core::Pdag::kVariableStartIndex;
  if (offset < 0 || static_cast<std::size_t>(offset) >= basic_events.size()) {
    return nullptr;
  }
  return basic_events[index];
}

void gather_all_nodes(const std::shared_ptr<scram::core::Gate> &gate,
                      std::vector<std::shared_ptr<scram::core::Node>> &nodes,
                      std::unordered_map<int, std::shared_ptr<scram::core::Node>> &nodes_by_index) {
  if (!gate || gate->Visited()) {
    return;
  }
  gate->Visit(1);
  nodes.push_back(gate);
  nodes_by_index[gate->index()] = gate;
  for (const auto &arg : gate->args<scram::core::Gate>()) {
    gather_all_nodes(arg.second, nodes, nodes_by_index);
  }
  for (const auto &arg : gate->args<scram::core::Variable>()) {
    if (!arg.second->Visited()) {
      arg.second->Visit(1);
      nodes.push_back(arg.second);
      nodes_by_index[arg.second->index()] = arg.second;
    }
  }
}

/// Layered toposort matching `layer_manager::layered_toposort` (host-only; no SYCL).
void layered_toposort(scram::core::Pdag *pdag,
                      std::vector<std::shared_ptr<scram::core::Node>> &nodes,
                      std::unordered_map<int, std::shared_ptr<scram::core::Node>> &nodes_by_index,
                      std::vector<std::vector<std::shared_ptr<scram::core::Node>>> &nodes_by_layer) {
  scram::core::pdag::LayeredTopologicalOrder(pdag);
  pdag->Clear<scram::core::Pdag::kVisit>();
  gather_all_nodes(pdag->root_ptr(), nodes, nodes_by_index);
  if (nodes.empty()) {
    return;
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const std::shared_ptr<scram::core::Node> &a, const std::shared_ptr<scram::core::Node> &b) {
              return a->order() < b->order();
            });
  const int max_layer = nodes.back()->order();
  nodes_by_layer.assign(static_cast<std::size_t>(max_layer) + 1, {});
  for (auto &node : nodes) {
    nodes_by_layer[static_cast<std::size_t>(node->order())].push_back(node);
  }
  for (auto &layer : nodes_by_layer) {
    std::sort(layer.begin(), layer.end(),
              [](const std::shared_ptr<scram::core::Node> &lhs, const std::shared_ptr<scram::core::Node> &rhs) {
                auto varL = std::dynamic_pointer_cast<scram::core::Variable>(lhs);
                auto varR = std::dynamic_pointer_cast<scram::core::Variable>(rhs);
                if (varL && !varR) {
                  return true;
                }
                if (!varL && varR) {
                  return false;
                }
                if (varL && varR) {
                  return lhs->index() < rhs->index();
                }
                auto gateL = std::dynamic_pointer_cast<scram::core::Gate>(lhs);
                auto gateR = std::dynamic_pointer_cast<scram::core::Gate>(rhs);
                if (gateL && gateR && gateL->type() != gateR->type()) {
                  return gateL->type() < gateR->type();
                }
                return lhs->index() < rhs->index();
              });
  }
}

pcfb::OpCode opcode_for(scram::core::Connective type) {
  switch (type) {
    case scram::core::kAnd:
      return pcfb::OpCode::BITWISE_AND;
    case scram::core::kOr:
      return pcfb::OpCode::BITWISE_OR;
    case scram::core::kXor:
      return pcfb::OpCode::BITWISE_XOR;
    case scram::core::kNot:
      return pcfb::OpCode::BITWISE_NOT;
    case scram::core::kNand:
      return pcfb::OpCode::BITWISE_NAND;
    case scram::core::kNor:
      return pcfb::OpCode::BITWISE_NOR;
    case scram::core::kAtleast:
      return pcfb::OpCode::BITWISE_ATLEAST;
    case scram::core::kNull:
      return pcfb::OpCode::BITWISE_NO_OP;
  }
  return pcfb::OpCode::BITWISE_NO_OP;
}

std::string generator_version() {
  std::string v = SCRAM_VERSION;
  if (constexpr const char *rev = SCRAM_GIT_REVISION; rev && rev[0] != '\0' && std::string_view(rev) != "unknown") {
    v += "+g";
    v += rev;
  }
  return v;
}

}  // namespace

std::string Options::to_cli_string() const {
  std::ostringstream os;
  os << "-c " << compilation_level;
  if (expand_atleast_gates) {
    os << " --no-kn";
  }
  if (expand_xor_gates) {
    os << " --no-xor";
  }
  if (!recognize_atleast_groups) {
    os << " --no-recognize-atleast";
  }
  if (!recognize_xor_groups) {
    os << " --no-recognize-xor";
  }
  if (keep_null_gates) {
    os << " --keep-null";
  }
  if (ccf) {
    os << " --ccf";
  }
  return os.str();
}

void apply_options(scram::core::Settings &settings, const Options &options) {
  settings.compilation_level(options.compilation_level);
  settings.expand_atleast_gates(options.expand_atleast_gates);
  settings.expand_xor_gates(options.expand_xor_gates);
  settings.recognize_atleast_groups(options.recognize_atleast_groups);
  settings.recognize_xor_groups(options.recognize_xor_groups);
  settings.keep_null_gates(options.keep_null_gates);
  settings.ccf_analysis(options.ccf);
}

pcfb::Result<std::vector<std::byte>> pdag_to_pcfb(scram::core::Pdag &pdag, const Options &options,
                                                  std::string_view model_name, std::string_view query_name,
                                                  std::span<const std::byte> source_sha256) {
  if (!pdag.root_ptr()) {
    return std::unexpected(compile_error("PDAG has no root"));
  }

  std::vector<std::shared_ptr<scram::core::Node>> nodes;
  std::unordered_map<int, std::shared_ptr<scram::core::Node>> nodes_by_index;
  std::vector<std::vector<std::shared_ptr<scram::core::Node>>> nodes_by_layer;
  layered_toposort(&pdag, nodes, nodes_by_index, nodes_by_layer);

  pcfb::builder b;
  b.name(model_name);
  if (auto r = b.set("general.generator", "fuzzybits"); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = b.set("general.generator_version", generator_version()); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = b.set("general.model_name", model_name); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = b.set("general.source_format", "openpsa-mef"); !r) {
    return std::unexpected(r.error());
  }
  if (!source_sha256.empty()) {
    if (auto r = b.set("general.source_sha256", source_sha256); !r) {
      return std::unexpected(r.error());
    }
  }
  if (auto r = b.set("compile.options", options.to_cli_string()); !r) {
    return std::unexpected(r.error());
  }

  std::unordered_map<int, pcfb::builder::tensor_id> tensor_of;
  std::optional<pcfb::builder::tensor_id> true_tensor;

  auto ensure_true = [&]() -> pcfb::Result<pcfb::builder::tensor_id> {
    if (true_tensor) {
      return *true_tensor;
    }
    auto t = b.add_op(pcfb::OpCode::BITWISE_ONES, {}, {}, std::nullopt, "true");
    if (!t) {
      return std::unexpected(t.error());
    }
    true_tensor = *t;
    return *t;
  };

  auto map_arg = [&](int signed_index) -> pcfb::Result<std::pair<pcfb::builder::tensor_id, bool>> {
    const int idx = std::abs(signed_index);
    const bool inverted = signed_index < 0;
    auto it = tensor_of.find(idx);
    if (it != tensor_of.end()) {
      return std::pair{it->second, inverted};
    }
    if (pdag.constant() && idx == pdag.constant()->index()) {
      auto t = ensure_true();
      if (!t) {
        return std::unexpected(t.error());
      }
      tensor_of[idx] = *t;
      return std::pair{*t, inverted};
    }
    return std::unexpected(compile_error("missing tensor for PDAG index " + std::to_string(idx) +
                                         " (not in topological order)"));
  };

  for (const auto &layer : nodes_by_layer) {
    for (const auto &node : layer) {
      if (auto var = std::dynamic_pointer_cast<scram::core::Variable>(node)) {
        // House events are folded to constants by the preprocessor. Remaining
        // variables are basic events (SampleSource::BASIC_EVENT). Module collapse
        // (COLLAPSED_MODULE, compile.exact_root) is deferred to fuzzybits#21.
        const scram::mef::BasicEvent *event = basic_event_for_variable(pdag, var->index());
        const std::string name = event ? event->id() : ("x" + std::to_string(var->index()));
        double p = 0.0;
        if (event && event->HasExpression()) {
          p = event->p();
        } else if (event) {
          return std::unexpected(compile_error("basic event " + name + " has no probability expression"));
        }
        auto t = b.add_sample(p, name, static_cast<std::uint32_t>(var->index()),
                              pcfb::SampleSource::BASIC_EVENT);
        if (!t) {
          return std::unexpected(t.error());
        }
        tensor_of[var->index()] = *t;
        continue;
      }

      auto gate = std::dynamic_pointer_cast<scram::core::Gate>(node);
      if (!gate) {
        continue;
      }

      std::vector<std::uint32_t> inputs;
      std::vector<std::uint32_t> inverted;
      for (int signed_index : gate->args()) {
        auto mapped = map_arg(signed_index);
        if (!mapped) {
          return std::unexpected(mapped.error());
        }
        auto [tid, is_inv] = *mapped;
        if (is_inv) {
          inverted.push_back(tid);
        } else {
          inputs.push_back(tid);
        }
      }

      const std::string gname = gate->mef_origin_ptr() ? std::string(gate->mef_origin_ptr()->id())
                                                       : ("g" + std::to_string(gate->index()));

      // Constant-folded gate: a single constant argument. Emit ONES or NOT(ONES).
      if (gate->constant() && inputs.empty() && inverted.size() == 1 && true_tensor &&
          inverted.front() == *true_tensor) {
        auto t = b.add_op(pcfb::OpCode::BITWISE_NOT, std::span<const std::uint32_t>{},
                          std::span<const std::uint32_t>{inverted.data(), inverted.size()}, std::nullopt, gname);
        if (!t) {
          return std::unexpected(t.error());
        }
        tensor_of[gate->index()] = *t;
        continue;
      }
      if (gate->constant() && inputs.size() == 1 && inverted.empty() && true_tensor &&
          inputs.front() == *true_tensor) {
        auto t = b.add_op(pcfb::OpCode::BITWISE_NO_OP, std::span<const std::uint32_t>{inputs.data(), inputs.size()},
                          {}, std::nullopt, gname);
        if (!t) {
          return std::unexpected(t.error());
        }
        tensor_of[gate->index()] = *t;
        continue;
      }

      // Empty AND is true, empty OR is false (Boolean identities after folding).
      if (inputs.empty() && inverted.empty()) {
        pcfb::OpCode fill = pcfb::OpCode::BITWISE_ZEROS;
        if (gate->type() == scram::core::kAnd || gate->type() == scram::core::kNand) {
          fill = (gate->type() == scram::core::kAnd) ? pcfb::OpCode::BITWISE_ONES : pcfb::OpCode::BITWISE_ZEROS;
        } else if (gate->type() == scram::core::kOr || gate->type() == scram::core::kNor) {
          fill = (gate->type() == scram::core::kOr) ? pcfb::OpCode::BITWISE_ZEROS : pcfb::OpCode::BITWISE_ONES;
        }
        auto t = b.add_op(fill, {}, {}, std::nullopt, gname);
        if (!t) {
          return std::unexpected(t.error());
        }
        tensor_of[gate->index()] = *t;
        continue;
      }

      pcfb::OpCode op = opcode_for(gate->type());
      std::optional<std::uint32_t> k;
      if (gate->type() == scram::core::kAtleast) {
        k = static_cast<std::uint32_t>(gate->min_number());
      }
      // Unary NO_OP / NOT: builder requires exactly one effective input.
      if ((op == pcfb::OpCode::BITWISE_NO_OP || op == pcfb::OpCode::BITWISE_NOT) &&
          (inputs.size() + inverted.size()) != 1) {
        return std::unexpected(compile_error("unary gate " + gname + " has arity " +
                                             std::to_string(inputs.size() + inverted.size())));
      }
      auto t = b.add_op(op, std::span<const std::uint32_t>{inputs.data(), inputs.size()},
                        std::span<const std::uint32_t>{inverted.data(), inverted.size()}, k, gname);
      if (!t) {
        return std::unexpected(t.error());
      }
      tensor_of[gate->index()] = *t;
    }
  }

  auto root_it = tensor_of.find(pdag.root_ptr()->index());
  if (root_it == tensor_of.end()) {
    return std::unexpected(compile_error("root gate was not lowered"));
  }
  pcfb::builder::tensor_id query_target = root_it->second;
  if (pdag.complement()) {
    const std::array<std::uint32_t, 1> ins{query_target};
    auto neg = b.add_op(pcfb::OpCode::BITWISE_NOT, ins, {}, std::nullopt, "root_complement");
    if (!neg) {
      return std::unexpected(neg.error());
    }
    query_target = *neg;
  }
  const std::string qname = query_name.empty() ? "top" : std::string(query_name);
  if (auto q = b.add_query(query_target, pcfb::QueryKind::PROBABILITY, qname); !q) {
    return std::unexpected(q.error());
  }
  return b.finish();
}

pcfb::Result<Compiled> compile_files(const std::vector<std::string> &input_files, const Options &options) {
  if (input_files.empty()) {
    return std::unexpected(compile_error("compile requires at least one input file"));
  }
  initialize_xml_once();

  scram::core::Settings settings;
  settings.algorithm(scram::core::Algorithm::kDirect);
  settings.probability_analysis(false);
  apply_options(settings, options);
  settings.input_files(input_files);

  std::unique_ptr<scram::mef::Model> model;
  try {
    model = scram::mef::Initializer(input_files, settings, options.allow_extern).model();
  } catch (const std::exception &ex) {
    return std::unexpected(compile_error(std::string("MEF initialize: ") + ex.what()));
  }
  settings.model(model.get());

  const scram::mef::Gate *target =
      options.target ? find_gate(*model, *options.target) : first_top_gate(*model);
  if (!target) {
    return std::unexpected(compile_error(options.target ? "target gate not found: " + *options.target
                                                        : "no fault-tree top gate"));
  }

  std::unordered_set<const scram::mef::Gate *> watched_gates;
  for (const scram::mef::Gate &gate : model->gates()) {
    watched_gates.insert(&gate);
  }

  std::unique_ptr<scram::core::Pdag> graph;
  try {
    scram::core::Pdag::SetWatchedGates(&watched_gates);
    graph = std::make_unique<scram::core::Pdag>(*target, settings.ccf_analysis(), model.get());
    scram::core::Pdag::SetWatchedGates(nullptr);
  } catch (const std::exception &ex) {
    scram::core::Pdag::SetWatchedGates(nullptr);
    return std::unexpected(compile_error(std::string("PDAG construction: ") + ex.what()));
  }

  if (graph->root_ptr() && !graph->root_ptr()->mef_origin_ptr()) {
    graph->root_ptr()->mef_origin_ptr(target);
  }

  try {
    scram::core::CustomPreprocessor<scram::mc::DirectEval> preprocessor(graph.get(), settings);
    preprocessor();
  } catch (const std::exception &ex) {
    return std::unexpected(compile_error(std::string("preprocessor: ") + ex.what()));
  }

  Compiled out;
  out.model_name = model->GetOptionalName();
  if (out.model_name.empty()) {
    out.model_name = std::filesystem::path(input_files.front()).stem().string();
  }
  out.target_id = target->id();
  out.target_name = target->name();

  std::array<std::uint8_t, 32> digest{};
  try {
    digest = sha256_files(input_files);
  } catch (const std::exception &ex) {
    return std::unexpected(compile_error(ex.what()));
  }
  std::byte sha_bytes[32];
  for (int i = 0; i < 32; ++i) {
    sha_bytes[i] = static_cast<std::byte>(digest[i]);
  }

  auto bytes = pdag_to_pcfb(*graph, options, out.model_name, out.target_id,
                            std::span<const std::byte>{sha_bytes, 32});
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  out.bytes = std::move(*bytes);
  return out;
}

}  // namespace fuzzybits::compile
