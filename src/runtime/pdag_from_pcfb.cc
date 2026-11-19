#include "runtime/runtime.h"

#include "event.h"
#include "expression/constant.h"
#include "pdag.h"

#include "pcfb/threshold.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace fuzzybits::runtime {
namespace {

constexpr double kTwoTo32 = 4294967296.0;  // 2^32

struct TensorBinding {
  std::shared_ptr<scram::core::Variable> variable;
  std::shared_ptr<scram::core::Gate> gate;

  [[nodiscard]] bool empty() const { return !variable && !gate; }
};

std::string sanitize_name(std::string name, std::uint32_t event_id) {
  if (name.empty()) {
    name = "e" + std::to_string(event_id);
  }
  for (char &c : name) {
    if (c == '.') {
      c = '_';
    }
  }
  return name;
}

std::string threshold_note(double p, std::uint64_t spec_threshold, bool always_true) {
  const auto kernel_threshold = static_cast<std::uint64_t>(p * kTwoTo32);
  std::ostringstream os;
  os << "p=" << p << " SampleArgs.threshold=" << spec_threshold
     << " always_true=" << (always_true ? "true" : "false")
     << " kernel_threshold=floor(p*2^32)=" << kernel_threshold
     << " (32-bit Philox word; #17 / Phase 2 lifts this to 64-bit)";
  return os.str();
}

scram::core::Connective connective_for(pcfb::OpCode op) {
  switch (op) {
    case pcfb::OpCode::BITWISE_AND:
      return scram::core::kAnd;
    case pcfb::OpCode::BITWISE_OR:
      return scram::core::kOr;
    case pcfb::OpCode::BITWISE_XOR:
      return scram::core::kXor;
    case pcfb::OpCode::BITWISE_NOT:
      return scram::core::kNot;
    case pcfb::OpCode::BITWISE_NAND:
      return scram::core::kNand;
    case pcfb::OpCode::BITWISE_NOR:
      return scram::core::kNor;
    case pcfb::OpCode::BITWISE_ATLEAST:
      return scram::core::kAtleast;
    case pcfb::OpCode::BITWISE_NO_OP:
      return scram::core::kNull;
    default:
      throw RuntimeError("pdag_from_circuit: opcode cannot map directly to a PDAG connective");
  }
}

void add_arg(const std::shared_ptr<scram::core::Gate> &parent, const TensorBinding &binding, bool complement) {
  if (binding.variable) {
    parent->AddArg(binding.variable, complement);
  } else if (binding.gate) {
    parent->AddArg(binding.gate, complement);
  } else {
    throw RuntimeError("pdag_from_circuit: empty tensor binding");
  }
}

std::shared_ptr<scram::core::Gate> make_gate(scram::core::Pdag *graph, scram::core::Connective type) {
  return std::make_shared<scram::core::Gate>(type, graph);
}

/// PDAG XOR is binary. Fold n-ary XOR left-to-right.
std::shared_ptr<scram::core::Gate>
xor_fold(scram::core::Pdag *graph, const std::vector<std::pair<TensorBinding, bool>> &args) {
  if (args.empty()) {
    throw RuntimeError("pdag_from_circuit: XOR with no inputs");
  }
  if (args.size() == 1) {
    auto g = make_gate(graph, args.front().second ? scram::core::kNot : scram::core::kNull);
    add_arg(g, args.front().first, false);
    return g;
  }
  auto acc = make_gate(graph, scram::core::kXor);
  add_arg(acc, args[0].first, args[0].second);
  add_arg(acc, args[1].first, args[1].second);
  for (std::size_t i = 2; i < args.size(); ++i) {
    auto next = make_gate(graph, scram::core::kXor);
    next->AddArg(acc, false);
    add_arg(next, args[i].first, args[i].second);
    acc = std::move(next);
  }
  return acc;
}

}  // namespace

ReconstructedPdag::ReconstructedPdag() = default;
ReconstructedPdag::ReconstructedPdag(ReconstructedPdag &&) noexcept = default;
ReconstructedPdag &ReconstructedPdag::operator=(ReconstructedPdag &&) noexcept = default;
ReconstructedPdag::~ReconstructedPdag() = default;

void apply_backend_mask(const std::string &backend) {
  if (backend.empty()) {
    return;
  }
  const char *mask = nullptr;
  if (backend == "cuda") {
    mask = "cuda";
  } else if (backend == "omp") {
    mask = "omp";
  } else {
    throw RuntimeError("unknown backend '" + backend + "' (expected cuda or omp)");
  }
  if (setenv("ACPP_VISIBILITY_MASK", mask, 1) != 0) {
    throw RuntimeError("failed to set ACPP_VISIBILITY_MASK=" + std::string(mask));
  }
}

ReconstructedPdag pdag_from_circuit(const pcfb::ir::Circuit &circuit) {
  if (!circuit.valid() || !circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    throw RuntimeError("pdag_from_circuit: empty circuit");
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (!sc || !sc->operators() || !sc->queries() || sc->queries()->size() == 0) {
    throw RuntimeError("pdag_from_circuit: missing operators or queries");
  }

  ReconstructedPdag out;
  out.graph = std::make_unique<scram::core::Pdag>();
  auto *graph = out.graph.get();

  const auto *ops = sc->operators();
  const auto n_ops = ops->size();
  std::vector<TensorBinding> tensors;
  if (sc->tensors()) {
    tensors.resize(sc->tensors()->size());
  }

  std::shared_ptr<scram::core::Variable> ones_var;
  std::shared_ptr<scram::core::Variable> zeros_var;

  auto ensure_const_var = [&](bool value) -> std::shared_ptr<scram::core::Variable> & {
    auto &slot = value ? ones_var : zeros_var;
    if (slot) {
      return slot;
    }
    auto expr = std::make_unique<scram::mef::ConstantExpression>(value ? 1.0 : 0.0);
    const char *name = value ? "pcfb_ones" : "pcfb_zeros";
    auto event = std::make_unique<scram::mef::BasicEvent>(name);
    event->expression(expr.get());
    slot = std::make_shared<scram::core::Variable>(graph);
    graph->SetBasicEvent(slot->index(), event.get());
    out.expressions.push_back(std::move(expr));
    out.events.push_back(std::move(event));
    return slot;
  };

  // Pass 1: place SAMPLE variables at event_id so Philox pdag_idx matches compile.
  struct SampleInfo {
    flatbuffers::uoffset_t op_index;
    std::uint32_t event_id;
    double p;
    bool always_true;
    std::uint64_t threshold;
    std::string name;
  };
  std::vector<SampleInfo> samples;
  for (flatbuffers::uoffset_t i = 0; i < n_ops; ++i) {
    const auto *op = ops->Get(i);
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    const auto *sa = op->args_as_SampleArgs();
    if (!sa) {
      throw RuntimeError("pdag_from_circuit: SAMPLE without SampleArgs");
    }
    SampleInfo info;
    info.op_index = i;
    info.event_id = sa->event_id();
    info.p = sa->probability();
    info.always_true = sa->always_true();
    info.threshold = sa->threshold();
    info.name = sa->name() ? sa->name()->str() : "";
    if (info.event_id < static_cast<std::uint32_t>(scram::core::Pdag::kVariableStartIndex)) {
      throw RuntimeError("pdag_from_circuit: event_id " + std::to_string(info.event_id) +
                         " collides with Constant index 1");
    }
    samples.push_back(std::move(info));
  }
  std::sort(samples.begin(), samples.end(),
            [](const SampleInfo &a, const SampleInfo &b) { return a.event_id < b.event_id; });
  for (std::size_t i = 1; i < samples.size(); ++i) {
    if (samples[i].event_id == samples[i - 1].event_id) {
      throw RuntimeError("pdag_from_circuit: duplicate event_id " + std::to_string(samples[i].event_id));
    }
  }

  std::unordered_map<std::uint32_t, std::shared_ptr<scram::core::Variable>> vars_by_event;
  for (const auto &s : samples) {
    graph->SetNextNodeIndex(static_cast<int>(s.event_id));
    auto var = std::make_shared<scram::core::Variable>(graph);
    if (var->index() != static_cast<int>(s.event_id)) {
      throw RuntimeError("pdag_from_circuit: failed to place variable at event_id " +
                         std::to_string(s.event_id));
    }
    auto expr = std::make_unique<scram::mef::ConstantExpression>(s.always_true ? 1.0 : s.p);
    auto event = std::make_unique<scram::mef::BasicEvent>(sanitize_name(s.name, s.event_id));
    event->expression(expr.get());
    graph->SetBasicEvent(var->index(), event.get());
    vars_by_event[s.event_id] = var;
    out.threshold_notes.push_back(threshold_note(s.p, s.threshold, s.always_true));
    const auto expected = pcfb::probability_to_threshold(s.always_true ? 1.0 : s.p);
    if (!s.always_true && expected != s.threshold) {
      // validate() already enforces ±1 ulp; keep a hard check for the runtime.
      throw RuntimeError("pdag_from_circuit: SampleArgs.threshold does not match p·2^64 for event_id " +
                         std::to_string(s.event_id));
    }
    out.expressions.push_back(std::move(expr));
    out.events.push_back(std::move(event));
  }

  auto bind_output = [&](std::uint32_t output, TensorBinding binding) {
    if (output >= tensors.size()) {
      tensors.resize(output + 1);
    }
    if (!tensors[output].empty()) {
      throw RuntimeError("pdag_from_circuit: tensor " + std::to_string(output) + " assigned twice");
    }
    tensors[output] = std::move(binding);
  };

  auto lookup = [&](std::uint32_t id) -> TensorBinding {
    if (id >= tensors.size() || tensors[id].empty()) {
      throw RuntimeError("pdag_from_circuit: tensor " + std::to_string(id) + " used before assignment");
    }
    return tensors[id];
  };

  auto collect_args = [&](const pcfb::Op *op) {
    std::vector<std::pair<TensorBinding, bool>> args;
    if (op->inputs()) {
      for (auto id : *op->inputs()) {
        args.emplace_back(lookup(id), false);
      }
    }
    if (op->inverted_inputs()) {
      for (auto id : *op->inverted_inputs()) {
        args.emplace_back(lookup(id), true);
      }
    }
    return args;
  };

  auto emit_simple = [&](pcfb::OpCode opcode, const std::vector<std::pair<TensorBinding, bool>> &args,
                         std::uint32_t k) -> std::shared_ptr<scram::core::Gate> {
    if (opcode == pcfb::OpCode::BITWISE_XOR || opcode == pcfb::OpCode::BITWISE_XNOR) {
      auto g = xor_fold(graph, args);
      if (opcode == pcfb::OpCode::BITWISE_XNOR) {
        auto neg = make_gate(graph, scram::core::kNot);
        neg->AddArg(g, false);
        return neg;
      }
      return g;
    }
    if (opcode == pcfb::OpCode::BITWISE_ATMOST) {
      auto inner = make_gate(graph, scram::core::kAtleast);
      inner->min_number(static_cast<int>(k + 1));
      for (const auto &[b, inv] : args) {
        add_arg(inner, b, inv);
      }
      auto neg = make_gate(graph, scram::core::kNot);
      neg->AddArg(inner, false);
      return neg;
    }
    if (opcode == pcfb::OpCode::BITWISE_EXACTLY) {
      auto atleast = make_gate(graph, scram::core::kAtleast);
      atleast->min_number(static_cast<int>(k));
      auto atleast_next = make_gate(graph, scram::core::kAtleast);
      atleast_next->min_number(static_cast<int>(k + 1));
      for (const auto &[b, inv] : args) {
        add_arg(atleast, b, inv);
        add_arg(atleast_next, b, inv);
      }
      auto not_next = make_gate(graph, scram::core::kNot);
      not_next->AddArg(atleast_next, false);
      auto both = make_gate(graph, scram::core::kAnd);
      both->AddArg(atleast, false);
      both->AddArg(not_next, false);
      return both;
    }

    auto g = make_gate(graph, connective_for(opcode));
    if (opcode == pcfb::OpCode::BITWISE_ATLEAST) {
      g->min_number(static_cast<int>(k));
    }
    for (const auto &[b, inv] : args) {
      add_arg(g, b, inv);
    }
    return g;
  };

  // Pass 2: topological ops.
  for (flatbuffers::uoffset_t i = 0; i < n_ops; ++i) {
    const auto *op = ops->Get(i);
    if (!op) {
      continue;
    }
    const auto output = op->output();
    switch (op->opcode()) {
      case pcfb::OpCode::SAMPLE: {
        const auto *sa = op->args_as_SampleArgs();
        TensorBinding b;
        b.variable = vars_by_event.at(sa->event_id());
        bind_output(output, std::move(b));
        break;
      }
      case pcfb::OpCode::BITWISE_ONES: {
        TensorBinding b;
        b.variable = ensure_const_var(true);
        bind_output(output, std::move(b));
        break;
      }
      case pcfb::OpCode::BITWISE_ZEROS: {
        TensorBinding b;
        b.variable = ensure_const_var(false);
        bind_output(output, std::move(b));
        break;
      }
      default: {
        const auto args = collect_args(op);
        const std::uint32_t k =
            (op->args_type() == pcfb::OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k()
                                                                                  : 0;
        TensorBinding b;
        b.gate = emit_simple(op->opcode(), args, k);
        bind_output(output, std::move(b));
        break;
      }
    }
  }

  for (flatbuffers::uoffset_t qi = 0; qi < sc->queries()->size(); ++qi) {
    const auto *q = sc->queries()->Get(qi);
    if (!q) {
      continue;
    }
    const auto binding = lookup(q->target());
    ReconstructedQuery rq;
    rq.name = q->name() ? q->name()->str() : ("q" + std::to_string(qi));
    if (binding.gate) {
      rq.node = binding.gate;
      rq.node_index = binding.gate->index();
    } else if (binding.variable) {
      auto wrap = make_gate(graph, scram::core::kNull);
      wrap->AddArg(binding.variable, false);
      rq.node = wrap;
      rq.node_index = wrap->index();
    } else {
      throw RuntimeError("pdag_from_circuit: query target is empty");
    }
    out.queries.push_back(std::move(rq));
  }
  if (out.queries.empty()) {
    throw RuntimeError("pdag_from_circuit: no queries");
  }

  auto root_gate = std::dynamic_pointer_cast<scram::core::Gate>(out.queries.front().node);
  if (!root_gate) {
    throw RuntimeError("pdag_from_circuit: query node is not a gate");
  }
  if (out.queries.size() > 1) {
    auto dummy = make_gate(graph, scram::core::kAnd);
    for (const auto &q : out.queries) {
      auto g = std::dynamic_pointer_cast<scram::core::Gate>(q.node);
      if (!g) {
        throw RuntimeError("pdag_from_circuit: query node is not a gate");
      }
      dummy->AddArg(g, false);
    }
    root_gate = dummy;
  }
  graph->root(root_gate);
  return out;
}

}  // namespace fuzzybits::runtime
