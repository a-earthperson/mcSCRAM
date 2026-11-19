#include "runtime/runtime.h"

#include "event.h"
#include "expression/constant.h"
#include "pdag.h"

#include "pcfb/threshold.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace fuzzybits::runtime {
namespace {

struct TensorBinding {
  std::shared_ptr<scram::core::Variable> variable;
  std::shared_ptr<scram::core::Gate> gate;
  bool is_constant = false;
  bool constant_true = true;

  [[nodiscard]] bool empty() const { return !variable && !gate && !is_constant; }
};

std::string sanitize_name(std::string name, std::uint32_t event_id) {
  if (name.empty()) {
    name = "e" + std::to_string(event_id);
  }
  if (name.find('.') != std::string::npos) {
    for (char &c : name) {
      if (c == '.') {
        c = '_';
      }
    }
  }
  return name;
}

std::optional<std::string> meta_string(const pcfb::ir::Circuit &circuit, std::string_view key) {
  const auto *fb = circuit.get();
  if (!fb || !fb->metadata()) {
    return std::nullopt;
  }
  for (const auto *kv : *fb->metadata()) {
    if (!kv || !kv->key() || kv->key()->str() != key) {
      continue;
    }
    if (const auto *s = kv->value_as_StringValue(); s && s->value()) {
      return s->value()->str();
    }
    if (const auto *i = kv->value_as_IntValue()) {
      return std::to_string(i->value());
    }
  }
  return std::nullopt;
}

std::optional<std::int64_t> meta_int(const pcfb::ir::Circuit &circuit, std::string_view key) {
  const auto *fb = circuit.get();
  if (!fb || !fb->metadata()) {
    return std::nullopt;
  }
  for (const auto *kv : *fb->metadata()) {
    if (!kv || !kv->key() || kv->key()->str() != key) {
      continue;
    }
    if (const auto *i = kv->value_as_IntValue()) {
      return i->value();
    }
  }
  return std::nullopt;
}

std::string json_unescape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 >= s.size()) {
      out += s[i];
      continue;
    }
    const char n = s[++i];
    switch (n) {
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case '"':
        out += '"';
        break;
      case '\\':
        out += '\\';
        break;
      default:
        out += n;
        break;
    }
  }
  return out;
}

std::optional<std::string> json_string_field(std::string_view obj, std::string_view key) {
  const std::string pat = "\"" + std::string(key) + "\":\"";
  const auto pos = obj.find(pat);
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  std::size_t i = pos + pat.size();
  std::string raw;
  while (i < obj.size()) {
    if (obj[i] == '\\' && i + 1 < obj.size()) {
      raw += obj[i];
      raw += obj[i + 1];
      i += 2;
      continue;
    }
    if (obj[i] == '"') {
      break;
    }
    raw += obj[i];
    ++i;
  }
  return json_unescape(raw);
}

std::optional<int> json_int_field(std::string_view obj, std::string_view key) {
  const std::string pat = "\"" + std::string(key) + "\":";
  const auto pos = obj.find(pat);
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  std::size_t i = pos + pat.size();
  while (i < obj.size() && (obj[i] == ' ' || obj[i] == '\t')) {
    ++i;
  }
  const bool neg = i < obj.size() && obj[i] == '-';
  if (neg) {
    ++i;
  }
  if (i >= obj.size() || obj[i] < '0' || obj[i] > '9') {
    return std::nullopt;
  }
  int v = 0;
  while (i < obj.size() && obj[i] >= '0' && obj[i] <= '9') {
    v = v * 10 + (obj[i] - '0');
    ++i;
  }
  return neg ? -v : v;
}

std::vector<OriginRecord> parse_origins(std::string_view json) {
  std::vector<OriginRecord> out;
  std::size_t i = 0;
  while (i < json.size()) {
    const auto start = json.find('{', i);
    if (start == std::string_view::npos) {
      break;
    }
    const auto end = json.find('}', start + 1);
    if (end == std::string_view::npos) {
      break;
    }
    const auto obj = json.substr(start, end - start + 1);
    OriginRecord rec;
    if (auto idx = json_int_field(obj, "index")) {
      rec.pdag_index = *idx;
    }
    rec.kind = json_string_field(obj, "kind").value_or("");
    rec.id = json_string_field(obj, "id").value_or("");
    rec.name = json_string_field(obj, "name").value_or("");
    rec.label = json_string_field(obj, "label").value_or("");
    rec.full_path = json_string_field(obj, "full_path").value_or("");
    if (rec.pdag_index != 0 || !rec.id.empty() || !rec.name.empty()) {
      out.push_back(std::move(rec));
    }
    i = end + 1;
  }
  return out;
}

std::unordered_map<std::uint32_t, int> parse_tensor_pdag(std::string_view s) {
  std::unordered_map<std::uint32_t, int> out;
  std::size_t i = 0;
  while (i < s.size()) {
    std::uint32_t tid = 0;
    int pdag = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      tid = tid * 10 + static_cast<std::uint32_t>(s[i] - '0');
      ++i;
    }
    if (i >= s.size() || s[i] != ':') {
      break;
    }
    ++i;
    const bool neg = i < s.size() && s[i] == '-';
    if (neg) {
      ++i;
    }
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      pdag = pdag * 10 + (s[i] - '0');
      ++i;
    }
    out[tid] = neg ? -pdag : pdag;
    if (i < s.size() && s[i] == ',') {
      ++i;
    }
  }
  return out;
}

std::vector<std::string> split_lines(std::string_view s) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    auto nl = s.find('\n', i);
    if (nl == std::string_view::npos) {
      out.emplace_back(s.substr(i));
      break;
    }
    out.emplace_back(s.substr(i, nl - i));
    i = nl + 1;
  }
  return out;
}

GraphMeta graph_meta_from_circuit(const pcfb::ir::Circuit &circuit) {
  GraphMeta meta;
  if (auto v = meta_string(circuit, "general.model_name")) {
    meta.model_name = *v;
  }
  if (auto v = meta_string(circuit, "fuzzybits.source")) {
    meta.source = *v;
  }
  if (auto v = meta_string(circuit, "fuzzybits.input_files")) {
    meta.input_files = split_lines(*v);
  } else if (!meta.source.empty()) {
    meta.input_files = {meta.source};
  }
  if (auto v = meta_string(circuit, "fuzzybits.target")) {
    meta.target = *v;
  }
  if (auto v = meta_string(circuit, "fuzzybits.target_name")) {
    meta.target_name = *v;
  }
  if (auto v = meta_int(circuit, "compile.level")) {
    meta.compilation_level = static_cast<int>(*v);
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.ccf")) {
    meta.ccf = *v != 0;
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.expand_atleast_gates")) {
    meta.expand_atleast_gates = *v != 0;
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.expand_xor_gates")) {
    meta.expand_xor_gates = *v != 0;
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.recognize_atleast_groups")) {
    meta.recognize_atleast_groups = *v != 0;
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.recognize_xor_groups")) {
    meta.recognize_xor_groups = *v != 0;
    meta.has_compilation = true;
  }
  if (auto v = meta_int(circuit, "compile.keep_null_gates")) {
    meta.keep_null_gates = *v != 0;
    meta.has_compilation = true;
  }
  return meta;
}

std::unique_ptr<scram::mef::Gate> make_origin_gate(std::string name, const OriginRecord *rec) {
  if (name.empty() && rec) {
    name = rec->name.empty() ? rec->id : rec->name;
  }
  if (name.empty()) {
    return nullptr;
  }
  std::string base;
  auto role = scram::mef::RoleSpecifier::kPublic;
  if (name.find('.') != std::string::npos) {
    const auto pos = name.rfind('.');
    base = name.substr(0, pos);
    name = name.substr(pos + 1);
    if (!base.empty() && base.front() == '.') {
      base.erase(base.begin());
    }
    if (!base.empty()) {
      role = scram::mef::RoleSpecifier::kPrivate;
    }
  } else if (rec && rec->id != rec->name && !rec->full_path.empty() && rec->full_path.front() != '.') {
    const auto pos = rec->full_path.rfind('.');
    if (pos != std::string::npos && pos > 0) {
      base = rec->full_path.substr(0, pos);
      role = scram::mef::RoleSpecifier::kPrivate;
    }
  }
  try {
    auto gate = std::make_unique<scram::mef::Gate>(name, base, role);
    if (rec && !rec->label.empty()) {
      gate->label(rec->label);
    }
    return gate;
  } catch (const std::exception &) {
    try {
      return std::make_unique<scram::mef::Gate>(sanitize_name(std::move(name), 0));
    } catch (const std::exception &) {
      return nullptr;
    }
  }
}

std::string threshold_note(double p, std::uint64_t spec_threshold, bool always_true) {
  const auto T = pcfb::probability_to_threshold(always_true ? 1.0 : p);
  std::ostringstream os;
  os << "p=" << p << " SampleArgs.threshold=" << spec_threshold
     << " always_true=" << (always_true ? "true" : "false")
     << " kernel_T_hi=" << static_cast<std::uint32_t>(T >> 32U)
     << " kernel_T_lo=" << static_cast<std::uint32_t>(T)
     << " (two-word Philox compare; #17)";
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

void add_arg(scram::core::Pdag *graph, const std::shared_ptr<scram::core::Gate> &parent,
             const TensorBinding &binding, bool complement) {
  if (binding.is_constant) {
    const bool want_false = binding.constant_true ? complement : !complement;
    parent->AddArg(graph->constant(), want_false);
    return;
  }
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
    add_arg(graph, g, args.front().first, false);
    return g;
  }
  auto acc = make_gate(graph, scram::core::kXor);
  add_arg(graph, acc, args[0].first, args[0].second);
  add_arg(graph, acc, args[1].first, args[1].second);
  for (std::size_t i = 2; i < args.size(); ++i) {
    auto next = make_gate(graph, scram::core::kXor);
    next->AddArg(acc, false);
    add_arg(graph, next, args[i].first, args[i].second);
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
  } else if (backend == "metal") {
    mask = "metal";
  } else {
    throw RuntimeError("unknown backend '" + backend + "' (expected cuda, omp, or metal)");
  }
#ifdef _WIN32
  if (_putenv_s("ACPP_VISIBILITY_MASK", mask) != 0) {
#else
  if (setenv("ACPP_VISIBILITY_MASK", mask, 1) != 0) {
#endif
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
  out.meta = graph_meta_from_circuit(circuit);
  if (auto origins = meta_string(circuit, "fuzzybits.origins")) {
    out.origins = parse_origins(*origins);
  }
  std::unordered_map<int, const OriginRecord *> origin_by_index;
  for (const auto &rec : out.origins) {
    origin_by_index[rec.pdag_index] = &rec;
  }
  std::unordered_map<std::uint32_t, int> tensor_pdag;
  if (auto map = meta_string(circuit, "fuzzybits.tensor_pdag")) {
    tensor_pdag = parse_tensor_pdag(*map);
  }
  auto *graph = out.graph.get();

  const auto *ops = sc->operators();
  const auto n_ops = ops->size();
  std::vector<TensorBinding> tensors;
  if (sc->tensors()) {
    tensors.resize(sc->tensors()->size());
  }

  auto place_gate = [&](scram::core::Connective type, std::uint32_t output) {
    auto it = tensor_pdag.find(output);
    if (it != tensor_pdag.end() && it->second > graph->last_node_index()) {
      try {
        graph->SetNextNodeIndex(it->second);
      } catch (const std::exception &) {
      }
    }
    return make_gate(graph, type);
  };

  auto attach_origin = [&](const std::shared_ptr<scram::core::Gate> &gate, const pcfb::Op *op,
                           std::uint32_t output) {
    if (!gate) {
      return;
    }
    const OriginRecord *rec = nullptr;
    auto pdag_it = tensor_pdag.find(output);
    if (pdag_it != tensor_pdag.end()) {
      auto rec_it = origin_by_index.find(pdag_it->second);
      if (rec_it != origin_by_index.end()) {
        rec = rec_it->second;
      }
    }
    std::string name;
    if (op && op->name() && op->name()->size()) {
      name = op->name()->str();
    } else if (rec) {
      name = rec->id.empty() ? rec->name : rec->id;
    }
    if (name.empty() || name == "true" || name == "root_complement") {
      return;
    }
    auto origin = make_origin_gate(name, rec);
    if (!origin) {
      return;
    }
    gate->mef_origin_ptr(origin.get());
    out.origin_gates.push_back(std::move(origin));
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
    if (!std::isfinite(info.p) || info.p < 0.0 || info.p > 1.0) {
      throw RuntimeError("pdag_from_circuit: SAMPLE '" + info.name + "' event_id " +
                         std::to_string(info.event_id) + " has probability outside [0, 1]");
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
                         std::uint32_t k, std::uint32_t output) -> std::shared_ptr<scram::core::Gate> {
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
        add_arg(graph, inner, b, inv);
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
        add_arg(graph, atleast, b, inv);
        add_arg(graph, atleast_next, b, inv);
      }
      auto not_next = make_gate(graph, scram::core::kNot);
      not_next->AddArg(atleast_next, false);
      auto both = make_gate(graph, scram::core::kAnd);
      both->AddArg(atleast, false);
      both->AddArg(not_next, false);
      return both;
    }

    auto g = place_gate(connective_for(opcode), output);
    if (opcode == pcfb::OpCode::BITWISE_ATLEAST) {
      g->min_number(static_cast<int>(k));
    }
    for (const auto &[b, inv] : args) {
      add_arg(graph, g, b, inv);
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
      case pcfb::OpCode::BITWISE_ONES:
      case pcfb::OpCode::BITWISE_ZEROS: {
        const bool is_true = op->opcode() == pcfb::OpCode::BITWISE_ONES;
        const std::string name = (op->name() && op->name()->size()) ? op->name()->str() : std::string();
        TensorBinding constant;
        constant.is_constant = true;
        constant.constant_true = is_true;
        // Anonymous ONES ("true") stay as the PDAG Constant. Named fills are
        // constant-folded gates and must remain addressable DAG nodes.
        if (name.empty() || name == "true") {
          bind_output(output, std::move(constant));
          break;
        }
        auto g = place_gate(scram::core::kNull, output);
        add_arg(graph, g, constant, false);
        attach_origin(g, op, output);
        TensorBinding b;
        b.gate = std::move(g);
        bind_output(output, std::move(b));
        break;
      }
      default: {
        const auto args = collect_args(op);
        const std::uint32_t k =
            (op->args_type() == pcfb::OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k()
                                                                                  : 0;
        TensorBinding b;
        b.gate = emit_simple(op->opcode(), args, k, output);
        attach_origin(b.gate, op, output);
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
      if (qi == 0) {
        auto wrap = make_gate(graph, scram::core::kNull);
        wrap->AddArg(binding.variable, false);
        rq.node = wrap;
        rq.node_index = wrap->index();
      } else {
        rq.node = binding.variable;
        rq.node_index = binding.variable->index();
      }
    } else if (binding.is_constant) {
      auto wrap = make_gate(graph, scram::core::kNull);
      add_arg(graph, wrap, binding, false);
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
  graph->root(root_gate);
  if (out.meta.target.empty() && !out.queries.empty()) {
    out.meta.target = out.queries.front().name;
  }
  if (out.meta.target_name.empty()) {
    out.meta.target_name = out.meta.target;
  }
  if (out.meta.model_name.empty()) {
    out.meta.model_name = out.meta.target_name;
  }
  return out;
}

namespace {

void collect_named_nodes(const std::shared_ptr<scram::core::Gate> &gate,
                         std::unordered_map<int, std::shared_ptr<scram::core::Node>> &nodes) {
  if (!gate || !nodes.emplace(gate->index(), gate).second) {
    return;
  }
  for (const auto &arg : gate->args<scram::core::Gate>()) {
    collect_named_nodes(arg.second, nodes);
  }
  for (const auto &arg : gate->args<scram::core::Variable>()) {
    nodes.emplace(arg.second->index(), arg.second);
  }
}

bool origin_matches(std::string_view want, const scram::mef::Id &id) {
  return id.id() == want || id.name() == want || id.full_path() == want;
}

}  // namespace

std::vector<int> Session::resolve_watch(const ReconstructedPdag &reconstructed,
                                        const std::vector<std::string> &names) {
  if (names.empty()) {
    return {};
  }
  if (!reconstructed.graph || !reconstructed.graph->root_ptr()) {
    throw RuntimeError("watch: reconstructed PDAG has no root");
  }
  std::unordered_map<int, std::shared_ptr<scram::core::Node>> nodes;
  collect_named_nodes(reconstructed.graph->root_ptr(), nodes);
  std::vector<int> indices;
  indices.reserve(names.size());
  for (const auto &want : names) {
    if (want.empty()) {
      continue;
    }
    std::optional<int> found;
    for (const auto &[index, node] : nodes) {
      if (std::to_string(index) == want) {
        found = index;
        break;
      }
      if (auto gate = std::dynamic_pointer_cast<scram::core::Gate>(node)) {
        if (const auto *origin = gate->mef_origin_ptr(); origin && origin_matches(want, *origin)) {
          found = index;
          break;
        }
      } else if (auto var = std::dynamic_pointer_cast<scram::core::Variable>(node)) {
        const auto *event = reconstructed.graph->basic_events()[var->index()];
        if (event && origin_matches(want, *event)) {
          found = index;
          break;
        }
      }
    }
    if (!found) {
      for (const auto &q : reconstructed.queries) {
        if (q.name == want) {
          found = q.node_index;
          break;
        }
      }
    }
    if (!found) {
      throw RuntimeError("watched node not in reconstructed PDAG: " + want);
    }
    indices.push_back(*found);
  }
  return indices;
}

}  // namespace fuzzybits::runtime
