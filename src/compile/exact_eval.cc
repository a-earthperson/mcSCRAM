#include "compile/exact_eval.h"

#include "pcfb_generated.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fuzzybits::compile {
namespace {

bool eval_op(pcfb::OpCode opcode, const std::vector<std::uint8_t> &ins, std::uint32_t k) {
  auto fold_and = [&] {
    std::uint8_t v = 1;
    for (auto b : ins) {
      v = static_cast<std::uint8_t>(v & b);
    }
    return v;
  };
  auto fold_or = [&] {
    std::uint8_t v = 0;
    for (auto b : ins) {
      v = static_cast<std::uint8_t>(v | b);
    }
    return v;
  };
  auto fold_xor = [&] {
    std::uint8_t v = 0;
    for (auto b : ins) {
      v = static_cast<std::uint8_t>(v ^ b);
    }
    return v;
  };
  auto count = [&] {
    std::uint32_t n = 0;
    for (auto b : ins) {
      n += b;
    }
    return n;
  };

  switch (opcode) {
    case pcfb::OpCode::BITWISE_NO_OP:
      return ins.at(0) != 0;
    case pcfb::OpCode::BITWISE_NOT:
      return ins.at(0) == 0;
    case pcfb::OpCode::BITWISE_AND:
      return fold_and() != 0;
    case pcfb::OpCode::BITWISE_OR:
      return fold_or() != 0;
    case pcfb::OpCode::BITWISE_XOR:
      return fold_xor() != 0;
    case pcfb::OpCode::BITWISE_NAND:
      return fold_and() == 0;
    case pcfb::OpCode::BITWISE_NOR:
      return fold_or() == 0;
    case pcfb::OpCode::BITWISE_XNOR:
      return fold_xor() == 0;
    case pcfb::OpCode::BITWISE_ATLEAST:
      return count() >= k;
    case pcfb::OpCode::BITWISE_ATMOST:
      return count() <= k;
    case pcfb::OpCode::BITWISE_EXACTLY:
      return count() == k;
    case pcfb::OpCode::BITWISE_ZEROS:
      return false;
    case pcfb::OpCode::BITWISE_ONES:
      return true;
    default:
      throw ExactEvalError("exact evaluator: reserved opcode");
  }
}

struct RandomEvent {
  std::uint32_t tensor{};
  double p{};
};

void collect_events(const pcfb::ir::Circuit &circuit, std::vector<RandomEvent> *events,
                    std::vector<std::int8_t> *constant) {
  if (!circuit.valid() || !circuit.subcircuits() || circuit.subcircuits()->size() == 0) {
    throw ExactEvalError("exact evaluator: empty circuit");
  }
  const auto *sc = circuit.subcircuits()->Get(0);
  if (!sc || !sc->tensors() || !sc->operators() || !sc->queries()) {
    throw ExactEvalError("exact evaluator: missing subcircuit fields");
  }
  const auto n_tensors = sc->tensors()->size();
  constant->assign(n_tensors, -1);
  events->clear();
  const auto *ops = sc->operators();
  for (flatbuffers::uoffset_t i = 0; i < ops->size(); ++i) {
    const auto *op = ops->Get(i);
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    const auto *sa = op->args_as_SampleArgs();
    if (!sa) {
      throw ExactEvalError("exact evaluator: SAMPLE without SampleArgs");
    }
    if (sa->always_true() || sa->probability() == 1.0) {
      (*constant)[op->output()] = 1;
    } else if (sa->probability() == 0.0) {
      (*constant)[op->output()] = 0;
    } else {
      events->push_back({op->output(), sa->probability()});
    }
  }
}

}  // namespace

std::uint32_t random_sample_count(const pcfb::ir::Circuit &circuit) {
  std::vector<RandomEvent> events;
  std::vector<std::int8_t> constant;
  collect_events(circuit, &events, &constant);
  return static_cast<std::uint32_t>(events.size());
}

double exact_probability(const pcfb::ir::Circuit &circuit, std::uint32_t query_index) {
  std::vector<RandomEvent> events;
  std::vector<std::int8_t> constant;
  collect_events(circuit, &events, &constant);

  const auto *sc = circuit.subcircuits()->Get(0);
  if (query_index >= sc->queries()->size()) {
    throw ExactEvalError("exact evaluator: missing query");
  }
  if (events.size() > kMaxExactEvents) {
    throw ExactEvalError("exact evaluator: " + std::to_string(events.size()) +
                         " random events exceeds limit " + std::to_string(kMaxExactEvents));
  }

  const auto n_tensors = sc->tensors()->size();
  const auto *ops = sc->operators();
  const auto target = sc->queries()->Get(query_index)->target();
  const auto n_rand = events.size();
  const std::uint64_t n_assign = n_rand == 0 ? 1ull : (1ull << n_rand);

  double total = 0.0;
  double c = 0.0;  // Kahan compensation
  std::vector<std::uint8_t> value(n_tensors, 0);
  std::vector<std::uint8_t> ins;
  ins.reserve(16);

  for (std::uint64_t mask = 0; mask < n_assign; ++mask) {
    std::fill(value.begin(), value.end(), 0);
    double mass = 1.0;
    for (std::size_t e = 0; e < n_rand; ++e) {
      const bool bit = ((mask >> e) & 1ull) != 0;
      value[events[e].tensor] = bit ? 1 : 0;
      mass *= bit ? events[e].p : (1.0 - events[e].p);
    }
    for (std::uint32_t t = 0; t < n_tensors; ++t) {
      if (constant[t] >= 0) {
        value[t] = static_cast<std::uint8_t>(constant[t]);
      }
    }

    for (flatbuffers::uoffset_t i = 0; i < ops->size(); ++i) {
      const auto *op = ops->Get(i);
      if (!op || op->opcode() == pcfb::OpCode::SAMPLE) {
        continue;
      }
      ins.clear();
      if (op->inputs()) {
        for (auto id : *op->inputs()) {
          ins.push_back(value[id]);
        }
      }
      if (op->inverted_inputs()) {
        for (auto id : *op->inverted_inputs()) {
          ins.push_back(static_cast<std::uint8_t>(1u - value[id]));
        }
      }
      const std::uint32_t k =
          (op->args_type() == pcfb::OpArgs::KofNArgs && op->args_as_KofNArgs()) ? op->args_as_KofNArgs()->k() : 0;
      value[op->output()] = eval_op(op->opcode(), ins, k) ? 1 : 0;
    }
    if (value[target]) {
      const double y = mass - c;
      const double t = total + y;
      c = (t - total) - y;
      total = t;
    }
  }
  return total;
}

}  // namespace fuzzybits::compile
