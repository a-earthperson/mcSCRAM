#pragma once

#include "pcfb/ir/circuit.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fuzzybits::compile {

inline constexpr std::uint32_t kMaxExactEvents = 30;

class ExactEvalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Exhaustive exact P(query) over independent SAMPLE events.
/// Constant SAMPLE (p ∈ {0,1} / always_true) are substituted, not enumerated.
/// Throws ExactEvalError if the circuit has more than kMaxExactEvents random events.
[[nodiscard]] double exact_probability(const pcfb::ir::Circuit &circuit, std::uint32_t query_index = 0);

[[nodiscard]] std::uint32_t random_sample_count(const pcfb::ir::Circuit &circuit);

}  // namespace fuzzybits::compile
