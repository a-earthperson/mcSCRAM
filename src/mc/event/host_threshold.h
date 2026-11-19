#pragma once

#include "mc/prng/bernoulli.h"
#include "pcfb/threshold.h"

#include <cstdint>

namespace scram::mc::prng {

/// Host-only. Exact `T = min(floor(p · 2^64), 2^64 − 1)` via pcfb, split into
/// the two uint32 words the device compare consumes. `p >= 1` sets
/// `always_true` so the device never relies on the saturated `2^64 − 1`
/// sentinel.
[[nodiscard]] inline bernoulli_threshold threshold_from_probability(double p) noexcept {
    bernoulli_threshold t{};
    t.always_true = (p >= 1.0) ? 1u : 0u;
    const auto T = pcfb::probability_to_threshold(p);
    t.hi = static_cast<std::uint32_t>(T >> 32U);
    t.lo = static_cast<std::uint32_t>(T);
    return t;
}

}  // namespace scram::mc::prng
