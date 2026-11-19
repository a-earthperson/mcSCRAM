#pragma once

#include "mc/prng/state128.h"

#include <cstdint>

namespace scram::mc::prng {

/// Host-computed 64-bit Bernoulli threshold, stored as two uint32 words plus
/// an `always_true` flag for p = 1 (`T = 2^64 − 1` is one ULP short of a sure
/// hit, so p = 1 must not go through the compare).
struct bernoulli_threshold {
    std::uint32_t hi{};
    std::uint32_t lo{};
    std::uint32_t always_true{};
};

/// High-word PRNG generations occupy `[0, num_generations)`. Tie-breaker
/// (low-word) generations occupy `[kLowWordGenerationBase, …)`. The Philox
/// seed's `x[3]` leaves 6 bits for the generation counter (0..63), so a base
/// of 32 supports up to 128-bit packs without colliding with the high-word
/// stream. Streams stay independent: low words are a distinct counter offset,
/// never a reuse of the high-word Philox output.
inline constexpr std::uint32_t kLowWordGenerationBase = 32;
inline constexpr std::uint32_t kBernoulliBitsPerGeneration = 4;
inline constexpr std::uint32_t kGenerationFieldModulus = 64;

/// `num_generations = sizeof(bitpack) * 8 / 4`. High-word gens are
/// `[0, num_generations)`; low-word gens are
/// `[kLowWordGenerationBase, kLowWordGenerationBase + num_generations)`.
/// Both ranges must be disjoint and fit in the 6-bit Philox generation field.
template <typename bitpack_t_>
struct tie_stream_independence {
    static constexpr std::uint32_t num_generations =
        static_cast<std::uint32_t>(sizeof(bitpack_t_) * 8 / kBernoulliBitsPerGeneration);
    static_assert(num_generations <= kLowWordGenerationBase,
                  "high-word generations must not overlap the low-word stream");
    static_assert(kLowWordGenerationBase + num_generations <= kGenerationFieldModulus,
                  "low-word generations must fit in the 6-bit generation field");
};

namespace detail {
/// Force the bounds at include time for every supported pack width, not only
/// the currently instantiated `bitpack_t_`.
inline constexpr tie_stream_independence<std::uint8_t> kTieWidth8{};
inline constexpr tie_stream_independence<std::uint16_t> kTieWidth16{};
inline constexpr tie_stream_independence<std::uint32_t> kTieWidth32{};
inline constexpr tie_stream_independence<std::uint64_t> kTieWidth64{};
#if defined(__SIZEOF_INT128__)
inline constexpr tie_stream_independence<unsigned __int128> kTieWidth128{};
#endif
}  // namespace detail

/// Exact two-word test `u < T ⇔ u_hi < T_hi ∨ (u_hi == T_hi ∧ u_lo < T_lo)`.
[[gnu::always_inline]] static inline bool
less_than_threshold(const std::uint32_t u_hi, const std::uint32_t u_lo,
                    const bernoulli_threshold &t) {
    if (t.always_true != 0) {
        return true;
    }
    return u_hi < t.hi || (u_hi == t.hi && u_lo < t.lo);
}

/// The 32-bit sampler's threshold: `T_hi`, or `2^32` for `always_true`, so
/// that `(u64)u_hi < high_word_threshold(t)` is the old compare verbatim and
/// `(u64)u_hi == high_word_threshold(t)` can never hold for `always_true`.
[[gnu::always_inline]] static inline std::uint64_t high_word_threshold(const bernoulli_threshold &t) {
    return t.always_true != 0 ? (std::uint64_t{1} << 32) : static_cast<std::uint64_t>(t.hi);
}

/// High-word-only pack: `(u64)u_hi < T_hi64`. Matches the old 32-bit
/// sampler (ties are misses). The rare-path helper then ORs in `u_lo < T_lo`.
template <typename bitpack_t_>
[[gnu::always_inline]] static inline bitpack_t_
pack_four_bits_hi(const state128 &hi, const std::uint64_t t_hi64, const std::uint32_t bit_offset) {
    using b = bitpack_t_;
    bitpack_t_ out_bits = b(0);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const bool hit = static_cast<std::uint64_t>(hi.x[lane]) < t_hi64;
        out_bits |= (hit ? b(1) : b(0)) << bitpack_t_(bit_offset + static_cast<std::uint32_t>(lane));
    }
    return out_bits;
}

/// Pack four Bernoulli bits from four high words. `lo` is consulted only on a
/// high-word tie; callers skip generating it when no lane ties.
template <typename bitpack_t_>
[[gnu::always_inline]] static inline bitpack_t_
pack_four_bits(const state128 &hi, const state128 &lo,
               const bernoulli_threshold &t, const std::uint32_t bit_offset) {
    using b = bitpack_t_;
    bitpack_t_ out_bits = b(0);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const bool hit = less_than_threshold(hi.x[lane], lo.x[lane], t);
        out_bits |= (hit ? b(1) : b(0)) << bitpack_t_(bit_offset + static_cast<std::uint32_t>(lane));
    }
    return out_bits;
}

[[gnu::always_inline]] static inline bool any_high_word_tie(const state128 &hi,
                                                            const bernoulli_threshold &t) {
    if (t.always_true != 0) {
        return false;
    }
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        if (hi.x[lane] == t.hi) {
            return true;
        }
    }
    return false;
}

/// Branch-free `∃ lane: (u64)u_hi == T_hi64` (never true for `always_true`).
/// Must stay free of control flow so the host JIT keeps the sampler loop
/// vectorized.
[[gnu::always_inline]] static inline bool high_word_tie_any(const state128 &hi, const std::uint64_t t_hi64) {
    return (static_cast<std::uint64_t>(hi.x[0]) == t_hi64) | (static_cast<std::uint64_t>(hi.x[1]) == t_hi64) |
           (static_cast<std::uint64_t>(hi.x[2]) == t_hi64) | (static_cast<std::uint64_t>(hi.x[3]) == t_hi64);
}

}  // namespace scram::mc::prng
