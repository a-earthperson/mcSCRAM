#pragma once
#include <cstdint>

#include "mc/prng/bernoulli.h"
#include "mc/prng/state128.h"

namespace scram::mc::prng::philox {

[[gnu::always_inline]] static inline void round(state128 &counters, Vec2<uint32_t> &key) {
    static constexpr Vec2<uint32_t> PHILOX_M4x32 = {
        .A = 0xD2511F53u,
        .B = 0xCD9E8D57u,
    };
    // Split into high and low parts
    Vec2<uint32_t> hi{};

    const Vec2<uint32_t> lo = {
        .A = prng::mulhilo(PHILOX_M4x32.A, counters.x[0], &hi.A),
        .B = prng::mulhilo(PHILOX_M4x32.B, counters.x[2], &hi.B),
    };

    // Mix in the key
    counters.x[0] = hi.B ^ counters.x[1] ^ key.A;
    counters.x[1] = lo.B;
    counters.x[2] = hi.A ^ counters.x[3] ^ key.B;
    counters.x[3] = lo.A;

    static constexpr Vec2<uint32_t> PHILOX_W32 = {
        .A = 0x9E3779B9u,
        .B = 0xBB67AE85u,
    };

    // Bump the key
    key.A += PHILOX_W32.A;
    key.B += PHILOX_W32.B;
}

[[gnu::always_inline]] static void generate(const state128 *seeds, state128 *results, const uint8_t generation) {
    // Key as Vec2
    Vec2<uint32_t> key = {
        .A = 382307844u,
        .B = 293830103u,
    };

    // Counter
    state128 counters = *seeds;
    counters.x[3] += generation;

    #define PHILOX4x32_DEFAULT_ROUNDS 10
    #pragma unroll
    for (auto i = 0; i < PHILOX4x32_DEFAULT_ROUNDS; i++) {
        prng::philox::round(counters, key);
    }
    *results = counters;
}

/**
 * @brief Bernoulli-sample four bits using the Philox 4x32-10 counter-based PRNG.
 *
 * High words are the four Philox outputs at `generation`. On a high-word tie
 * (`u_hi == T_hi`, probability ≈ 2^-32) a second independent Philox block is
 * drawn at `generation + kLowWordGenerationBase` and compared as `u_lo < T_lo`.
 */
template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ sample(const state128 *seeds, const bernoulli_threshold &threshold,
                                                const std::uint32_t generation) {
    prng::state128 hi{};
    prng::philox::generate(seeds, &hi, static_cast<uint8_t>(generation));

    prng::state128 lo{};
    if (any_high_word_tie(hi, threshold)) [[unlikely]] {
        generate(seeds, &lo, static_cast<uint8_t>(generation + kLowWordGenerationBase));
    }

    static constexpr std::uint32_t bernoulli_bits_per_generation = 4;
    const std::uint32_t bernoulli_bits_offset = bernoulli_bits_per_generation * generation;
    return pack_four_bits<bitpack_t_>(hi, lo, threshold, bernoulli_bits_offset);
}

/**
 * @brief High-word-only pack: bit = `always_true ∨ u_hi < T_hi`, plus a flag
 * for `∃ draw: u_hi == T_hi`.
 *
 * Same shape as the 32-bit sampler (`T_hi == floor(p·2^32)`): unrolled, one
 * Philox block per generation, no control flow. Tied draws come back as
 * misses; `patch_high_word_ties` (run from a separate fix-up kernel on the
 * device) turns them into the exact two-word result.
 */
template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ pack_bernoulli_draws_hi(const state128 &seed_base,
                                                                const std::uint64_t t_hi64,
                                                                bool &any_tie) {
    static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
    static constexpr std::uint8_t num_generations = sizeof(bitpack_t_) * 8 / bernoulli_bits_per_generation;
    using tie_check_ = tie_stream_independence<bitpack_t_>;
    static_assert(tie_check_::num_generations == static_cast<std::uint32_t>(num_generations));

    bitpack_t_ bitpacked_sample = bitpack_t_(0);
    bool tie = false;
#pragma unroll
    for (std::uint32_t i = 0; i < num_generations; ++i) {
        prng::state128 hi{};
        generate(&seed_base, &hi, static_cast<uint8_t>(i));
        bitpacked_sample |= pack_four_bits_hi<bitpack_t_>(hi, t_hi64, i * bernoulli_bits_per_generation);
    }
    any_tie = tie;
    return bitpacked_sample;
}

/// OR the `u_lo < T_lo` bits of every tied draw into a high-word-only pack.
/// Rolled on purpose: this is the rare path and must not be unrolled into
/// the sampler's live range.
template <typename bitpack_t_>
static bitpack_t_ patch_high_word_ties(const state128 &seed_base, const bernoulli_threshold &p_threshold,
                                       bitpack_t_ bitpacked_sample) {
    static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
    static constexpr std::uint8_t num_generations = sizeof(bitpack_t_) * 8 / bernoulli_bits_per_generation;
    for (std::uint32_t i = 0; i < num_generations; ++i) {
        prng::state128 hi{};
        generate(&seed_base, &hi, static_cast<uint8_t>(i));
        if (!any_high_word_tie(hi, p_threshold)) {
            continue;
        }
        prng::state128 lo{};
        generate(&seed_base, &lo, static_cast<uint8_t>(i + kLowWordGenerationBase));
        bitpacked_sample |= pack_four_bits<bitpack_t_>(hi, lo, p_threshold, i * bernoulli_bits_per_generation);
    }
    return bitpacked_sample;
}

/// Exact two-word pack: `pack_bernoulli_draws_hi` followed by the tie patch.
template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ pack_bernoulli_draws(const state128 &seed_base,
                                                             const bernoulli_threshold &p_threshold) {
    bool any_tie = false;
    const bitpack_t_ hi_only =
        pack_bernoulli_draws_hi<bitpack_t_>(seed_base, high_word_threshold(p_threshold), any_tie);
    if (any_tie) [[unlikely]] {
        return patch_high_word_ties<bitpack_t_>(seed_base, p_threshold, hi_only);
    }
    return hi_only;
}
} // namespace scram::mc::prng