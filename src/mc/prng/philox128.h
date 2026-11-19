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

/// Out-of-line so the inlined high-word Philox does not pay the second
/// 10-round generate's registers on the ~1−2^-32 taken path.
[[gnu::noinline]] [[gnu::cold]]
static void generate_tie_low(const state128 *seeds, state128 *results, const uint8_t generation) {
    generate(seeds, results, generation);
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
        generate_tie_low(seeds, &lo, static_cast<uint8_t>(generation + kLowWordGenerationBase));
    }

    static constexpr std::uint32_t bernoulli_bits_per_generation = 4;
    const std::uint32_t bernoulli_bits_offset = bernoulli_bits_per_generation * generation;
    return pack_four_bits<bitpack_t_>(hi, lo, threshold, bernoulli_bits_offset);
}

template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ pack_bernoulli_draws(const state128 &seed_base,
                                                             const bernoulli_threshold &p_threshold) {
    static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
    static constexpr std::uint8_t num_generations = sizeof(bitpack_t_) * 8 / bernoulli_bits_per_generation;
    using tie_check_ = tie_stream_independence<bitpack_t_>;
    static_assert(tie_check_::num_generations == static_cast<std::uint32_t>(num_generations));

    // Fast path is unrolled and hi-word-only — the same shape as the old
    // 32-bit sampler (`T_hi == floor(p·2^32)`). The lo-word generate lives
    // in a second, rolled loop so it does not inflate the unrolled live
    // range (a noinline call here doubled kernel time; inlining it into the
    // unrolled loop hit 128 regs).
    bitpack_t_ bitpacked_sample = bitpack_t_(0);
    std::uint32_t tie_mask = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < num_generations; ++i) {
        prng::state128 hi{};
        generate(&seed_base, &hi, static_cast<uint8_t>(i));
        bitpacked_sample |= pack_four_bits_hi<bitpack_t_>(hi, p_threshold, i * bernoulli_bits_per_generation);
        if (any_high_word_tie(hi, p_threshold)) {
            tie_mask |= (1u << i);
        }
    }
    if (tie_mask != 0) [[unlikely]] {
        for (std::uint32_t i = 0; i < num_generations; ++i) {
            if ((tie_mask & (1u << i)) == 0) {
                continue;
            }
            prng::state128 hi{};
            prng::state128 lo{};
            generate(&seed_base, &hi, static_cast<uint8_t>(i));
            generate(&seed_base, &lo, static_cast<uint8_t>(i + kLowWordGenerationBase));
            bitpacked_sample |= pack_four_bits<bitpack_t_>(hi, lo, p_threshold, i * bernoulli_bits_per_generation);
        }
    }
    return bitpacked_sample;
}
} // namespace scram::mc::prng