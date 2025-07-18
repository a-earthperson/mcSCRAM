#pragma once
#include "mc/prng/state128.h"

#include <cstdint>
namespace scram::mc::prng::xorshift {

// =====================
//  Lightweight PRNG
// =====================
// The following xorshift128 implementation is a much cheaper, latency-oriented
// alternative to the full Philox engine.  It is **not** intended for
// cryptographic use – only for fast Monte-Carlo sampling where statistical
// robustness requirements are moderate.
//
// We expose exactly the same public interface (`sample`) as the Philox
// variant but overload it for a different `state` type so existing call
// sites can switch PRNGs simply by changing the seed struct they pass in.
// ---------------------------------------------------------------------
[[gnu::always_inline]] static inline uint32_t next(state128 &s) {
    /* Correct Marsaglia xorshift128 (32-bit output).
     * Algorithm reference: Marsaglia, "Xorshift RNGs" (2003).
     * State transition:
     *   t  = x3
     *   t ^= t << 11;
     *   t ^= t >> 8;
     *   x3 = x2; x2 = x1; x1 = x0;
     *   x0 ^= x0 >> 19;
     *   x0 ^= t;
     * Output is the new x0.
     */

    uint32_t t   = s.x[3];       // use highest word for scrambling (was x[0] – bug)
    t ^= t << 11;
    t ^= t >> 8;

    const uint32_t s0 = s.x[0];  // preserve original x0 for later use

    /* Rotate the register contents downward */
    s.x[3] = s.x[2];
    s.x[2] = s.x[1];
    s.x[1] = s0;

    /* Final scrambling step */
    s.x[0] = s0 ^ (s0 >> 19) ^ t;

    return s.x[0];
}

[[gnu::always_inline]] static void generate(const state128 *seeds, state128 *results, const uint8_t generation) {
    /* Copy the seed so we can mutate locally without touching the caller’s
     * state.  Introduce `generation` via a simple Weyl step on the last
     * word – cheap decorrelation that avoids a loop over `generation`.
     */
    state128 local = *seeds;
    local.x[3] += 0x9E3779B9u * generation; // golden-ratio Weyl increment

    /* Produce four 32-bit outputs (one full 128-bit block). */
    results->x[0] = prng::xorshift::next(local);
    results->x[1] = prng::xorshift::next(local);
    results->x[2] = prng::xorshift::next(local);
    results->x[3] = prng::xorshift::next(local);
}

/**
 * @brief Bernoulli-sample four bits using the lightweight xorshift128 PRNG.
 *
 * Mirrors exactly the semantics of the Philox `sample` overload
 * but accepts an `xorshift128_state` pointer, enabling the caller
 * to switch RNGs simply by choosing which seed type to pass.
 */
template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ sample(const state128 *seeds, const std::uint64_t threshold,
                                                const std::uint32_t generation) {
    prng::state128 results;
    prng::xorshift::generate(seeds, &results, static_cast<uint8_t>(generation));

    static constexpr std::uint32_t bernoulli_bits_per_generation = 4;
    const std::uint32_t bernoulli_bits_offset = bernoulli_bits_per_generation * generation;

    using b = bitpack_t_;
    bitpack_t_ out_bits = b(0);
    out_bits |= (static_cast<std::uint64_t>(results.x[0]) < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 0);
    out_bits |= (static_cast<std::uint64_t>(results.x[1]) < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 1);
    out_bits |= (static_cast<std::uint64_t>(results.x[2]) < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 2);
    out_bits |= (static_cast<std::uint64_t>(results.x[3]) < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 3);

    return out_bits;
}

template <typename bitpack_t_>
[[gnu::always_inline]] static bitpack_t_ pack_bernoulli_draws(const state128 &seed_base, const std::uint64_t p_threshold) {
    static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
    static constexpr std::uint8_t num_generations = sizeof(bitpack_t_) * 8 / bernoulli_bits_per_generation;

    bitpack_t_ bitpacked_sample = bitpack_t_(0);
    #pragma unroll
    for (std::uint32_t i = 0; i < num_generations; ++i) {
        const bitpack_t_ four_bits = prng::xorshift::sample<bitpack_t_>(&seed_base, p_threshold, i);
        bitpacked_sample |= four_bits;
    }
    return bitpacked_sample;
}
} // namespace scram::mc::prng::xorshift