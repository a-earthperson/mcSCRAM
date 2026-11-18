#pragma once

/*
 * @file info_gain.h
 * @brief Helper routines to compute Shannon information gain for a Bernoulli
 *        probability estimated with a Beta prior/posterior.
 *
 * The Monte-Carlo engine represents the unknown failure probability *p* with a
 * conjugate Beta(α, β) distribution.  After each batch of Bernoulli trials the
 * parameters are updated
 *
 *     α ← α + #successes,
 *     β ← β + #failures.
 *
 * The expected Shannon information gained from that batch is the *reduction of
 * entropy* of the Beta distribution,
 *
 *     I  =  H(Beta(α_prev, β_prev)) − H(Beta(α_new, β_new)) [bits].
 *
 * Boost.Math (digamma) is used only in the host-only .cc so that this header
 * is safe to include from device-compiled translation units.
 *
 * References
 * ----------
 *   H(Beta) = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)
 *   where  ψ  is the digamma function and  B  the Beta function.
 */

#include <cstddef>

namespace scram::mc::stats {

/**
 * Shannon entropy H[Beta(α,β)] in natural units (nats).
 * Preconditions: α>0, β>0.
 */
[[nodiscard]] double beta_entropy_nats(double alpha, double beta);

/**
 * Shannon entropy of Beta(α,β) in bits.
 */
[[nodiscard]] double beta_entropy_bits(double alpha, double beta);

/**
 * Expected information gained (bits) when moving from (α_prev,β_prev) to
 * (α_new,β_new).
 */
[[nodiscard]] double information_gain_bits(double alpha_prev,
                                           double beta_prev,
                                           double alpha_new,
                                           double beta_new);

/**
 * Information-gain rate (bits per second).
 * @param seconds  Wall-clock time elapsed.  Values ≤0 return NaN.
 */
[[nodiscard]] double info_gain_bits_per_second(double alpha_prev,
                                               double beta_prev,
                                               double alpha_new,
                                               double beta_new,
                                               double seconds);

/**
 * Simple accumulator that tracks the Beta parameters
 */
class info_gain {
public:
    explicit info_gain(const double alpha0 = 0.5, const double beta0 = 0.5)
        : alpha_(alpha0), beta_(beta0) {}

    /**
     * Register a batch of Bernoulli results and return the information gained
     * for this batch **in bits** (time-agnostic).
     */
    [[nodiscard]] double add_batch(std::size_t successes, std::size_t failures);

    [[nodiscard]] double alpha() const { return alpha_; }
    [[nodiscard]] double beta()  const { return beta_;  }
    [[nodiscard]] double cumulative_bits() const { return cumulative_bits_; }
    [[nodiscard]] double last_bits() const { return last_bits_; }

private:
    double alpha_;
    double beta_;
    double cumulative_bits_ = 0.0;
    double last_bits_ = 0.0;
};

} // namespace scram::mc::stats
