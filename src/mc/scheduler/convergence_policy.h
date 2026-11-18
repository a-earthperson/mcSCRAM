#pragma once

#include "mc/stats/ci_utils.h"
#include "mc/stats/tally.h"

namespace scram::mc::stats {

// -----------------------------------------------------------------------------
//  Tag types that identify the statistical strategy at *compile-time*.
// -----------------------------------------------------------------------------
struct wald_policy  { };   // classical Wald (normal approximation) interval
struct bayes_policy { };   // Beta–posterior credible interval (Jeffreys prior)

// -----------------------------------------------------------------------------
//  Internal helpers – implementation lives in the host-only .cc so that
//  Boost.Math is never parsed in a device-compiled translation unit.
// -----------------------------------------------------------------------------
namespace detail {

void update_wald(mc::stats::tally &t, double rel_margin_error, double z);
void update_bayes(mc::stats::tally &t, double rel_margin_error, double z);

/// Quantile of Beta(1/2, β).  Uses Boost.Math's inverse incomplete beta while
/// β is in the series-stable range, and `beta_half_quantile_approx` above it.
double beta_half_quantile(double p, double beta);

/// Large-n approximation of the Beta(1/2, β) quantile.
///
/// Gamma(1/2) has CDF erf(√x), so the inverse is [erf⁻¹(u)]² — that part is
/// exact.  The Beta(1/2, β) = X / (X + Y) representation then *replaces*
/// Y ~ Gamma(β) by its mean β, yielding x_u / (x_u + β).  This is not a
/// closed form of the Beta quantile; the relative error is O(1/√β).
/// β = 1/2 recovers the arcsine law exactly via the trigonometric form.
double beta_half_quantile_approx(double p, double beta);

} // namespace detail

// -----------------------------------------------------------------------------
//  Primary template – intentionally undefined.  Only the specializations below
//  are ever instantiated; attempting to use an unknown Policy triggers a clear
//  compiler error.
// -----------------------------------------------------------------------------
template <class Policy> struct ConvergencePolicy; // undefined

// Wald specialization ---------------------------------------------------------------------------
template <> struct ConvergencePolicy<wald_policy> {
    static void update(mc::stats::tally &t, const double rel_margin_error, const double z)
    {
        detail::update_wald(t, rel_margin_error, z);
    }
};

// Bayesian specialization -----------------------------------------------------------------------
template <> struct ConvergencePolicy<bayes_policy> {
    static void update(mc::stats::tally &t, const double rel_margin_error, const double z)
    {
        detail::update_bayes(t, rel_margin_error, z);
    }
};

// Convenience wrapper – deduces Policy from template argument -----------------------------------
template <class Policy>
inline void update_convergence(mc::stats::tally &t,
                               const double rel_margin_error,
                               const double z)
{
    ConvergencePolicy<Policy>::update(t, rel_margin_error, z);
}

} // namespace scram::mc::stats
