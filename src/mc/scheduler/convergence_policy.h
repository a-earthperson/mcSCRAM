#pragma once

#include "../stats/ci_utils.h"
#include "../stats/tally.h"

namespace scram::mc::stats {

// -----------------------------------------------------------------------------
//  Tag types that identify the statistical strategy at *compile-time*.
// -----------------------------------------------------------------------------
struct wald_policy  { };   // classical Wald (normal approximation) interval
struct bayes_policy { };   // Beta–posterior credible interval (Jeffreys prior)

// -----------------------------------------------------------------------------
//  Internal helpers – implementation is hidden inside the detail namespace so
//  that we can specialize cleanly per-policy while exposing only one public
//  API:  ConvergencePolicy<Policy>::update(tally, rel_err, z_score)
// -----------------------------------------------------------------------------
namespace detail {

// Utility: update a tally with Wald (frequentist) formulas --------------------------------------
inline void update_wald(mc::stats::tally &t,
                        const double rel_margin_error,
                        const double z)
{
    // ------------------------------------------------------------------
    //  Ensure first- and second-order moments are up-to-date.
    // ------------------------------------------------------------------
    mc::stats::tally::compute_moments(t);

    // ------------------------------------------------------------------
    //  Linear space (probability)
    // ------------------------------------------------------------------
    const double eps_linear = half_width(t, z);                // current half-width ε
    const double p_hat      = std::max(t.mean, DELTA_EPSILON); // guard against p→0
    const double target_eps = rel_margin_error * p_hat;        // desired ε
    const std::size_t N_req_linear =
        required_trials_from_normal_quantile_two_sided(p_hat, target_eps, z);

    t.linear.epsilon        = eps_linear;
    t.linear.target_epsilon = target_eps;
    t.linear.target_trials  = N_req_linear;

    // ------------------------------------------------------------------
    //  Log-scaled probability (log10 p)
    // ------------------------------------------------------------------
    const double eps_log10  = half_width_log10(t, z);
    const double target_eps_log10 = rel_margin_error;          // currently fixed fraction
    const std::size_t N_req_log10 =
        required_trials_log10_from_normal_quantile_two_sided(p_hat, target_eps_log10, z);

    t.log10.epsilon         = eps_log10;
    t.log10.target_epsilon  = target_eps_log10;
    t.log10.target_trials   = N_req_log10;
}

// TODO: Complete Bayesian implementation.  We provide a placeholder that simply
//       falls back to the Wald update so that compilation succeeds.  The
//       signature is identical so that callers are oblivious to the strategy.
inline void update_bayes(mc::stats::tally &t,
                         const double rel_margin_error,
                         const double z)
{
    // *** Placeholder *** – call frequentist routine until Bayesian maths lands
    update_wald(t, rel_margin_error, z);
}

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