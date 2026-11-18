#include "mc/scheduler/convergence_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace scram::mc::stats {
namespace detail {

using math_policy = boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::errno_on_error>,
    boost::math::policies::pole_error<boost::math::policies::errno_on_error>,
    boost::math::policies::overflow_error<boost::math::policies::errno_on_error>,
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::errno_on_error>>;

using beta_dist = boost::math::beta_distribution<double, math_policy>;
using normal_dist = boost::math::normal_distribution<double, math_policy>;

namespace {

void mark_not_converged(mc::stats::tally &t, const double rel_margin_error)
{
    const double p_hat = std::max(t.mean, DELTA_EPSILON);
    t.linear.epsilon = std::numeric_limits<double>::infinity();
    t.log10.epsilon = std::numeric_limits<double>::infinity();
    t.linear.target_epsilon = rel_margin_error * p_hat;
    t.log10.target_epsilon = rel_margin_error;
    t.linear.target_trials = t.total_bits;
    t.log10.target_trials = t.total_bits;
}

void store_bayes_interval(mc::stats::tally &t,
                          const double lower,
                          const double upper,
                          const double p_hat,
                          const double rel_margin_error,
                          const double z)
{
    const double lo = std::clamp(lower, 0.0, 1.0);
    const double hi = std::clamp(upper, 0.0, 1.0);
    const double eps_linear = 0.5 * (hi - lo);
    const double p = std::max(p_hat, DELTA_EPSILON);

    t.linear.epsilon = eps_linear;
    t.log10.epsilon = eps_linear / (p * std::log(10.0));

    const double target_eps_linear = rel_margin_error * p;
    t.linear.target_epsilon = target_eps_linear;
    t.log10.target_epsilon = rel_margin_error;

    const double var_term = p * (1.0 - p);
    const double rhs_lin = (z * z * var_term) / (target_eps_linear * target_eps_linear);
    const double rhs_log =
        (z * z * (1.0 - p)) /
        (p * rel_margin_error * rel_margin_error * std::pow(std::log(10.0), 2));

    constexpr double correction = 0.5 + 0.5 + 1.0;
    const auto N_req_linear =
        static_cast<std::size_t>(std::ceil(std::max(0.0, rhs_lin - correction)));
    const auto N_req_log10 =
        static_cast<std::size_t>(std::ceil(std::max(0.0, rhs_log - correction)));

    t.linear.target_trials = std::max<std::size_t>(N_req_linear, t.total_bits);
    t.log10.target_trials = std::max<std::size_t>(N_req_log10, t.total_bits);
}

} // namespace

// Boost's incomplete-beta series is stable for moderate β; above this the
// series can fail to converge and the O(1/√β) approximation is used instead.
// n = 1e6 → β = n + 1/2 still converges in Boost; the series starts failing
// for β ≳ 2e6 in the far tails under errno_on_error.
constexpr double kBoostStableBetaMax = 2.0e6;

double beta_half_quantile_approx(const double p, const double beta)
{
    const double u = std::clamp(p, 0.0, 1.0);
    if (u <= 0.0) {
        return 0.0;
    }
    if (u >= 1.0) {
        return 1.0;
    }
    if (beta <= 0.5 + 1.0e-15) {
        const double s = std::sin(0.5 * std::numbers::pi * u);
        return s * s;
    }
    const double g = std::pow(boost::math::erf_inv(u, math_policy{}), 2.0);
    if (!std::isfinite(g)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return g / (g + beta);
}

double beta_half_quantile(const double p, const double beta)
{
    const double u = std::clamp(p, 0.0, 1.0);
    if (u <= 0.0) {
        return 0.0;
    }
    if (u >= 1.0) {
        return 1.0;
    }
    if (beta <= 0.5 + 1.0e-15) {
        const double s = std::sin(0.5 * std::numbers::pi * u);
        return s * s;
    }
    if (beta <= kBoostStableBetaMax) {
        try {
            const beta_dist dist(0.5, beta);
            const double q = boost::math::quantile(dist, u);
            if (std::isfinite(q)) {
                return std::clamp(q, 0.0, 1.0);
            }
        } catch (const boost::math::evaluation_error &) {
        } catch (const std::exception &) {
        }
    }
    return beta_half_quantile_approx(p, beta);
}

namespace {

bool closed_form_boundary_interval(const std::size_t successes,
                                   const std::size_t n,
                                   const double tail,
                                   double &lower,
                                   double &upper)
{
    if (n == 0) {
        lower = beta_half_quantile(tail, 0.5);
        upper = beta_half_quantile(1.0 - tail, 0.5);
        return std::isfinite(lower) && std::isfinite(upper) && upper >= lower;
    }
    const double beta = static_cast<double>(n) + 0.5;
    if (successes == 0) {
        lower = beta_half_quantile(tail, beta);
        upper = beta_half_quantile(1.0 - tail, beta);
    } else if (successes == n) {
        lower = 1.0 - beta_half_quantile(1.0 - tail, beta);
        upper = 1.0 - beta_half_quantile(tail, beta);
    } else {
        return false;
    }
    return std::isfinite(lower) && std::isfinite(upper) && upper >= lower;
}

} // namespace

void update_wald(mc::stats::tally &t, const double rel_margin_error, const double z)
{
    mc::stats::tally::compute_moments(t);

    const double eps_linear = half_width(t, z);
    const double p_hat = std::max(t.mean, DELTA_EPSILON);
    const double target_eps = rel_margin_error * p_hat;
    const std::size_t N_req_linear =
        required_trials_from_normal_quantile_two_sided(p_hat, target_eps, z);

    t.linear.epsilon = eps_linear;
    t.linear.target_epsilon = target_eps;
    t.linear.target_trials = N_req_linear;

    const double eps_log10 = half_width_log10(t, z);
    const double target_eps_log10 = rel_margin_error;
    const std::size_t N_req_log10 =
        required_trials_log10_from_normal_quantile_two_sided(p_hat, target_eps_log10, z);

    t.log10.epsilon = eps_log10;
    t.log10.target_epsilon = target_eps_log10;
    t.log10.target_trials = N_req_log10;
}

void update_bayes(mc::stats::tally &t, const double rel_margin_error, const double z)
{
    mc::stats::tally::compute_moments(t);

    constexpr double alpha0 = 0.5;
    constexpr double beta0 = 0.5;

    const std::size_t successes = t.num_one_bits;
    const std::size_t n = t.total_bits;
    const std::size_t failures = n - successes;

    const double alpha = alpha0 + static_cast<double>(successes);
    const double beta = beta0 + static_cast<double>(failures);
    const double p_hat = alpha / (alpha + beta);

    double cred_level = 0.0;
    try {
        const normal_dist Zdist(0.0, 1.0);
        const double Phi_z = boost::math::cdf(Zdist, z);
        cred_level = std::clamp(2.0 * Phi_z - 1.0, 0.0, 1.0);
    } catch (const boost::math::evaluation_error &) {
        mark_not_converged(t, rel_margin_error);
        return;
    } catch (const std::exception &) {
        mark_not_converged(t, rel_margin_error);
        return;
    }
    if (!std::isfinite(cred_level)) {
        mark_not_converged(t, rel_margin_error);
        return;
    }

    const double tail = (1.0 - cred_level) / 2.0;

    double lower = std::numeric_limits<double>::quiet_NaN();
    double upper = std::numeric_limits<double>::quiet_NaN();
    bool have_interval = false;

    // Zero / all-one hit counts: incomplete-beta series diverge for large n.
    // Use beta_half_quantile (Boost for moderate n, O(1/√n) erf/mean approx above).
    if (successes == 0 || successes == n) {
        have_interval = closed_form_boundary_interval(successes, n, tail, lower, upper);
    }

    if (!have_interval) {
        try {
            const beta_dist post(alpha, beta);
            lower = boost::math::quantile(post, tail);
            upper = boost::math::quantile(post, 1.0 - tail);
            have_interval = std::isfinite(lower) && std::isfinite(upper) && upper >= lower;
        } catch (const boost::math::evaluation_error &) {
            have_interval = false;
        } catch (const std::exception &) {
            have_interval = false;
        }
    }

    if (!have_interval && (successes == 0 || successes == n)) {
        have_interval = closed_form_boundary_interval(successes, n, tail, lower, upper);
    }

    if (!have_interval) {
        mark_not_converged(t, rel_margin_error);
        return;
    }

    store_bayes_interval(t, lower, upper, p_hat, rel_margin_error, z);
}

} // namespace detail
} // namespace scram::mc::stats
