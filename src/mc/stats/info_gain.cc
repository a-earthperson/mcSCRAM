#include "mc/stats/info_gain.h"

#include <cmath>
#include <limits>

#include <boost/math/special_functions/digamma.hpp>

namespace scram::mc::stats {
namespace {

[[nodiscard]] double ln_beta(const double a, const double b)
{
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

} // namespace

double beta_entropy_nats(const double alpha, const double beta)
{
    using boost::math::digamma;
    const double lnB = ln_beta(alpha, beta);
    const double term1 = (alpha - 1.0) * digamma(alpha);
    const double term2 = (beta - 1.0) * digamma(beta);
    const double term3 = (alpha + beta - 2.0) * digamma(alpha + beta);
    return lnB - term1 - term2 + term3;
}

double beta_entropy_bits(const double alpha, const double beta)
{
    static constexpr double LN2 = 0.693147180559945309417232121458176568;
    return beta_entropy_nats(alpha, beta) / LN2;
}

double information_gain_bits(const double alpha_prev,
                             const double beta_prev,
                             const double alpha_new,
                             const double beta_new)
{
    return beta_entropy_bits(alpha_prev, beta_prev) - beta_entropy_bits(alpha_new, beta_new);
}

double info_gain_bits_per_second(const double alpha_prev,
                                 const double beta_prev,
                                 const double alpha_new,
                                 const double beta_new,
                                 const double seconds)
{
    const double ig = information_gain_bits(alpha_prev, beta_prev, alpha_new, beta_new);
    return (seconds > 0.0) ? ig / seconds : std::numeric_limits<double>::quiet_NaN();
}

double info_gain::add_batch(const std::size_t successes, const std::size_t failures)
{
    const double alpha_prev = alpha_;
    const double beta_prev = beta_;

    alpha_ += static_cast<double>(successes);
    beta_ += static_cast<double>(failures);

    last_bits_ = information_gain_bits(alpha_prev, beta_prev, alpha_, beta_);
    cumulative_bits_ += last_bits_;
    return last_bits_;
}

} // namespace scram::mc::stats
