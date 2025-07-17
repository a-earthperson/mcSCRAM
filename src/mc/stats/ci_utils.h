#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace scram::mc::stats {

/**
 * Compute the two-sided Z-score that corresponds to a required confidence
 * level.  E.g. confidence = 0.95 → 1.95996, 0.99 → 2.57583.
 * The implementation uses the inverse complementary error function which is
 * available in C++17 ( <cmath> ).
 */
[[nodiscard]] inline double z_score(const double confidence) {
    // Clamp to a sensible open interval to avoid infinities / NaNs.
    const double p = std::clamp(confidence, 1e-12, 1.0 - 1e-12);
    // Two-sided: need quantile(1 − α/2) where α = 1-confidence
    const double alpha = 1.0 - p;
    const double q = 1.0 - alpha / 2.0;   // central CDF point

    // -----------------------------------------------------------------
    //  Inverse normal CDF (Acklam 2003).  Max error ~5e-16.
    // -----------------------------------------------------------------
    static constexpr double a[] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00};

    static constexpr double b[] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01};

    static constexpr double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00};

    static constexpr double d[] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00};

    // Define break-points.
    constexpr double plow  = 0.02425;
    constexpr double phigh = 1 - plow;

    double x;
    if (q < plow) {
        // Rational approximation for lower region
        const double u = std::sqrt(-2.0 * std::log(q));
        x = (((((c[0]*u + c[1])*u + c[2])*u + c[3])*u + c[4])*u + c[5]) /
            ((((d[0]*u + d[1])*u + d[2])*u + d[3])*u + 1.0);
    } else if (phigh < q) {
        // Rational approximation for upper region
        const double u = std::sqrt(-2.0 * std::log(1.0 - q));
        x = -(((((c[0]*u + c[1])*u + c[2])*u + c[3])*u + c[4])*u + c[5]) /
             ((((d[0]*u + d[1])*u + d[2])*u + d[3])*u + 1.0);
    } else {
        // Rational approximation for central region
        const double u = q - 0.5;
        const double t = u * u;
        x = (((((a[0]*t + a[1])*t + a[2])*t + a[3])*t + a[4])*t + a[5])*u /
            (((((b[0]*t + b[1])*t + b[2])*t + b[3])*t + b[4])*t + 1.0);
    }
    return x;
}

/**
 * Sample-size formula for a Bernoulli proportion.
 *   N ≥ z² · p(1-p) / ε²
 * where ε is the desired half-width (margin of error).
 */
[[nodiscard]] inline std::size_t required_trials(const double p,
                                                const double eps,
                                                const double confidence) {
    const double z  = z_score(confidence);
    const double pq = p * (1.0 - p);
    return static_cast<std::size_t>(std::ceil((z * z * pq) / (eps * eps)));
}

/**
 * Conservative upper bound obtained by plugging in p = 0.5 (maximum variance).
 */
[[nodiscard]] inline std::size_t worst_case_trials(const double eps,
                                                  const double confidence) {
    return required_trials(0.5, eps, confidence);
}

/**
 * Quick rule-of-thumb validity check for the normal approximation of the
 * sample proportion (np ≥ 10 and n(1-p) ≥ 10).
 */
[[nodiscard]] inline bool clt_ok(const std::size_t n, const double p) {
    return static_cast<double>(n) * p >= 10.0 && static_cast<double>(n) * (1.0 - p) >= 10.0;
}

} // namespace scram::mc::stats