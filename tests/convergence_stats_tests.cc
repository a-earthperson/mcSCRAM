#include "mc/scheduler/convergence_policy.h"
#include "mc/stats/ci_utils.h"
#include "mc/stats/tally.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <vector>

#include <boost/math/distributions/beta.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using scram::mc::stats::bayes_policy;
using scram::mc::stats::tally;
using scram::mc::stats::update_convergence;
using scram::mc::stats::wald_policy;

namespace {

tally make_tally(const std::size_t k, const std::size_t n)
{
    tally t{};
    t.num_one_bits = k;
    t.total_bits = n;
    return t;
}

double z99()
{
    return scram::mc::stats::normal_quantile_two_sided(0.99);
}

} // namespace

TEST_CASE("bayes policy does not throw at k = 0 for small and huge n", "[convergence][bayes]")
{
    const double z = z99();
    const double delta = 0.001;

    for (const std::size_t n : {std::size_t{1}, std::size_t{100}, std::size_t{1'000'000},
                                static_cast<std::size_t>(1e12)}) {
        tally t = make_tally(0, n);
        REQUIRE_NOTHROW(update_convergence<bayes_policy>(t, delta, z));
        CHECK(t.mean == 0.0);
        CHECK(std::isfinite(t.linear.epsilon));
        CHECK(t.linear.epsilon >= 0.0);
        CHECK(t.linear.target_trials >= n);
    }
}

TEST_CASE("bayes policy does not throw at k = n for small and huge n", "[convergence][bayes]")
{
    const double z = z99();
    const double delta = 0.001;

    for (const std::size_t n : {std::size_t{1}, std::size_t{100}, std::size_t{1'000'000},
                                static_cast<std::size_t>(1e12)}) {
        tally t = make_tally(n, n);
        REQUIRE_NOTHROW(update_convergence<bayes_policy>(t, delta, z));
        CHECK(t.mean == 1.0);
        CHECK(std::isfinite(t.linear.epsilon));
        CHECK(t.linear.epsilon >= 0.0);
        CHECK(t.linear.target_trials >= n);
    }
}

TEST_CASE("bayes policy handles k = 1 interior counts", "[convergence][bayes]")
{
    const double z = z99();
    const double delta = 0.001;

    SECTION("modest n") {
        tally t = make_tally(1, 10'000);
        REQUIRE_NOTHROW(update_convergence<bayes_policy>(t, delta, z));
        CHECK(t.mean == 1.0 / 10'000.0);
        CHECK(t.linear.epsilon > 0.0);
        CHECK(std::isfinite(t.linear.epsilon));
    }

    SECTION("n = 1e12") {
        const auto n = static_cast<std::size_t>(1e12);
        tally t = make_tally(1, n);
        REQUIRE_NOTHROW(update_convergence<bayes_policy>(t, delta, z));
        // Interior k at this n may fall back to "not converged" if Boost cannot
        // evaluate the incomplete beta; either a finite half-width or +inf is
        // acceptable as long as the policy does not throw.
        CHECK((std::isfinite(t.linear.epsilon) ||
               t.linear.epsilon == std::numeric_limits<double>::infinity()));
    }
}

TEST_CASE("wald policy is defined at k = 0, 1, n", "[convergence][wald]")
{
    const double z = z99();
    const double delta = 0.001;

    SECTION("k = 0") {
        tally t = make_tally(0, 1'000'000);
        REQUIRE_NOTHROW(update_convergence<wald_policy>(t, delta, z));
        CHECK(t.mean == 0.0);
        CHECK(t.linear.epsilon == 0.0);
    }

    SECTION("k = 1") {
        tally t = make_tally(1, 1'000'000);
        REQUIRE_NOTHROW(update_convergence<wald_policy>(t, delta, z));
        CHECK(t.linear.epsilon > 0.0);
        CHECK(std::isfinite(t.linear.epsilon));
    }

    SECTION("k = n") {
        tally t = make_tally(1'000'000, 1'000'000);
        REQUIRE_NOTHROW(update_convergence<wald_policy>(t, delta, z));
        CHECK(t.mean == 1.0);
        CHECK(t.linear.epsilon == 0.0);
    }
}

TEST_CASE("beta_half_quantile_approx error vs Boost for n = 1e2..1e6",
          "[convergence][bayes][approx]")
{
    using scram::mc::stats::detail::beta_half_quantile;
    using scram::mc::stats::detail::beta_half_quantile_approx;
    using boost::math::beta_distribution;
    using boost::math::quantile;

    // Include the Jeffreys-prior two-sided tail used at z = 4
    // (Φ(4) ≈ 0.999968, cred ≈ 0.999936, tail ≈ 3.17e-5) plus a coarse grid.
    const std::vector<double> us{
        3.167e-5, 1e-4, 1e-3, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 1.0 - 1e-3,
        1.0 - 1e-4, 1.0 - 3.167e-5,
    };
    const std::vector<double> ns{1e2, 1e3, 1e4, 1e5, 1e6};

    for (const double n : ns) {
        const double beta = n + 0.5;
        double max_rel = 0.0;
        std::size_t compared = 0;
        const beta_distribution<double> dist(0.5, beta);
        for (const double u : us) {
            const double exact = quantile(dist, u);
            const double approx = beta_half_quantile_approx(u, beta);
            REQUIRE(std::isfinite(exact));
            REQUIRE(std::isfinite(approx));
            const double denom = std::max(std::abs(exact), 1e-18);
            const double rel = std::abs(approx - exact) / denom;
            max_rel = std::max(max_rel, rel);
            ++compared;

            // Production wrapper uses Boost for β ≤ 2e6 (covers n = 1e2..1e6).
            const double hybrid = beta_half_quantile(u, beta);
            CHECK(hybrid == Catch::Approx(exact).epsilon(1e-8).margin(1e-15));
        }
        INFO("n=" << n << " max_rel_error=" << max_rel << " compared=" << compared);
        // Relative error of the erf/mean approximation is O(1/√n).  At n = 100
        // this is tens of percent in the tails, so we only bound the scaled
        // error; the production path uses Boost for these n.
        CHECK(max_rel * std::sqrt(n) < 8.0);
        WARN("beta_half_quantile_approx n=" << n << " max relative error=" << max_rel);
    }
}
