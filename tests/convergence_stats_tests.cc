#include "mc/scheduler/convergence_policy.h"
#include "mc/stats/ci_utils.h"
#include "mc/stats/tally.h"

#include <cmath>
#include <limits>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

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
