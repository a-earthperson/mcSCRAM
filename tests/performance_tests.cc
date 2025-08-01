/*
 * Copyright (C) 2014-2018 Olzhas Rakhimov
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// Performance testing is done only if requested
// by activating disabled tests.
//
// To run the performance tests,
// supply "--gtest_also_run_disabled_tests" flag to GTest.
// The GTest filter may be applied to filter only performance tests.
// Different tests are compiled depending on the build type.
// Generally, debug or non-debug types are recognized.
//
// Reference performance values are taken
// from a computer with the following specs:
//
//   Proc         Core i7-2820QM
//   Ubuntu       16.04 64bit
//   GCC          5.4.0
//   Boost        1.58
//   TCMalloc     2.4
//
// The performance measurements are expected to have some random variation.
// Better as well as worse performance are reported
// as test failures to indicate the change.
//
// NOTE: Running all the tests may take considerable time.
// NOTE: Running tests several times is recommended
//       to take into account the random variation of time results.

#include "performance_tests.h"

#include "bdd.h"
#include "zbdd.h"

namespace scram::core::test {

// Regression check for performance assumptions of developers.
#ifndef NDEBUG
// Test for performance critical object sizes.
TEST_CASE("regression test BDD/ZBDD", "[object_size]") {
  // x86-64 platform.
  // 64-bit platform with alignment at 8-byte boundaries.
  CHECK(sizeof(WeakIntrusivePtr<Vertex<Ite>>) == 8);
  CHECK(sizeof(IntrusivePtr<Vertex<Ite>>) == 8);
  CHECK(sizeof(Vertex<Ite>) == 16);
  CHECK(sizeof(NonTerminal<Ite>) == 48);
  CHECK(sizeof(Ite) == 64);
  CHECK(sizeof(SetNode) == 56);
}
#endif

// Tests the performance of probability calculations.
TEST_CASE_METHOD(PerformanceTest, "probability", "[.perf]") {
  double p_time_std = 0.01;
  std::string input = "input/ThreeMotor/three_motor.xml";
  settings.probability_analysis(true);
  REQUIRE_NOTHROW(Analyze({input}));
  REQUIRE(ProbabilityCalculationTime() < p_time_std);
}

TEST_CASE_METHOD(PerformanceTest, "perf chinese", "[.perf]") {
  double mcs_time = 0.1;
  std::vector<std::string> input_files{
      "input/Chinese/chinese.xml", "input/Chinese/chinese-basic-events.xml"};
  settings.probability_analysis(false);
  REQUIRE_NOTHROW(Analyze(input_files));
  REQUIRE(ProductGenerationTime() < mcs_time);
}

TEST_CASE_METHOD(PerformanceTest, "perf 200Event", "[.perf]") {
  double mcs_time = 0.2;
  std::string input = "input/Autogenerated/200_event.xml";
  REQUIRE_NOTHROW(Analyze({input}));
  CHECK(NumOfProducts() == 15347);
  CHECK(ProductGenerationTime() < mcs_time);
}

TEST_CASE_METHOD(PerformanceTest, "perf Baobab1L7", "[.perf]") {
  double mcs_time = 1.8;
#ifdef NDEBUG
  mcs_time = 0.35;
#endif
  std::vector<std::string> input_files{"input/Baobab/baobab1.xml",
                                       "input/Baobab/baobab1-basic-events.xml"};
  settings.limit_order(7);
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 17432);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}

TEST_CASE_METHOD(PerformanceTest, "perf CEA9601_L4", "[.perf]") {
  double mcs_time = 7.7;
#ifdef NDEBUG
  mcs_time = 2.0;
#endif
  std::vector<std::string> input_files{
      "input/CEA9601/CEA9601.xml", "input/CEA9601/CEA9601-basic-events.xml"};
  settings.limit_order(4).algorithm("bdd");
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 54436);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}

#ifdef NDEBUG
TEST_CASE_METHOD(PerformanceTest, "perf CEA9601_L5", "[.perf]") {
  double mcs_time = 3.8;
  std::vector<std::string> input_files{
      "input/CEA9601/CEA9601.xml", "input/CEA9601/CEA9601-basic-events.xml"};
  settings.limit_order(5).algorithm("bdd");
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 1615876);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}

TEST_CASE_METHOD(PerformanceTest, "perf CEA9601_L3_ZBDD", "[.perf]") {
  double mcs_time = 1.5;
  std::vector<std::string> input_files{
      "input/CEA9601/CEA9601.xml", "input/CEA9601/CEA9601-basic-events.xml"};
  settings.limit_order(3).algorithm("zbdd");
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 1144);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}
#endif

TEST_CASE_METHOD(PerformanceTest, "perf Baobab2", "[.perf]") {
  double mcs_time = 0.1;
  std::vector<std::string> input_files{"input/Baobab/baobab2.xml",
                                       "input/Baobab/baobab2-basic-events.xml"};
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 4805);
  CHECK(ProductGenerationTime() < mcs_time);
}

TEST_CASE_METHOD(PerformanceTest, "perf Baobab1", "[.perf]") {
  double mcs_time = 1.9;
#ifdef NDEBUG
  mcs_time = 0.30;
#endif
  std::vector<std::string> input_files{"input/Baobab/baobab1.xml",
                                       "input/Baobab/baobab1-basic-events.xml"};
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 46188);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}

TEST_CASE_METHOD(PerformanceTest, "perf Baobab1_ZBDD", "[.perf]") {
  double mcs_time = 0.95;
#ifdef NDEBUG
  mcs_time = 0.15;
#endif
  std::vector<std::string> input_files{"input/Baobab/baobab1.xml",
                                       "input/Baobab/baobab1-basic-events.xml"};
  settings.algorithm("zbdd");
  REQUIRE_NOTHROW(Analyze(input_files));
  CHECK(NumOfProducts() == 46188);
  CHECK(ProductGenerationTime() == Approx(mcs_time).epsilon(delta));
}

}  // namespace scram::core::test
