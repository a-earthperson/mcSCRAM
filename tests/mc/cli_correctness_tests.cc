#include <catch2/catch_test_macros.hpp>

#include "xml.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

#ifndef SCRAM_CLI_PATH
#error "SCRAM_CLI_PATH must be provided by CMake"
#endif

#ifndef PROJECT_SOURCE_DIR
#error "PROJECT_SOURCE_DIR must be provided by CMake"
#endif

namespace {

// Fixed trial count: large enough that 4·σ̂ is a meaningful bound, small
// enough that the OpenMP run of each model stays well under 60 s.
constexpr unsigned long long kNumTrials = 4'194'304ull;  // 2^22
constexpr double kZ = 4.0;

std::string shell_quote(const std::string &s) {
#ifdef _WIN32
  return "\"" + s + "\"";
#else
  return "'" + s + "'";
#endif
}

int run_scram(const std::filesystem::path &model, const std::filesystem::path &report) {
  std::ostringstream cmd;
#ifndef _WIN32
  cmd << "timeout 60 ";
#endif
  cmd << shell_quote(SCRAM_CLI_PATH)
      << " --monte-carlo --probability --pdag -c 2 --no-kn"
      << " -P wald -N " << kNumTrials
      << " -r " << kNumTrials
      << " --seed 372 --no-indent -V 0"
      << " -o " << shell_quote(report.string())
      << " " << shell_quote(model.string());
  return std::system(cmd.str().c_str());
}

double parse_p_hat(const std::filesystem::path &report) {
  const scram::xml::Document doc(report.string());
  const auto root = doc.root();
  const auto results = root.child("results");
  REQUIRE(results.has_value());

  for (const auto measure : results->children("measure")) {
    if (const auto mean = measure.child("mean")) {
      const auto value = mean->attribute<double>("value");
      REQUIRE(value.has_value());
      return *value;
    }
  }
  for (const auto sop : results->children("sum-of-products")) {
    const auto value = sop.attribute<double>("probability");
    if (value.has_value()) {
      return *value;
    }
  }
  FAIL("XML report has neither <measure><mean> nor <sum-of-products probability>");
  return 0.0;
}

void check_model(const char *name, const char *relative_path, const double p_ref) {
  const std::filesystem::path src_root{PROJECT_SOURCE_DIR};
  const auto input = src_root / relative_path;
  REQUIRE(std::filesystem::exists(input));

  const auto report = std::filesystem::temp_directory_path() /
                      ("fuzzybits-mc-" + std::string(name) + ".xml");
  std::filesystem::remove(report);

  const int status = run_scram(input, report);
  INFO("scram-cli wait status=" << status);
  REQUIRE(status == 0);
  REQUIRE(std::filesystem::exists(report));

  const double p_hat = parse_p_hat(report);
  const double sigma_hat =
      std::sqrt(p_hat * (1.0 - p_hat) / static_cast<double>(kNumTrials));
  const double abs_err = std::abs(p_hat - p_ref);
  INFO("model=" << name << " p_hat=" << p_hat << " p_ref=" << p_ref
                << " sigma_hat=" << sigma_hat << " |err|=" << abs_err
                << " bound=" << kZ * sigma_hat);
  REQUIRE(sigma_hat > 0.0);
  CHECK(abs_err <= kZ * sigma_hat);

  std::filesystem::remove(report);
}

}  // namespace

TEST_CASE("MC chinese vs BDD reference within 4 sigma", "[mc][chinese]") {
  check_model("chinese", "input/Aralia/chinese.xml", 1.17058e-3);
}

TEST_CASE("MC das9201 vs BDD reference within 4 sigma", "[mc][das9201]") {
  check_model("das9201", "input/Aralia/das9201.xml", 1.34237e-2);
}

TEST_CASE("MC isp9601 vs BDD reference within 4 sigma", "[mc][isp9601]") {
  check_model("isp9601", "input/Aralia/isp9601.xml", 5.71245e-2);
}

TEST_CASE("MC baobab2 vs BDD reference within 4 sigma", "[mc][baobab2]") {
  check_model("baobab2", "input/Aralia/baobab2.xml", 7.13018e-4);
}
