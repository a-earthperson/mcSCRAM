#include "compile/pcfb_writer.h"
#include "runtime/runtime.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>

namespace {

std::string aralia(const char *name) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / name).string();
}

}  // namespace

// Runs as fzb_runtime_tests.metal_step_n; skipped without a Metal device.
TEST_CASE("Session::step(n) matches n single steps on chinese, on Metal (#171)",
          "[runtime][mc][bit-identity][issue-171]") {
  constexpr std::uint64_t kTrialsPerPass = 65536;
  fuzzybits::compile::Options opt;
  opt.compilation_level = 2;
  opt.expand_atleast_gates = true;
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);

  std::optional<fuzzybits::runtime::Session> batched;
  std::optional<fuzzybits::runtime::Session> single;
  try {
    batched.emplace(fuzzybits::runtime::Session::from_bytes(compiled->bytes, "metal", kTrialsPerPass));
    single.emplace(fuzzybits::runtime::Session::from_bytes(compiled->bytes, "metal", kTrialsPerPass));
  } catch (const std::exception &e) {
    SKIP("no Metal device: " << e.what());
  }

  const auto once = batched->step(3);
  fuzzybits::runtime::PassResult last;
  for (int i = 0; i < 3; ++i) {
    last = single->step();
  }

  REQUIRE(once.nodes.size() == last.nodes.size());
  for (const auto &node : once.nodes) {
    const auto match = std::find_if(last.nodes.begin(), last.nodes.end(),
                                    [&](const auto &other) { return other.index == node.index; });
    REQUIRE(match != last.nodes.end());
    CHECK(node.num_one_bits == match->num_one_bits);
    CHECK(node.total_bits == match->total_bits);
  }
}
