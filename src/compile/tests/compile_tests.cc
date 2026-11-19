#include "compile/exact_eval.h"
#include "compile/pcfb_writer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <string>
#include <vector>

namespace {

std::string aralia(const char *name) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / name).string();
}

double rel_err(double got, double ref) {
  return std::abs(got - ref) / std::abs(ref);
}

fuzzybits::compile::Compiled must_compile(const std::string &xml) {
  auto compiled = fuzzybits::compile::compile_files({xml}, {});
  REQUIRE(compiled);
  REQUIRE_FALSE(compiled->bytes.empty());
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  const auto diags = pcfb::validate(*circuit);
  INFO("validate diagnostics: " << diags.size());
  for (const auto &d : diags) {
    UNSCOPED_INFO(d.message);
  }
  REQUIRE(diags.empty());
  return std::move(*compiled);
}

}  // namespace

TEST_CASE("fzb compile chinese.xml validates and matches the Aralia BDD reference", "[compile][chinese]") {
  // Issue #43 / spec §10 published digit string: 1.1705818e-3 (8 sig figs).
  // CSV stores 0.00117058. Bit-sliced exhaustive eval over 25 events yields
  // 1.1705818107586689e-03 (rel. err vs the published rounding ≈ 9.2e-9,
  // which is the last published digit). Require < 1e-8.
  constexpr double kChineseRef = 1.1705818e-3;
  auto compiled = must_compile(aralia("chinese.xml"));
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(circuit->schema_version() == pcfb::kSchemaVersionV11);

  const auto n_rand = fuzzybits::compile::random_sample_count(*circuit);
  REQUIRE(n_rand <= fuzzybits::compile::kMaxExactEvents);
  const double p = fuzzybits::compile::exact_probability(*circuit);
  INFO("exact_probability " << std::scientific << std::setprecision(16) << p);
  REQUIRE(rel_err(p, kChineseRef) < 1e-8);
}

TEST_CASE("fzb compile baobab2.xml validates", "[compile][baobab2]") {
  // BDD reference (benchmarks/exact_results_prime_implicants_bdd.csv): 0.000713018.
  // 32 defined basic events — exact eval is attempted only if the lowered
  // circuit has ≤ 30 random SAMPLE ops.
  constexpr double kBaobab2Ref = 0.000713018;
  auto compiled = must_compile(aralia("baobab2.xml"));
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  const auto n_rand = fuzzybits::compile::random_sample_count(*circuit);
  if (n_rand <= fuzzybits::compile::kMaxExactEvents) {
    const double p = fuzzybits::compile::exact_probability(*circuit);
    // CSV is printed to 6 significant figures; allow 5e-6 relative.
    REQUIRE(rel_err(p, kBaobab2Ref) < 5e-6);
  }
}

TEST_CASE("fzb compile das9201.xml validates", "[compile][das9201]") {
  // BDD reference: 0.0134237. Too many events for exhaustive eval.
  auto compiled = must_compile(aralia("das9201.xml"));
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(circuit->subcircuits()->Get(0)->queries()->size() >= 1);
}

TEST_CASE("compile.options records effective -c/--no-kn", "[compile][metadata]") {
  fuzzybits::compile::Options opt;
  opt.compilation_level = 1;
  opt.expand_atleast_gates = true;
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  bool found = false;
  const auto *meta = circuit->get()->metadata();
  REQUIRE(meta);
  for (flatbuffers::uoffset_t i = 0; i < meta->size(); ++i) {
    const auto *kv = meta->Get(i);
    if (!kv || !kv->key() || kv->key()->str() != "compile.options") {
      continue;
    }
    REQUIRE(kv->value_as_StringValue());
    const auto *s = kv->value_as_StringValue()->value();
    REQUIRE(s);
    REQUIRE(std::string(s->c_str()).find("-c 1") != std::string::npos);
    REQUIRE(std::string(s->c_str()).find("--no-kn") != std::string::npos);
    found = true;
  }
  REQUIRE(found);
}

TEST_CASE("compile writes fuzzybits origins and compile.level metadata", "[compile][metadata]") {
  fuzzybits::compile::Options opt;
  opt.compilation_level = 2;
  opt.expand_atleast_gates = true;
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto *meta = circuit->get()->metadata();
  REQUIRE(meta);
  bool saw_origins = false;
  bool saw_level = false;
  bool saw_expand = false;
  for (flatbuffers::uoffset_t i = 0; i < meta->size(); ++i) {
    const auto *kv = meta->Get(i);
    if (!kv || !kv->key()) {
      continue;
    }
    const auto key = kv->key()->str();
    if (key == "fuzzybits.origins") {
      REQUIRE(kv->value_as_StringValue());
      const auto *s = kv->value_as_StringValue()->value();
      REQUIRE(s);
      const std::string json = s->c_str();
      REQUIRE(json.find("\"id\":\"") != std::string::npos);
      saw_origins = true;
    }
    if (key == "compile.level") {
      REQUIRE(kv->value_as_IntValue());
      REQUIRE(kv->value_as_IntValue()->value() == 2);
      saw_level = true;
    }
    if (key == "compile.expand_atleast_gates") {
      REQUIRE(kv->value_as_IntValue());
      REQUIRE(kv->value_as_IntValue()->value() == 1);
      saw_expand = true;
    }
  }
  REQUIRE(saw_origins);
  REQUIRE(saw_level);
  REQUIRE(saw_expand);
}

TEST_CASE("compile watch emits an extra PROBABILITY query", "[compile][watch]") {
  fuzzybits::compile::Options opt;
  opt.compilation_level = 2;
  opt.expand_atleast_gates = true;
  opt.watch = {"g1"};
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  REQUIRE(circuit->subcircuits()->Get(0)->queries()->size() >= 2);
  bool saw_g1 = false;
  for (const auto *q : *circuit->subcircuits()->Get(0)->queries()) {
    if (q && q->name() && q->name()->str() == "g1") {
      saw_g1 = true;
    }
  }
  REQUIRE(saw_g1);
}
