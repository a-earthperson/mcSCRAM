#include "compile/exact_eval.h"
#include "compile/pcfb_writer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace {

std::string aralia(const char *name) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / name).string();
}

std::string repo_input(const char *relative) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / relative).string();
}

double exp_unreliability(double lambda, double mission_hours = 8760.0) {
  return 1.0 - std::exp(-lambda * mission_hours);
}

std::map<std::string, double> sample_probabilities(const pcfb::ir::Circuit &circuit) {
  std::map<std::string, double> out;
  REQUIRE(circuit.subcircuits());
  REQUIRE(circuit.subcircuits()->size() > 0);
  const auto *sc = circuit.subcircuits()->Get(0);
  REQUIRE(sc);
  REQUIRE(sc->operators());
  for (const auto *op : *sc->operators()) {
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    const auto *sa = op->args_as_SampleArgs();
    REQUIRE(sa);
    out[sa->name() ? sa->name()->str() : std::string()] = sa->probability();
  }
  return out;
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

TEST_CASE("SmallTree SAMPLE probabilities are exponential point values", "[compile][smalltree]") {
  // Default mission time is 8760 h. lambda1=lambda3=2e-5, lambda2=lambda4=1e-5
  // (lognormal-deviate *mean*; uncertainty sampling is out of scope).
  auto compiled = must_compile(repo_input("input/SmallTree/SmallTree.xml"));
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto samples = sample_probabilities(*circuit);
  REQUIRE(samples.size() == 4);
  const double p12 = exp_unreliability(2.0e-5);
  const double p_e2 = exp_unreliability(1.0e-5);
  REQUIRE(std::abs(samples.at("e1") - p12) < 1e-12);
  REQUIRE(std::abs(samples.at("e2") - p_e2) < 1e-12);
  REQUIRE(std::abs(samples.at("e3") - p12) < 1e-12);
  REQUIRE(std::abs(samples.at("e4") - p_e2) < 1e-12);
}

TEST_CASE("BSCU expression probabilities lower to SAMPLE in [0, 1]", "[compile][bscu]") {
  auto compiled = must_compile(repo_input("input/BSCU/BSCU.xml"));
  auto circuit = pcfb::io::from_bytes(compiled.bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto samples = sample_probabilities(*circuit);
  REQUIRE(samples.size() >= 1);
  for (const auto &[name, p] : samples) {
    INFO("SAMPLE " << name << " p=" << std::scientific << std::setprecision(16) << p);
    REQUIRE(std::isfinite(p));
    REQUIRE(p >= 0.0);
    REQUIRE(p <= 1.0);
    REQUIRE(p > 0.0);  // every BSCU event has a positive mean rate
  }
}

TEST_CASE("compile rejects a basic event with no probability expression", "[compile][error]") {
  auto compiled =
      fuzzybits::compile::compile_files({repo_input("tests/input/fta/missing_expression.xml")}, {});
  REQUIRE_FALSE(compiled);
  REQUIRE(compiled.error().message.find("no probability expression") != std::string::npos);
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
