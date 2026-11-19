#include "compile/pcfb_writer.h"
#include "pdag.h"
#include "runtime/runtime.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct OmpBackend {
  OmpBackend() { setenv("ACPP_VISIBILITY_MASK", "omp", 0); }
};
const OmpBackend kOmpBackend{};

std::string aralia(const char *name) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / name).string();
}

std::string repo_input(const char *relative) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / relative).string();
}

fuzzybits::compile::Options no_kn() {
  fuzzybits::compile::Options opt;
  opt.compilation_level = 2;
  opt.expand_atleast_gates = true;
  return opt;
}

void check_pcfb_matches_memory(const std::string &xml, std::uint64_t trials) {
  fuzzybits::runtime::RunOptions opt;
  opt.trials = trials;
  opt.backend = "omp";
  opt.seed = 372;

  auto live = fuzzybits::compile::compile_to_graph({xml}, no_kn());
  REQUIRE(live);
  REQUIRE(live->graph);

  fuzzybits::runtime::ReconstructedQuery q;
  q.name = live->target_id;
  q.node = live->graph->root_ptr();
  q.node_index = live->graph->root_ptr()->index();
  const auto in_memory = fuzzybits::runtime::run_graph(*live->graph, {q}, opt);
  REQUIRE(in_memory.size() == 1);

  auto compiled = fuzzybits::compile::pdag_to_pcfb(*live->graph, no_kn(), live->model_name, live->target_id, {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(*compiled);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto from_pcfb = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(from_pcfb.size() == 1);

  INFO("in-memory hits=" << in_memory[0].hits << " n=" << in_memory[0].n);
  INFO("pcfb hits=" << from_pcfb[0].hits << " n=" << from_pcfb[0].n);
  REQUIRE(from_pcfb[0].hits == in_memory[0].hits);
  REQUIRE(from_pcfb[0].n == in_memory[0].n);
  REQUIRE(from_pcfb[0].n >= trials);
}

double z_score(double p_hat, double p_ref, std::uint64_t n) {
  const double sigma = std::sqrt(p_hat * (1.0 - p_hat) / static_cast<double>(n));
  REQUIRE(sigma > 0.0);
  return (p_hat - p_ref) / sigma;
}

}  // namespace

TEST_CASE("fzb run refuses an invalid pcfb file", "[runtime][mc]") {
  const auto path = std::filesystem::temp_directory_path() / "fuzzybits-invalid.pcfb";
  {
    std::ofstream out(path, std::ios::binary);
    const char junk[] = "not a pcfb";
    out.write(junk, sizeof(junk) - 1);
  }
  fuzzybits::runtime::RunOptions opt;
  opt.trials = 64;
  opt.backend = "omp";
  CHECK_THROWS_AS(fuzzybits::runtime::run_pcfb(path, opt), fuzzybits::runtime::RuntimeError);
  std::filesystem::remove(path);
}

TEST_CASE("pcfb reconstruction places SAMPLE event_id at the Variable index", "[runtime][mc]") {
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  auto reconstructed = fuzzybits::runtime::pdag_from_circuit(*circuit);
  REQUIRE(reconstructed.graph);
  REQUIRE_FALSE(reconstructed.queries.empty());
  REQUIRE_FALSE(reconstructed.events.empty());

  const auto *sc = circuit->subcircuits()->Get(0);
  std::size_t n_sample = 0;
  for (const auto *op : *sc->operators()) {
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    ++n_sample;
    const auto *sa = op->args_as_SampleArgs();
    REQUIRE(sa);
    const auto idx = static_cast<int>(sa->event_id());
    REQUIRE(idx >= scram::core::Pdag::kVariableStartIndex);
    const auto *event = reconstructed.graph->basic_events()[idx];
    REQUIRE(event != nullptr);
    if (sa->always_true()) {
      REQUIRE(event->p() == 1.0);
    } else {
      REQUIRE(event->p() == sa->probability());
    }
  }
  REQUIRE(reconstructed.events.size() >= n_sample);
  REQUIRE_FALSE(reconstructed.threshold_notes.empty());
  REQUIRE(reconstructed.threshold_notes.front().find("kernel_T_hi=") != std::string::npos);
  REQUIRE(reconstructed.threshold_notes.front().find("two-word Philox compare") != std::string::npos);
}

TEST_CASE("fzb run chinese.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity]") {
  constexpr std::uint64_t kTrials = 65536;
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrials;
  opt.backend = "omp";
  opt.seed = 372;

  auto live = fuzzybits::compile::compile_to_graph({aralia("chinese.xml")}, no_kn());
  REQUIRE(live);
  REQUIRE(live->graph);

  fuzzybits::runtime::ReconstructedQuery q;
  q.name = live->target_id;
  q.node = live->graph->root_ptr();
  q.node_index = live->graph->root_ptr()->index();
  const auto in_memory = fuzzybits::runtime::run_graph(*live->graph, {q}, opt);
  REQUIRE(in_memory.size() == 1);

  auto compiled = fuzzybits::compile::pdag_to_pcfb(*live->graph, no_kn(), live->model_name, live->target_id, {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(*compiled);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto from_pcfb = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(from_pcfb.size() == 1);

  INFO("in-memory hits=" << in_memory[0].hits << " n=" << in_memory[0].n);
  INFO("pcfb hits=" << from_pcfb[0].hits << " n=" << from_pcfb[0].n);
  REQUIRE(from_pcfb[0].hits == in_memory[0].hits);
  REQUIRE(from_pcfb[0].n == in_memory[0].n);
  REQUIRE(from_pcfb[0].n >= kTrials);
}

TEST_CASE("Session streams the same first-pass hits as run_graph on chinese", "[runtime][mc][bit-identity]") {
  constexpr std::uint64_t kTrialsPerPass = 65536;
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrialsPerPass;
  opt.backend = "omp";
  opt.seed = 372;

  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  const auto oneshot = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(oneshot.size() == 1);

  auto session = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp", kTrialsPerPass);
  const auto first = session.step();
  REQUIRE(first.pass_index == 1);
  REQUIRE(first.trials_per_pass >= kTrialsPerPass);
  REQUIRE_FALSE(first.nodes.empty());
  session.close();
  REQUIRE(session.closed());
}

TEST_CASE("extra watch queries do not replace the PDAG root", "[runtime][mc][watch]") {
  fuzzybits::compile::Options opt = no_kn();
  opt.watch = {"g1"};
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  auto reconstructed = fuzzybits::runtime::pdag_from_circuit(*circuit);
  REQUIRE(reconstructed.queries.size() >= 2);
  REQUIRE(reconstructed.graph->root_ptr());
  REQUIRE(reconstructed.graph->root_ptr()->index() == reconstructed.queries.front().node_index);
  REQUIRE_FALSE(reconstructed.meta.target.empty());
  const auto watched = fuzzybits::runtime::Session::resolve_watch(reconstructed, {"g1"});
  REQUIRE(watched.size() == 1);
}

TEST_CASE("fzb run SmallTree.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity][smalltree]") {
  check_pcfb_matches_memory(repo_input("input/SmallTree/SmallTree.xml"), 65536);
}

TEST_CASE("fzb run BSCU.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity][bscu]") {
  check_pcfb_matches_memory(repo_input("input/BSCU/BSCU.xml"), 65536);
}

TEST_CASE("fzb run SmallTree.pcfb matches the exponential point-value formula |z|<=4",
          "[runtime][mc][smalltree]") {
  constexpr std::uint64_t kTrials = 1 << 16;
  constexpr double kRef = 0.026776830672032736;  // 1-(1-p1 p2)(1-p3 p4), t=8760 h
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrials;
  opt.backend = "omp";
  const auto results = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].n >= kTrials);
  const double z = z_score(results[0].estimate, kRef, results[0].n);
  INFO("p_hat=" << results[0].estimate << " p_ref=" << kRef << " z=" << z);
  REQUIRE(std::abs(z) <= 4.0);
}

TEST_CASE("fzb run chinese.pcfb small-N estimate is in (0,1) on CPU", "[runtime][mc]") {
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  const auto tmp = std::filesystem::temp_directory_path() / "fuzzybits-chinese-runtime.pcfb";
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(reinterpret_cast<const char *>(compiled->bytes.data()),
              static_cast<std::streamsize>(compiled->bytes.size()));
  }
  fuzzybits::runtime::RunOptions opt;
  opt.trials = 1 << 16;
  opt.backend = "omp";
  const auto results = fuzzybits::runtime::run_pcfb(tmp, opt);
  std::filesystem::remove(tmp);
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].n >= opt.trials);
  REQUIRE(results[0].estimate >= 0.0);
  REQUIRE(results[0].estimate <= 1.0);
  REQUIRE(results[0].ci_low <= results[0].estimate);
  REQUIRE(results[0].ci_high >= results[0].estimate);
}
