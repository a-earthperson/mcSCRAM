#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pcfb.h"

namespace scram::core {
class Pdag;
class Node;
}  // namespace scram::core

namespace scram::mef {
class BasicEvent;
class ConstantExpression;
class Gate;
}  // namespace scram::mef

namespace fuzzybits::runtime {

class RuntimeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class InvalidPcfbError : public RuntimeError {
 public:
  explicit InvalidPcfbError(std::string message, std::vector<pcfb::Diagnostic> diags = {})
      : RuntimeError(std::move(message)), diagnostics(std::move(diags)) {}

  std::vector<pcfb::Diagnostic> diagnostics;
};

struct RunOptions {
  std::uint64_t trials = 0;
  int seed = 372;
  /// "cuda", "metal", "omp", or empty (AdaptiveCpp default). Applied via
  /// ACPP_VISIBILITY_MASK, which make_throwing_sycl_queue() reads at queue
  /// construction.
  std::string backend;
};

struct QueryResult {
  std::string name;
  double estimate = 0;
  double ci_low = 0;
  double ci_high = 0;
  std::uint64_t n = 0;     ///< total Bernoulli trials (tally.total_bits)
  std::uint64_t hits = 0;  ///< raw one-bits (tally.num_one_bits)
};

struct ReconstructedQuery {
  std::string name;
  int node_index = 0;
  std::shared_ptr<scram::core::Node> node;
};

/// Fields `Pdag.to_graph()` / the server DAG views need, restored from
/// `compile.*` / `fuzzybits.*` metadata written at lowering.
struct GraphMeta {
  std::string model_name;
  std::string source;
  std::vector<std::string> input_files;
  std::string target;
  std::string target_name;
  int compilation_level = 2;
  bool ccf = false;
  bool expand_atleast_gates = false;
  bool expand_xor_gates = false;
  bool recognize_atleast_groups = true;
  bool recognize_xor_groups = true;
  bool keep_null_gates = false;
  bool has_compilation = false;
};

struct OriginRecord {
  int pdag_index = 0;
  std::string kind;
  std::string id;
  std::string name;
  std::string label;
  std::string full_path;
};

/// Owns the reconstructed PDAG plus the MEF objects kernel_builder reads
/// through `Pdag::basic_events()`. Special members are out-of-line because
/// `Pdag` / `BasicEvent` / `ConstantExpression` are incomplete here.
struct ReconstructedPdag {
  std::unique_ptr<scram::core::Pdag> graph;
  std::vector<std::unique_ptr<scram::mef::BasicEvent>> events;
  std::vector<std::unique_ptr<scram::mef::ConstantExpression>> expressions;
  std::vector<std::unique_ptr<scram::mef::Gate>> origin_gates;
  std::vector<ReconstructedQuery> queries;
  GraphMeta meta;
  std::vector<OriginRecord> origins;
  /// Notes about 32-bit Bernoulli truncation (#17 is Phase 2).
  std::vector<std::string> threshold_notes;

  ReconstructedPdag();
  ReconstructedPdag(ReconstructedPdag &&) noexcept;
  ReconstructedPdag &operator=(ReconstructedPdag &&) noexcept;
  ~ReconstructedPdag();
  ReconstructedPdag(const ReconstructedPdag &) = delete;
  ReconstructedPdag &operator=(const ReconstructedPdag &) = delete;
};

/// Apply `--backend` to the process environment. Must run before any
/// AdaptiveCpp / SYCL runtime initialisation (before the first sycl::queue).
void apply_backend_mask(const std::string &backend);

/// Reconstruct a preprocessed `core::Pdag` from a validated pcfb circuit.
/// Variables are created at SAMPLE `event_id` so Philox `pdag_idx` matches
/// the in-memory compile path.
[[nodiscard]] ReconstructedPdag pdag_from_circuit(const pcfb::ir::Circuit &circuit);

[[nodiscard]] std::vector<QueryResult> run_graph(scram::core::Pdag &pdag,
                                                 const std::vector<ReconstructedQuery> &queries,
                                                 const RunOptions &options);

[[nodiscard]] std::vector<QueryResult> run_pcfb(const pcfb::ir::Circuit &circuit,
                                                const RunOptions &options);

[[nodiscard]] std::vector<QueryResult> run_pcfb(const std::filesystem::path &path,
                                                const RunOptions &options);

[[nodiscard]] std::vector<QueryResult> run_pcfb(std::span<const std::byte> bytes,
                                                const RunOptions &options);

/// Read + `pcfb::validate`. Throws `InvalidPcfbError` on diagnostics, `RuntimeError`
/// on I/O / verification failure. `from_bytes` is a non-owning view: `bytes` must
/// outlive the returned circuit.
[[nodiscard]] pcfb::ir::Circuit load_pcfb(const std::filesystem::path &path);
[[nodiscard]] pcfb::ir::Circuit load_pcfb(std::span<const std::byte> bytes);

struct SampleShape {
  std::size_t batch_size = 0;
  std::size_t bitpacks_per_batch = 0;
  std::size_t bits_per_bitpack = 64;
  std::uint64_t trials_per_pass = 0;
  std::uint64_t total_iterations = 0;  ///< 0 when the shaper is unbounded
};

struct PassTally {
  int index = 0;
  std::uint64_t num_one_bits = 0;  ///< cumulative one-bits through this pass
  std::uint64_t total_bits = 0;    ///< cumulative Bernoulli trials through this pass
};

struct PassResult {
  std::uint64_t pass_index = 0;  ///< passes run so far, including this step's
  std::uint64_t trials_per_pass = 0;
  int target_index = 0;
  std::vector<PassTally> nodes;
};

/// Streaming Monte Carlo over a reconstructed pcfb PDAG. One `step()` is one
/// device pass (same kernel path as `fzb run` / `layer_manager`). Destroy or
/// `close()` to release USM. Not thread-safe.
class Session {
 public:
  Session(ReconstructedPdag reconstructed, const std::string &backend = {},
          std::size_t requested_trials_per_pass = 0, std::vector<std::string> watch = {},
          std::size_t num_trials = 0);
  Session(Session &&) noexcept;
  Session &operator=(Session &&) noexcept;
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  static Session from_circuit(const pcfb::ir::Circuit &circuit, const std::string &backend = {},
                              std::size_t requested_trials_per_pass = 0, std::vector<std::string> watch = {},
                              std::size_t num_trials = 0);
  static Session from_bytes(std::span<const std::byte> bytes, const std::string &backend = {},
                            std::size_t requested_trials_per_pass = 0, std::vector<std::string> watch = {},
                            std::size_t num_trials = 0);
  static Session from_path(const std::filesystem::path &path, const std::string &backend = {},
                           std::size_t requested_trials_per_pass = 0, std::vector<std::string> watch = {},
                           std::size_t num_trials = 0);

  [[nodiscard]] scram::core::Pdag &pdag();
  [[nodiscard]] const scram::core::Pdag &pdag() const;
  [[nodiscard]] const std::vector<ReconstructedQuery> &queries() const;
  [[nodiscard]] const GraphMeta &meta() const;
  [[nodiscard]] SampleShape sample_shape() const;
  [[nodiscard]] bool closed() const;

  /// Resolve extra watch names (id / name / full_path / decimal PDAG index)
  /// against the reconstructed graph. Unknown names throw `RuntimeError`.
  [[nodiscard]] static std::vector<int> resolve_watch(const ReconstructedPdag &reconstructed,
                                                      const std::vector<std::string> &names);

  /// Run `passes` device passes (at least 1), waiting once at the end, and
  /// return the cumulative tallies.
  PassResult step(std::size_t passes = 1);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuzzybits::runtime
