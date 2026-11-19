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
  /// "cuda", "omp", or empty (AdaptiveCpp default). Applied via
  /// ACPP_VISIBILITY_MASK before the first sycl::queue is constructed.
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

/// Owns the reconstructed PDAG plus the MEF objects kernel_builder reads
/// through `Pdag::basic_events()`. Special members are out-of-line because
/// `Pdag` / `BasicEvent` / `ConstantExpression` are incomplete here.
struct ReconstructedPdag {
  std::unique_ptr<scram::core::Pdag> graph;
  std::vector<std::unique_ptr<scram::mef::BasicEvent>> events;
  std::vector<std::unique_ptr<scram::mef::ConstantExpression>> expressions;
  std::vector<ReconstructedQuery> queries;
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
  std::uint64_t pass_index = 0;  ///< 1-based
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
          std::size_t requested_trials_per_pass = 0);
  Session(Session &&) noexcept;
  Session &operator=(Session &&) noexcept;
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  static Session from_circuit(const pcfb::ir::Circuit &circuit, const std::string &backend = {},
                              std::size_t requested_trials_per_pass = 0);
  static Session from_bytes(std::span<const std::byte> bytes, const std::string &backend = {},
                            std::size_t requested_trials_per_pass = 0);
  static Session from_path(const std::filesystem::path &path, const std::string &backend = {},
                           std::size_t requested_trials_per_pass = 0);

  [[nodiscard]] scram::core::Pdag &pdag();
  [[nodiscard]] const scram::core::Pdag &pdag() const;
  [[nodiscard]] const std::vector<ReconstructedQuery> &queries() const;
  [[nodiscard]] SampleShape sample_shape() const;
  [[nodiscard]] bool closed() const;

  PassResult step();
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuzzybits::runtime
