#include "runtime/runtime.h"

#include "mc/queue/layer_manager.h"
#include "mc/scheduler/convergence_controller.h"
#include "mc/stats/tally_node_map.h"
#include "pdag.h"
#include "settings.h"

#include <cstdlib>
#include <exception>
#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuzzybits::runtime {
namespace {

using bitpack_t = std::uint64_t;
using prob_t = std::double_t;
using size_t_ = std::size_t;

void collect_nodes(const std::shared_ptr<scram::core::Gate> &gate,
                   std::unordered_map<int, std::shared_ptr<scram::core::Node>> &nodes) {
  if (!gate || !nodes.emplace(gate->index(), gate).second) {
    return;
  }
  for (const auto &arg : gate->args<scram::core::Gate>()) {
    collect_nodes(arg.second, nodes);
  }
  for (const auto &arg : gate->args<scram::core::Variable>()) {
    nodes.emplace(arg.second->index(), arg.second);
  }
}

}  // namespace

std::vector<QueryResult> run_graph(scram::core::Pdag &pdag,
                                   const std::vector<ReconstructedQuery> &queries,
                                   const RunOptions &options) {
  if (options.trials == 0) {
    throw RuntimeError("run: --trials must be positive");
  }
  if (queries.empty()) {
    throw RuntimeError("run: no queries");
  }
  if (!pdag.root_ptr()) {
    throw RuntimeError("run: PDAG has no root");
  }

  apply_backend_mask(options.backend);

  scram::core::Settings settings;
  settings.num_trials(static_cast<std::double_t>(options.trials));
  settings.early_stop(false);
  settings.seed(options.seed);
  settings.ci_policy(scram::core::CIPolicy::kWald);
  settings.watch_mode(false);

  scram::mc::stats::TallyNodeMap monitored;
  for (const auto &q : queries) {
    if (!q.node) {
      throw RuntimeError("run: query '" + q.name + "' has no node");
    }
    monitored.insert_or_assign(q.node->index(), scram::mc::stats::TallyNode{
                                                    .status = scram::mc::stats::not_tracked,
                                                    .tally_stats = {},
                                                    .node = q.node,
                                                });
  }

  try {
    scram::mc::queue::layer_manager<bitpack_t, prob_t, size_t_> manager(
        &pdag, static_cast<size_t_>(options.trials), monitored, /*requested_trials_per_pass=*/0);

    const auto total_passes = manager.shaper().TOTAL_ITERATIONS;
    if (total_passes == 0) {
      throw RuntimeError("run: sample shaper produced zero iterations");
    }
    manager.pass_wait_collect(monitored, total_passes, /*passes_between_waits=*/1);

    std::vector<QueryResult> results;
    results.reserve(queries.size());
    for (const auto &q : queries) {
      const auto &t = monitored.at(q.node->index()).tally_stats;
      QueryResult r;
      r.name = q.name;
      r.estimate = t.mean;
      r.ci_low = t.ci[0];
      r.ci_high = t.ci[1];
      r.n = static_cast<std::uint64_t>(t.total_bits);
      r.hits = static_cast<std::uint64_t>(t.num_one_bits);
      results.push_back(std::move(r));
    }
    return results;
  } catch (const RuntimeError &) {
    throw;
  } catch (const sycl::exception &e) {
    throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
  } catch (const std::exception &e) {
    throw RuntimeError(std::string("Monte Carlo run failed: ") + e.what());
  }

}

std::vector<QueryResult> run_pcfb(const pcfb::ir::Circuit &circuit, const RunOptions &options) {
  apply_backend_mask(options.backend);
  auto reconstructed = pdag_from_circuit(circuit);
  return run_graph(*reconstructed.graph, reconstructed.queries, options);
}

namespace {

pcfb::ir::Circuit require_valid_circuit(pcfb::Result<pcfb::ir::Circuit> circuit, const char *op) {
  if (!circuit) {
    throw RuntimeError(std::string(op) + ": " + circuit.error().message);
  }
  const auto diags = pcfb::validate(*circuit);
  if (!diags.empty()) {
    throw InvalidPcfbError(std::string(op) + ": pcfb validation failed (" + std::to_string(diags.size()) +
                               " diagnostics)",
                           diags);
  }
  return std::move(*circuit);
}

}  // namespace

pcfb::ir::Circuit load_pcfb(const std::filesystem::path &path) {
  return require_valid_circuit(pcfb::io::read(path), "load_pcfb");
}

pcfb::ir::Circuit load_pcfb(std::span<const std::byte> bytes) {
  return require_valid_circuit(pcfb::io::from_bytes(bytes), "load_pcfb");
}

std::vector<QueryResult> run_pcfb(const std::filesystem::path &path, const RunOptions &options) {
  apply_backend_mask(options.backend);
  return run_pcfb(load_pcfb(path), options);
}

std::vector<QueryResult> run_pcfb(std::span<const std::byte> bytes, const RunOptions &options) {
  apply_backend_mask(options.backend);
  return run_pcfb(load_pcfb(bytes), options);
}

struct Session::Impl {
  using bitpack_t = std::uint64_t;
  using prob_t = std::double_t;
  using size_t_ = std::size_t;
  using Manager = scram::mc::queue::layer_manager<bitpack_t, prob_t, size_t_>;

  ReconstructedPdag reconstructed;
  std::unique_ptr<Manager> manager;
  std::uint64_t pass_index = 0;
  bool closed = false;
};

Session::Session(ReconstructedPdag reconstructed, const std::string &backend,
                 std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                 std::size_t num_trials) {
  apply_backend_mask(backend);
  auto impl = std::make_unique<Impl>();
  impl->reconstructed = std::move(reconstructed);
  if (!impl->reconstructed.graph || !impl->reconstructed.graph->root_ptr()) {
    throw RuntimeError("session: reconstructed PDAG has no root");
  }
  if (impl->reconstructed.queries.empty()) {
    throw RuntimeError("session: no queries");
  }

  scram::mc::stats::TallyNodeMap tallies;
  for (const auto &q : impl->reconstructed.queries) {
    if (!q.node) {
      throw RuntimeError("session: query '" + q.name + "' has no node");
    }
    tallies.insert_or_assign(q.node->index(), scram::mc::stats::TallyNode{
                                                  .status = scram::mc::stats::not_tracked,
                                                  .tally_stats = {},
                                                  .node = q.node,
                                              });
  }
  if (!watch.empty()) {
    std::unordered_map<int, std::shared_ptr<scram::core::Node>> nodes;
    collect_nodes(impl->reconstructed.graph->root_ptr(), nodes);
    for (int index : resolve_watch(impl->reconstructed, watch)) {
      auto it = nodes.find(index);
      if (it == nodes.end()) {
        throw RuntimeError("session: watched index " + std::to_string(index) + " is not in the PDAG");
      }
      tallies.insert_or_assign(index, scram::mc::stats::TallyNode{
                                          .status = scram::mc::stats::not_tracked,
                                          .tally_stats = {},
                                          .node = it->second,
                                      });
    }
  }
  try {
    impl->manager = std::make_unique<Impl::Manager>(impl->reconstructed.graph.get(), num_trials, tallies,
                                                    requested_trials_per_pass);
  } catch (const RuntimeError &) {
    throw;
  } catch (const sycl::exception &e) {
    throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
  } catch (const std::exception &e) {
    throw RuntimeError(std::string("Monte Carlo session failed: ") + e.what());
  }
  impl_ = std::move(impl);
}

Session::Session(Session &&) noexcept = default;
Session &Session::operator=(Session &&) noexcept = default;
Session::~Session() = default;

Session Session::from_circuit(const pcfb::ir::Circuit &circuit, const std::string &backend,
                              std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                              std::size_t num_trials) {
  const auto diags = pcfb::validate(circuit);
  if (!diags.empty()) {
    throw InvalidPcfbError(
        "session: pcfb validation failed (" + std::to_string(diags.size()) + " diagnostics)", diags);
  }
  return Session(pdag_from_circuit(circuit), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

Session Session::from_bytes(std::span<const std::byte> bytes, const std::string &backend,
                            std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                            std::size_t num_trials) {
  return from_circuit(load_pcfb(bytes), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

Session Session::from_path(const std::filesystem::path &path, const std::string &backend,
                           std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                           std::size_t num_trials) {
  return from_circuit(load_pcfb(path), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

scram::core::Pdag &Session::pdag() {
  if (!impl_ || impl_->closed) {
    throw RuntimeError("session: closed");
  }
  return *impl_->reconstructed.graph;
}

const scram::core::Pdag &Session::pdag() const {
  if (!impl_ || impl_->closed) {
    throw RuntimeError("session: closed");
  }
  return *impl_->reconstructed.graph;
}

const std::vector<ReconstructedQuery> &Session::queries() const {
  if (!impl_ || impl_->closed) {
    throw RuntimeError("session: closed");
  }
  return impl_->reconstructed.queries;
}

const GraphMeta &Session::meta() const {
  if (!impl_ || impl_->closed) {
    throw RuntimeError("session: closed");
  }
  return impl_->reconstructed.meta;
}

SampleShape Session::sample_shape() const {
  if (!impl_ || impl_->closed || !impl_->manager) {
    throw RuntimeError("session: closed");
  }
  const auto &shape = impl_->manager->shaper().SAMPLE_SHAPE;
  SampleShape out;
  out.batch_size = shape.batch_size;
  out.bitpacks_per_batch = shape.bitpacks_per_batch;
  out.bits_per_bitpack = sizeof(Impl::bitpack_t) * 8;
  out.trials_per_pass =
      static_cast<std::uint64_t>(shape.batch_size) * shape.bitpacks_per_batch * out.bits_per_bitpack;
  out.total_iterations = impl_->manager->shaper().TOTAL_ITERATIONS;
  return out;
}

bool Session::closed() const {
  return !impl_ || impl_->closed;
}

PassResult Session::step() {
  if (!impl_ || impl_->closed || !impl_->manager) {
    throw RuntimeError("session: closed");
  }
  scram::mc::stats::TallyNodeMap tallies;
  try {
    impl_->manager->pass_wait_collect(tallies, 1, 0);
  } catch (const RuntimeError &) {
    throw;
  } catch (const sycl::exception &e) {
    throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
  } catch (const std::exception &e) {
    throw RuntimeError(std::string("Monte Carlo session failed: ") + e.what());
  }
  ++impl_->pass_index;

  PassResult out;
  out.pass_index = impl_->pass_index;
  out.trials_per_pass = sample_shape().trials_per_pass;
  out.target_index = impl_->reconstructed.queries.front().node_index;
  out.nodes.reserve(tallies.size());
  for (const auto &[index, node_tally] : tallies) {
    out.nodes.push_back(PassTally{
        .index = index,
        .num_one_bits = static_cast<std::uint64_t>(node_tally.tally_stats.num_one_bits),
        .total_bits = static_cast<std::uint64_t>(node_tally.tally_stats.total_bits),
    });
  }
  return out;
}

void Session::close() {
  if (!impl_) {
    return;
  }
  impl_->manager.reset();
  impl_->closed = true;
}

}  // namespace fuzzybits::runtime
