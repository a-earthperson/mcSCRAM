#include "runtime/runtime.h"

#include "mc/queue/layer_manager.h"
#include "mc/scheduler/convergence_controller.h"
#include "mc/stats/tally_node_map.h"
#include "pdag.h"
#include "settings.h"

#include <cstdlib>
#include <utility>

namespace fuzzybits::runtime {
namespace {

using bitpack_t = std::uint64_t;
using prob_t = std::double_t;
using size_t_ = std::size_t;

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
}

std::vector<QueryResult> run_pcfb(const pcfb::ir::Circuit &circuit, const RunOptions &options) {
  apply_backend_mask(options.backend);
  auto reconstructed = pdag_from_circuit(circuit);
  return run_graph(*reconstructed.graph, reconstructed.queries, options);
}

std::vector<QueryResult> run_pcfb(const std::filesystem::path &path, const RunOptions &options) {
  apply_backend_mask(options.backend);
  auto circuit = pcfb::io::read(path);
  if (!circuit) {
    throw RuntimeError("run: " + circuit.error().message);
  }
  const auto diags = pcfb::validate(*circuit);
  if (!diags.empty()) {
    throw InvalidPcfbError("run: pcfb validation failed (" + std::to_string(diags.size()) + " diagnostics)",
                           diags);
  }
  return run_pcfb(*circuit, options);
}

}  // namespace fuzzybits::runtime
