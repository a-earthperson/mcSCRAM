/// Host-only stubs for ProbabilityAnalyzer<DirectEval>.
///
/// On Linux, libscram.so references these symbols (they live in SYCL libmc).
/// `fzb` does not link libmc; this TU satisfies the dynamic loader.

#include "probability_analysis.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace scram::core {

ProbabilityAnalyzer<mc::DirectEval>::ProbabilityAnalyzer(FaultTreeAnalyzer<mc::DirectEval> *fta,
                                                         mef::MissionTime *mission_time)
    : ProbabilityAnalyzerBase(fta, mission_time) {}

ProbabilityAnalyzer<mc::DirectEval>::~ProbabilityAnalyzer() = default;

double ProbabilityAnalyzer<mc::DirectEval>::CalculateTotalProbability(const Pdag::IndexMap<double> &) {
  throw std::runtime_error("host-only fzb: ProbabilityAnalyzer<DirectEval> requires libmc");
}

std::vector<std::pair<double, double>> ProbabilityAnalyzer<mc::DirectEval>::CalculateProbabilityOverTime() {
  return {};
}

void ProbabilityAnalyzer<mc::DirectEval>::ComputeTallies(bool) {
  throw std::runtime_error("host-only fzb: ProbabilityAnalyzer<DirectEval> requires libmc");
}

std::unordered_set<int> ProbabilityAnalyzer<mc::DirectEval>::ObserveNodes(Pdag *, mc::stats::TallyNodeMap &,
                                                                         const std::unordered_set<index_t> &, bool,
                                                                         bool) {
  return {};
}

auto ProbabilityAnalyzer<mc::DirectEval>::GatherGates(Pdag *, std::unordered_set<index_t> *) {}

std::unordered_set<int> ProbabilityAnalyzer<mc::DirectEval>::GatherGates(Pdag *) {
  return {};
}

void ProbabilityAnalyzer<mc::DirectEval>::SanitizeWatchState() {}

}  // namespace scram::core
