/**
 * @file importance_analysis.cc
 * @brief Monte Carlo importance analysis implementation using SYCL-based parallel computation
 * @author Arjun Earthperson
 * @date 2025
 *
 * @copyright Copyright (C) 2025 Arjun Earthperson
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **/

#include "importance_analysis.h"
#include "mc/direct_eval.h"

/// Specialization of importance analyzer with computed tallies
template <>
class scram::core::ImportanceAnalyzer<scram::mc::DirectEval> : public scram::core::ImportanceAnalyzerBase {
public:
    /// Constructs importance analyzer from probability analyzer.
    /// Probability analyzer facilities are used
    /// to calculate the total and conditional probabilities for factors.
    ///
    /// @param[in] prob_analyzer  Instantiated probability analyzer.
    explicit ImportanceAnalyzer(ProbabilityAnalyzer<Bdd>* prob_analyzer)
        : ImportanceAnalyzerBase(prob_analyzer),
          bdd_graph_(prob_analyzer->bdd_graph()) {}

private:
    double CalculateMif(int index) noexcept override;

    /// Calculates Marginal Importance Factor of a variable.
    ///
    /// @param[in] vertex  The root vertex of a function graph.
    /// @param[in] order  The identifying order of the variable.
    /// @param[in] mark  A flag to mark traversed vertices.
    ///
    /// @returns Importance factor value.
    ///
    /// @note Probability factor fields are used to save results.
    /// @note The graph needs cleaning its marks after this function
    ///       because the graph gets continuously-but-partially marked.
    double CalculateMif(const Bdd::VertexPtr& vertex, int order,
                        bool mark) noexcept;

    /// Retrieves memorized probability values for BDD function graphs.
    ///
    /// @param[in] vertex  Vertex with calculated probabilities.
    ///
    /// @returns Saved probability value of the vertex.
    double RetrieveProbability(const Bdd::VertexPtr& vertex) noexcept;

    Bdd* bdd_graph_;  ///< Binary decision diagram for the analyzer.
};