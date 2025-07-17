/**
 * @file analysis.cc
 * @brief Monte Carlo probability analysis implementation using SYCL-based parallel computation
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
 * @details This file implements the Monte Carlo-based probability analysis for SCRAM
 * using SYCL parallel computing. It provides the concrete implementation of the
 * ProbabilityAnalyzer<DirectEval> class specialization, which performs direct evaluation
 * of probabilistic directed acyclic graphs (PDAGs) through parallel sampling.
 * 
 * **Key Features:**
 * - High-performance Monte Carlo sampling using SYCL parallel execution
 * - Direct evaluation of PDAGs without intermediate product generation
 * - Configurable sampling parameters (trials, batch size, sample size)
 * - Statistical confidence interval computation
 * - Automatic device optimization for different hardware architectures
 * - Memory-efficient bit-packed sample representation
 * 
 * **Algorithm Overview:**
 * The implementation uses a layered approach where:
 * 1. **Graph Preparation**: The PDAG is topologically sorted into execution layers
 * 2. **Kernel Generation**: SYCL kernels are created for each layer's computations
 * 3. **Parallel Sampling**: Basic events are sampled using parallel PRNGs
 * 4. **Gate Evaluation**: Logical operations are computed in parallel
 * 5. **Result Aggregation**: Statistical measures are computed from sample tallies
 * 
 * **Performance Characteristics:**
 * - Scales linearly with number of sampling trials
 * - Parallel execution across all available compute units
 * - Memory bandwidth optimization through bit-packing
 * - Minimal synchronization overhead between layers
 * - Automatic work-group size optimization for target hardware
 * 
 * @note This implementation requires SYCL-compatible hardware (CPU/GPU)
 * @note Memory requirements scale with graph size and sample configuration
 * @note Computation precision depends on number of Monte Carlo trials
 * 
 * @see ProbabilityAnalyzer<DirectEval> for interface documentation
 * @see mc::queue::layer_manager for execution engine details
 * @see sample_shape for memory layout configuration
 * 
 * @example Basic usage:
 * @code
 * // Create analyzer from fault tree analysis
 * std::unique_ptr<ProbabilityAnalyzer<DirectEval>> analyzer = 
 *     std::make_unique<ProbabilityAnalyzer<DirectEval>>(fta, mission_time);
 * 
 * // Configure sampling parameters
 * analyzer->settings().num_trials(1000000);
 * analyzer->settings().batch_size(1024);
 * analyzer->settings().sample_size(16);
 * 
 * // Perform analysis
 * analyzer->Analyze();
 * 
 * // Get results
 * double probability = analyzer->p_total();
 * std::cout << "System failure probability: " << probability << std::endl;
 * @endcode
 * 
 * @example Advanced configuration:
 * @code
 * // Create analyzer with custom settings
 * ProbabilityAnalyzer<DirectEval> analyzer(fta, mission_time);
 * 
 * // Configure for high-precision analysis
 * analyzer.settings().num_trials(10000000);     // 10M trials for high precision
 * analyzer.settings().batch_size(2048);         // Larger batches for GPU efficiency
 * analyzer.settings().sample_size(32);          // More samples per batch
 * 
 * // Perform analysis with timing
 * auto start = std::chrono::high_resolution_clock::now();
 * analyzer.Analyze();
 * auto end = std::chrono::high_resolution_clock::now();
 * 
 * auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
 * std::cout << "Analysis completed in " << duration.count() << " ms" << std::endl;
 * std::cout << "Probability: " << analyzer.p_total() << std::endl;
 * @endcode
 *
 */

#include "direct_eval.h"
#include "logger.h"
#include "probability_analysis.h"
#include "queue/layer_manager.h"
#include "mc/stats/ci_utils.h"  // NEW: statistical helper utilities
#include <algorithm>

namespace scram::core {

    /**
     * @brief Constructs a Monte Carlo probability analyzer from a fault tree analyzer
     * 
     * @details Creates a probability analyzer that reuses the PDAG (Probabilistic Directed
     * Acyclic Graph) from the provided fault tree analyzer. This constructor is optimized
     * for performance by avoiding redundant graph construction and maintaining reference
     * consistency between the fault tree analysis and probability analysis phases.
     * 
     * The constructor establishes the connection between the fault tree analysis results
     * and the Monte Carlo sampling engine, ensuring that:
     * - The PDAG structure is preserved and accessible
     * - Basic event probabilities are correctly mapped
     * - Mission time expressions are properly configured
     * - Sampling parameters are initialized from analysis settings
     * 
     * **Key Initialization Steps:**
     * 1. **Base Class Initialization**: Calls ProbabilityAnalyzerBase constructor
     * 2. **Graph Reference Setup**: Establishes connection to existing PDAG
     * 3. **Probability Mapping**: Extracts variable probabilities from graph
     * 4. **Mission Time Configuration**: Associates time-dependent expressions
     * 5. **Logging Setup**: Configures debug output for analysis tracking
     * 
     * @param fta Pointer to fault tree analyzer containing the constructed PDAG
     * @param mission_time Pointer to mission time expression for time-dependent analysis
     * 
     * @pre fta must be a valid, completed fault tree analyzer
     * @pre fta->graph() must return a valid PDAG structure
     * @pre mission_time must be a valid mission time expression
     * @pre The underlying fault tree must not have changed since FTA completion
     * 
     * @post The analyzer is ready for Monte Carlo sampling
     * @post All basic event probabilities are extracted and mapped
     * @post The PDAG structure is accessible through graph()
     *
     * @note The analyzer maintains a reference to the original FTA's graph
     * @note Mission time expressions are not copied, only referenced
     * 
     * @example Basic construction:
     * @code
     * // After completing fault tree analysis
     * auto fta = std::make_unique<FaultTreeAnalyzer<DirectEval>>(fault_tree, settings);
     * fta->Analyze();
     * 
     * // Create probability analyzer reusing the PDAG
     * ProbabilityAnalyzer<DirectEval> prob_analyzer(fta.get(), mission_time);
     * 
     * // The analyzer is now ready for Monte Carlo sampling
     * prob_analyzer.Analyze();
     * @endcode
     * 
     * @example Advanced usage with validation:
     * @code
     * // Validate FTA before creating probability analyzer
     * if (!fta->graph() || !fta->graph()->root()) {
     *     throw std::runtime_error("Invalid fault tree analysis");
     * }
     * 
     * // Create analyzer with error handling
     * try {
     *     ProbabilityAnalyzer<DirectEval> analyzer(fta.get(), mission_time);
     *     LOG(INFO) << "Probability analyzer created successfully";
     *     LOG(INFO) << "PDAG contains " << analyzer.graph()->size() << " nodes";
     * } catch (const std::exception& e) {
     *     LOG(ERROR) << "Failed to create probability analyzer: " << e.what();
     *     throw;
     * }
     * @endcode
     * 
     * @see ProbabilityAnalyzerBase::ProbabilityAnalyzerBase for base class documentation
     * @see FaultTreeAnalyzer<DirectEval> for fault tree analysis details
     * @see core::Pdag for graph structure documentation
     */
    ProbabilityAnalyzer<DirectEval>::ProbabilityAnalyzer(FaultTreeAnalyzer<DirectEval> *fta, mef::MissionTime *mission_time)
        : ProbabilityAnalyzerBase(fta, mission_time) {
        LOG(DEBUG2) << "Re-using PDAG from FaultTreeAnalyzer for ProbabilityAnalyzer";
    }

    /**
     * @brief Destructor for Monte Carlo probability analyzer
     * 
     * @details Provides safe cleanup of the probability analyzer instance. The destructor
     * is marked as noexcept to ensure safe destruction even in exception scenarios.
     * 
     * **Cleanup Operations:**
     * - Automatic cleanup of base class resources
     * - No explicit resource management needed (RAII pattern)
     * - Safe destruction of member variables
     * 
     * @note The destructor is inline for performance optimization
     * @note All resource management is handled by RAII principles
     * @note The PDAG is not owned by this class, so no graph cleanup is performed
     * 
     * @example Safe destruction:
     * @code
     * {
     *     ProbabilityAnalyzer<DirectEval> analyzer(fta, mission_time);
     *     analyzer.Analyze();
     *     // Analyzer is automatically destroyed here
     * }
     * // All resources are properly cleaned up
     * @endcode
     */
    inline ProbabilityAnalyzer<DirectEval>::~ProbabilityAnalyzer() noexcept = default;

    /**
     * @brief Calculates total system failure probability using Monte Carlo sampling
     * 
     * @details Implements the core Monte Carlo probability calculation using SYCL-based
     * parallel computation. This method orchestrates the complete sampling process from
     * graph preparation through statistical result computation, providing high-performance
     * probabilistic analysis of complex pdags.
     * 
     * **Algorithm Implementation:**
     * 1. **Configuration Extraction**: Retrieves sampling parameters from analysis settings
     * 2. **Layer Manager Creation**: Instantiates SYCL-based computation engine
     * 3. **Parallel Sampling**: Executes Monte Carlo trials across all available cores
     * 4. **Statistical Computation**: Computes confidence intervals and error estimates
     * 5. **Result Aggregation**: Returns final probability estimate with timing information
     * 
     * **Sampling Parameters:**
     * - **num_trials**: Number of Monte Carlo iterations (affects precision)
     * - **batch_size**: Samples processed per kernel invocation (affects memory usage)
     * - **sample_size**: Bit-packs per batch (affects memory layout efficiency)
     * 
     * **Performance Characteristics:**
     * - **Time Complexity**: O(n × t) where n is graph size and t is trial count
     * - **Space Complexity**: O(n × b × s) where b is batch size and s is sample size
     * - **Parallel Efficiency**: Near-linear scaling with available compute units
     * - **Memory Bandwidth**: Optimized through bit-packed representations
     * 
     * **Statistical Properties:**
     * - **Convergence**: Follows Central Limit Theorem for large trial counts
     * - **Precision**: Standard error decreases as 1/√n for n trials
     * - **Confidence Intervals**: Computed using normal approximation
     * - **Bias**: Unbiased estimator for true probability
     * 
     * @param p_vars Index-mapped probability values for all basic events in the graph
     * 
     * @return Estimated total system failure probability (range: [0.0, 1.0])
     * 
     * @pre p_vars must contain probability values for all basic events in the PDAG
     * @pre All probability values must be in range [0.0, 1.0]
     * @pre The PDAG must be topologically sorted and acyclic
     * @pre SYCL device must be available and properly configured
     * 
     * @post The returned probability is a statistically valid estimate
     * @post Internal timing information is logged for performance analysis
     * @post The layer manager maintains computed statistics for inspection
     * 
     * @note The method is marked noexcept for performance optimization
     * @note Timing information is logged at DEBUG4 level
     * @note The p_vars parameter is currently unused but maintained for interface consistency
     * 
     * @throws std::runtime_error if SYCL device initialization fails
     * @throws std::bad_alloc if device memory allocation fails
     * @throws std::logic_error if graph structure is invalid
     * 
     * @example Basic probability calculation:
     * @code
     * // Configure sampling parameters
     * settings().num_trials(1000000);
     * settings().batch_size(1024);
     * settings().sample_size(16);
     * 
     * // Calculate probability
     * Pdag::IndexMap<double> probabilities;
     * double result = analyzer.CalculateTotalProbability(probabilities);
     * 
     * std::cout << "System failure probability: " << result << std::endl;
     * @endcode
     * 
     * @example High-precision analysis:
     * @code
     * // Configure for maximum precision
     * settings().num_trials(10000000);  // 10M trials
     * settings().batch_size(2048);      // Large batches for GPU efficiency
     * settings().sample_size(32);       // Optimal memory layout
     * 
     * // Perform calculation with timing
     * auto start = std::chrono::high_resolution_clock::now();
     * double probability = analyzer.CalculateTotalProbability(p_vars);
     * auto end = std::chrono::high_resolution_clock::now();
     * 
     * auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
     * std::cout << "Calculated probability " << probability 
     *           << " in " << duration.count() << " ms" << std::endl;
     * @endcode
     * 
     * @example Error handling:
     * @code
     * try {
     *     double result = analyzer.CalculateTotalProbability(p_vars);
     *     if (result < 0.0 || result > 1.0) {
     *         LOG(WARNING) << "Probability out of valid range: " << result;
     *     }
     * } catch (const std::exception& e) {
     *     LOG(ERROR) << "Monte Carlo calculation failed: " << e.what();
     *     throw;
     * }
     * @endcode
     * 
     * @see mc::queue::layer_manager for execution engine details
     * @see mc::queue::layer_manager::tally for statistical computation
     * @see sample_shape for memory layout configuration
     * @see Settings for parameter configuration options
     */
    double ProbabilityAnalyzer<DirectEval>::CalculateTotalProbability(const Pdag::IndexMap<double> &p_vars) noexcept {
        CLOCK(calc_time);
        LOG(WARNING) << "Calculating probability using monte carlo sampling...";

        using bitpack_t_ = std::uint64_t;

        // ---------------------------------------------------------------------
        // 0) Gather user-supplied sampling preferences.
        // ---------------------------------------------------------------------
        auto &st         = this->settings();
        const double eps = st.ci_margin_error();     // half-width ε
        const double conf= st.ci_confidence();       // confidence level (two-sided)
        const bool autotune = st.ci_autotune_trials() && eps > 0.0 && conf > 0.0 && conf < 1.0;

        // ---------------------------------------------------------------------
        // 1) Decide on the total number of Bernoulli trials.
        // ---------------------------------------------------------------------
        std::size_t trials = st.num_trials();
        auto *pdag = this->graph();

        if (autotune) {
            // --- Pilot run ----------------------------------------------------
            const std::size_t pilot_trials = std::max<std::size_t>(64, std::max<std::size_t>(trials, 4096));
            mc::queue::layer_manager<bitpack_t_> pilot_mgr(pdag, pilot_trials);
            const auto pilot_tally = pilot_mgr.tally(pdag->root()->index(), eps, conf); // early-stop allowed even for pilot
            const double phat      = pilot_tally.mean;

            trials = std::max<std::size_t>(
                        scram::mc::stats::required_trials(phat, eps, conf),
                        scram::mc::stats::worst_case_trials(eps, conf));

            // Round up to whole bit-packs so the scheduler is happy.
            constexpr std::size_t bits_in_pack = sizeof(bitpack_t_) * 8;
            trials = ((trials + bits_in_pack - 1) / bits_in_pack) * bits_in_pack;

            LOG(WARNING) << "[MC] auto-selected num_trials=" << trials
                      << " (pilot p̂=" << phat << ", ε=" << eps
                      << ", confidence=" << conf << ")";

            // Persist the choice so that downstream code sees the final value.
            st.num_trials(trials);
        }

        // ---------------------------------------------------------------------
        // 2) Full simulation with (possibly) auto-tuned sample size.
        // ---------------------------------------------------------------------
        mc::queue::layer_manager<bitpack_t_> manager(pdag, trials);

        const auto tally = manager.tally(pdag->root()->index(), eps, conf);

        LOG(WARNING) << "Calculated probability " << tally.mean << " in " << DUR(calc_time);
        return tally.mean;
    }
}// namespace scram::core