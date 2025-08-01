#include "preprocessor.h"

#include "logger.h"
#include "logger/log_benchmark.h"
#include "logger/log_build.h"
#include "logger/log_compressratio.h"
#include "logger/log_model.h"
#include "logger/log_pdag.h"
#include "logger/log_settings.h"

#include <chrono>
#include <string>
#include <unordered_set>

namespace scram::core {

void CustomPreprocessor<mc::DirectEval>::Run() {

    graph_->Log();
    // perform the actual run
    auto result = [this] {
        const int compilation_target = this->settings_->compilation_level();

        TIMER(DEBUG2, "CustomPreprocessor<DirectEval>::");
        LOG(DEBUG3) << "Compilation Target: " << std::to_string(compilation_target);

        if (compilation_target <= 0) {
            return;
        }

        // remove null gates, absorb not gates
        core::pdag::Transform(graph_, [this](core::Pdag *) { RunPhaseOne(); });

        if (compilation_target <= 1) {
            return;
        }

        for (auto pass = 2; pass <= compilation_target; ++pass) {
            core::pdag::Transform(
                graph_, [this](core::Pdag *) { RunPhaseOne(); }, [this](core::Pdag *) { RunPhaseTwo(); },
                [this](core::Pdag *) {
                    if (!graph_->normal() && (settings_->expand_atleast_gates() || settings_->expand_xor_gates()))
                        RunPhaseThree();
                },
                [this](core::Pdag *) {
                    if (!graph_->coherent() && (settings_->expand_atleast_gates() || settings_->expand_xor_gates()))
                        RunPhaseFour();
                },
                [this](core::Pdag *) { RunPhaseFive(); });
        }
    };
    graph_->Log();
    // perform the actual run
    auto result2 = [this] {
        const int compilation_target = this->settings_->compilation_level();

        TIMER(DEBUG2, "CustomPreprocessor<DirectEval>::");
        LOG(DEBUG2) << "Compilation Target: " << std::to_string(compilation_target);
        graph_->Log();

        /*-------------------------------------------------------------*
         * Compilation Level 0
         * -------------------
         *  - do nothing
         *  - only expand KofN, XOR, if requested
         *------------------------------------------------------------*/
        {
            if (settings_->expand_atleast_gates()) {
                pdag::Transform(graph_,[this](Pdag *) { if (!graph_->normal()) NormalizeGates(ATLEAST); });
            }

            if (settings_->expand_xor_gates()) {
                pdag::Transform(graph_,[this](Pdag *) { if (!graph_->normal()) NormalizeGates(XOR); });
            }

            // do nothing
            if (compilation_target <= 0) {
                return;
            }
        }

        /*-------------------------------------------------------------*
         * Compilation Level 1
         * -------------------
         *  - Replace NULL gates with edges
         *  - Replace NOT gates with complemented edges
         *------------------------------------------------------------*/
        {
            pdag::Transform(graph_,[this](Pdag *) { NormalizeGates(NONE); });
            if (compilation_target <= 1) {
                return;
            }
        }

        /*-------------------------------------------------------------*
         * Compilation Level 2
         * -------------------
         *  - Process Multiple Definitions
         *  - Detect Modules
         *  - Coalesce Gates (not shared/common gates)
         *  - Merge Common Arguments
         *  - Coalesce Gates (including shared/common gates)
         *------------------------------------------------------------*/
        {
            pdag::Transform(graph_, [this](Pdag*) { while (ProcessMultipleDefinitions()) continue; },
            [this](Pdag*) { DetectModules(); },
            [this](Pdag*) { while (CoalesceGates(/*common=*/false)) continue; },
            [this](Pdag*) { MergeCommonArgs(); },
            [this](Pdag*) { while (CoalesceGates(/*common=*/true)) continue; });
            if (compilation_target <= 2) {
                return;
            }
        }

        /*-------------------------------------------------------------*
         * Compilation Level 3
         * -------------------
         *  - "PhaseTwo"
         *      - Process Multiple Definitions
         *      - Detect Modules
         *      - Coalesce Gates (not shared/common gates)
         *      - Merge Common Arguments
         *      - Detect Distributivity
         *      - Boolean Optimization (Shannon Expansion)
         *      - Decompose Common Nodes
         *      - Detect Modules
         *      - Coalesce Gates (not shared/common gates)
         *      - Detect Modules
         *  - Coalesce Gates (including shared/common gates)
         *------------------------------------------------------------*/
        {
            // coalesce gates, run phase 2 again, coalesce gates
            pdag::Transform(graph_, [this](Pdag*) { while (ProcessMultipleDefinitions()) continue; },
                    [this](Pdag*) { DetectModules(); },
                    [this](Pdag*) { while (CoalesceGates(/*common=*/false)) continue; },
                      [this](Pdag*) { MergeCommonArgs(); },
                      [this](Pdag*) { DetectDistributivity(); },
                      [this](Pdag*) { DetectModules(); },
                      [this](Pdag*) { BooleanOptimization(); },
                      [this](Pdag*) { DecomposeCommonNodes(); },
                      [this](Pdag*) { DetectModules(); },
                      [this](Pdag*) { while (CoalesceGates(/*common=*/false)) continue; },
                      [this](Pdag*) { DetectModules(); },
                      [this](Pdag*) { while (CoalesceGates(/*common=*/true)) continue; });
            if (compilation_target <= 3) {
                return;
            }
        }

        /*-------------------------------------------------------------*
         * Compilation Level 4 -- 8
         * -------------------
         *  - Rakhimov's Five Phase Approach to get to NNF
         *      - PhaseOne: Remove Nulls, Normalize Negations if Non-Coherent
         *      - PhaseTwo: (see compilation level 3)
         *      - PhaseThree:
         *          - Fully Normalize (Part A): replace K/N, XOR, NOT, NULL with AND/OR
         *          - Run PhaseTwo again
         *      - PhaseFour:
         *          - Fully Normalize (Part B): Push down negations, if not coherent
         *          - Run PhaseTwo again
         *      - PhaseFive:
         *          - Coalesce Common/Shared Gates
         *          - Run PhaseTwo Again
         *          - Coalesce Common/Shared Gates
         *  - We will stagger these stages based on compilation level.
         *  - You get to NNF at compilation level 8, if --no-kn, --no-xor were not set.
         *------------------------------------------------------------*/
        {
            pdag::Transform(
                graph_,
                [this, &compilation_target](Pdag *) {
                    /*-----------------------*
                     * Compilation Level 4+
                     *----------------------*/
                    if (compilation_target >= 4) {
                        RunPhaseOne();
                    }
                },
                [this, &compilation_target](Pdag *) {
                    /*-----------------------*
                     * Compilation Level 5+
                     *----------------------*/
                    if (compilation_target >= 5) {
                        RunPhaseTwo();
                    }
                },
                [this, &compilation_target](Pdag *) {
                    /*-----------------------*
                     * Compilation Level 6+
                     *----------------------*/
                    if (compilation_target >= 6) {
                        if (!graph_->normal() && (settings_->expand_atleast_gates() || settings_->expand_xor_gates())) {
                            RunPhaseThree();
                        }
                    }
                },
                [this, &compilation_target](Pdag *) {
                    /*-----------------------*
                     * Compilation Level 7+
                     *----------------------*/
                    if (compilation_target >= 7) {
                        if (!graph_->coherent() && (settings_->expand_atleast_gates() || settings_->expand_xor_gates()))
                            RunPhaseFour();
                        }
                    },
                [this, &compilation_target](Pdag *) {
                    /*-----------------------*
                     * Compilation Level 8+
                     *----------------------*/
                    if (compilation_target >= 8) {
                        RunPhaseFive();
                    }
                });
        }
    };
    
    // Time the result() execution with high precision
    auto start_time = std::chrono::high_resolution_clock::now();
    result();
    auto end_time = std::chrono::high_resolution_clock::now();

    graph_->Log();

    // Calculate duration in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    long long preprocessing_time_us = duration.count();
    
    // benchmark specific log
    {
        std::vector<std::pair<std::string, std::string>> kv;
        // build
        {
            auto s_pairs = log::build::csv_pairs();
            kv.insert(kv.end(), s_pairs.begin(), s_pairs.end());
        }
        // settings
        {
            auto s_pairs = log::settings::csv_pairs(*settings_);
            kv.insert(kv.end(), s_pairs.begin(), s_pairs.end());
        }
        // input model
        {
            auto s_pairs = log::model::csv_pairs(*settings_->model());
            kv.insert(kv.end(), s_pairs.begin(), s_pairs.end());
        }
        // pdag
        {
            auto s_pairs = log::pdag::csv_pairs(*graph_);
            kv.insert(kv.end(), s_pairs.begin(), s_pairs.end());
        }
        // compression factors from mef-model, pdag
        {
            log::compressratio::csv_pairs(kv);
        }
        
        // Add preprocessing time
        kv.emplace_back("preprocessing_time_us", log::csv_string(preprocessing_time_us));
        
        log::BenchmarkLogger compilation_logger{"compiler.csv"};
        compilation_logger.log_pairs(kv);
    }
}

auto core::CustomPreprocessor<mc::DirectEval>::remove_null_gates() const {
    if (graph_->HasNullGates()) {
        TIMER(DEBUG3, "NULL gates found");
        graph_->Log();
        if (this->settings_->keep_null_gates()) {
            TIMER(DEBUG3, "Keeping NULL gates");
        } else {
            TIMER(DEBUG3, "Removing NULL gates");
            graph_->RemoveNullGates();
        }
    }
}
} // namespace scram::core