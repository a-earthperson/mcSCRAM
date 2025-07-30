#include "preprocessor.h"

#include "logger.h"
#include "logger/log_benchmark.h"
#include "logger/log_build.h"
#include "logger/log_compressratio.h"
#include "logger/log_model.h"
#include "logger/log_pdag.h"
#include "logger/log_settings.h"
#include "version.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_set>

namespace scram::core {

// stages:: somewhat different from phases, as defined by rakhimov
// stage 0: do nothing
// stage 1: just remove nulls, and that's it.
// stage 2: normalize, expanding k/n to and/or
// always a good idea to run Phase2 multiple times if we care about compression.
void CustomPreprocessor<mc::DirectEval>::Run() {

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
    
    // Time the result() execution with high precision
    auto start_time = std::chrono::high_resolution_clock::now();
    result();
    auto end_time = std::chrono::high_resolution_clock::now();
    
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