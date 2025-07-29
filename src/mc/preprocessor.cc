
#include "preprocessor.h"

#include "logger.h"

namespace scram::core {

// stages:: somewhat different from phases, as defined by rakhimov
// stage 0: do nothing
// stage 1: just remove nulls, and that's it.
// stage 2: normalize, expanding k/n to and/or
// always a good idea to run Phase2 multiple times if we care about compression.
void CustomPreprocessor<mc::DirectEval>::Run()  {

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
} // namespace scram::mc