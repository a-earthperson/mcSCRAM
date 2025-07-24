
#include "preprocessor.h"

#include "logger.h"

namespace scram::core {

// stages:: somewhat different from phases, as defined by rakhimov
// stage 0: do nothing
// stage 1: just remove nulls, and that's it.
// stage 2: normalize, expanding k/n to and/or
// always a good idea to run Phase2 multiple times if we care about compression.
void CustomPreprocessor<mc::DirectEval>::Run() noexcept {

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
    //             [this](core::Pdag*) {
    //               while (ProcessMultipleDefinitions())
    //                   continue;
    //               },
    //               [this](core::Pdag*) { DetectModules(); },
    //               [this](core::Pdag*) {
    //                 while (CoalesceGates(/*common=*/true))
    //                     continue;
    //                 },
    //                 [this](core::Pdag*) { MergeCommonArgs(); },
    //                 [this](core::Pdag*) { DetectDistributivity(); },
    //                 [this](core::Pdag*) { DetectModules(); },
    //                 // [this](core::Pdag*) { BooleanOptimization(); },
    //                 // [this](core::Pdag*) { DecomposeCommonNodes(); },
    //                 [this](core::Pdag*) {
    //                   while (CoalesceGates(/*common=*/true))
    //                       continue;
    //                   },
    //                   [this](core::Pdag*) { DetectModules(); },
    //                   [this](core::Pdag*) { if (settings_->expand_atleast_gates() && !graph_->normal()) {
    //                   RunPhaseThree(); }});
    // }
    // return;
    // if (compilation_target <=5) {
    //     return;
    // }
    // for (auto pass = 0; pass < compilation_target; ++pass) {
    //     LOG(DEBUG3) << "Compilation Pass: " << std::to_string(pass);
    //     pdag::Transform(graph_,
    //             [this](core::Pdag*){ while (ProcessMultipleDefinitions()) continue;},
    //             [this](core::Pdag*) { DetectModules(); },
    //             [this](core::Pdag*) { while (CoalesceGates(/*common=*/false)) continue; },
    //             [this](Pdag *) { MergeCommonArgs(); },
    //             [this](core::Pdag*) { while (CoalesceGates(/*common=*/true)) continue; },
    //             [this](Pdag *) { MergeCommonArgs(); },
    //             [this](core::Pdag*) { DetectModules(); },
    //              [this](core::Pdag*) { while (CoalesceGates(/*common=*/true)) continue; });
    // }

    // while (compilation_level != compilation_target) {
    //     switch (compilation_level++) {
    //         case 1:
    //             pdag::Transform(graph_, [this](core::Pdag*){ RunPhaseOne();});
    //             break;
    //         case 2:
    //             pdag::Transform(graph_,
    //                     [this](core::Pdag*){ RunPhaseOne();},
    //                             [this](core::Pdag*){ while (ProcessMultipleDefinitions()) continue;},
    //                             [this](core::Pdag*) { DetectModules(); },
    //                             [this](core::Pdag*) { while (CoalesceGates(/*common=*/false)) continue; },
    //                             [this](Pdag *) { MergeCommonArgs(); },
    //                             [this](core::Pdag*) { while (CoalesceGates(/*common=*/true)) continue; },
    //                             [this](Pdag *) { MergeCommonArgs(); },
    //                             [this](core::Pdag*) { DetectModules(); },
    //                              [this](core::Pdag*) { while (CoalesceGates(/*common=*/true)) continue; });
    //             break;
    //         case 3:
    //             pdag::Transform(graph_, [this](core::Pdag*){ if (!graph_->normal()) RunPhaseThree(); });
    //             break;
    //         case 4:
    //             pdag::Transform(graph_, [this](core::Pdag*){ if (!graph_->coherent()) RunPhaseFour(); });
    //             break;
    //         case 5:
    //             pdag::Transform(graph_, [this](core::Pdag*){ RunPhaseFive(); });
    //             break;
    //         case 0:
    //             pdag::Transform(graph_, [this](core::Pdag*){ remove_null_gates();});
    //             break;
    //         case -1:
    //         default:
    //             break;
    //     }
    // }
    // return;
    // TIMER(DEBUG2, "CustomPreprocessor<DirectEval>:: Running Transform Phases I, II with no normalization, followed by
    // layered toposort..."); pdag::Transform(graph_,
    //     [this](core::Pdag*) {
    //         if (graph_->HasNullGates()) {
    //             TIMER(DEBUG3, "NULL gates found");
    //             graph_->Log();
    //             if (this->settings_->keep_null_gates()) {
    //                 TIMER(DEBUG3, "Keeping NULL gates");
    //             } else {
    //                 TIMER(DEBUG3, "Removing NULL gates");
    //                 graph_->RemoveNullGates();
    //             }
    //         }
    //     },
    //     [this](core::Pdag*){ RunPhaseOne(); },
    //             [this](core::Pdag*){ RunPhaseTwo(); },
    //             [this](core::Pdag*) {
    //               if (!graph_->normal())
    //                   RunPhaseThree();
    //               },
    //               [this](core::Pdag*) {
    //                 if (!graph_->coherent())
    //                   RunPhaseFour();
    //               },
    //             [this](core::Pdag*){ RunPhaseFive();}
    //             );
    //
    // pdag::Transform(graph_,
    //         [this](core::Pdag*) {
    //           while (ProcessMultipleDefinitions())
    //               continue;
    //           },
    //           [this](core::Pdag*) { DetectModules(); },
    //           [this](core::Pdag*) {
    //             while (CoalesceGates(/*common=*/false))
    //                 continue;
    //             },
    //             [this](core::Pdag*) { MergeCommonArgs(); },
    //             [this](core::Pdag*) { DetectDistributivity(); },
    //             [this](core::Pdag*) { DetectModules(); },
    //             [this](core::Pdag*) { BooleanOptimization(); },
    //             [this](core::Pdag*) { DecomposeCommonNodes(); },
    //             [this](core::Pdag*) { DetectModules(); },
    //             [this](core::Pdag*) {
    //               while (CoalesceGates(/*common=*/false))
    //                   continue;
    //               },
    //               [this](core::Pdag*) { DetectModules(); });
    // pdag::Transform(graph_, &pdag::LayeredTopologicalOrder);
    // Preprocessor::Run();
    // pdag::Transform(graph_,
    //                 [this](core::Pdag*) {
    //                   if (!graph_->coherent())
    //                       RunPhaseFour();
    //                   },
    //                   [this](core::Pdag*) { RunPhaseFive(); }, &pdag::MarkCoherence,
    //                   &pdag::TopologicalOrder);
    // pdag::Transform(graph_, [this](core::Pdag*) { InvertOrder(); });
    //                   [this](core::Pdag*) { RunPhaseTwo(); },
    //                   [this](core::Pdag*) { RunPhaseFive(); },
    //                   &pdag::MarkCoherence,
    //                   &pdag::TopologicalOrder);
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