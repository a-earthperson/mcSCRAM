/**
 * @file convergence_controller.h
 * @brief High-level convergence manager for Monte-Carlo sampling runs.
 *
 * @details  This helper class delegates the *numerical* work to an existing
 * scram::mc::queue::layer_manager instance but owns the *policy* of how many
 * additional iterations to execute.  It centralises the classical
 * "half-width ≤ ε with CLT sanity" early-stopping rule that was previously
 * embedded in layer_manager.
 *
 * The controller can be used in two different ways:
 *   1. Call step() repeatedly from user code while interrogating the current
 *      tally after each pass – useful for detailed diagnostics.
 *   2. Call run_to_convergence() to let the controller iterate until either
 *      the requested precision is achieved or all planned iterations have
 *      been executed.
 *
 * The implementation is header-only and therefore fully inlined for maximum
 * flexibility across translation units.
 */
#pragma once

#include "mc/queue/layer_manager.h"
#include "mc/stats/ci_utils.h"
#include "mc/stats/diagnostics.h"
#include "mc/scheduler/progressbar.h"

#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>

#define PRECISION_LOG_SCIENTIFIC_DIGITS 3

namespace scram::mc {
template<typename bitpack_t_>
struct iteration_shape {
public:
    std::size_t iterations{};
    std::size_t trials;
    template<event::sample_shape<std::size_t> shape>
    [[nodiscard]] iteration_shape(const std::size_t trials)
        : trials(trials), shape_(shape) {}

    [[nodiscard]] static std::size_t cumulative_bits(const event::sample_shape<std::size_t> &shape,
                                                     const size_t &iteration = 1) {
        return iteration * shape.num_bitpacks() * sizeof(bitpack_t_) * 8;
    }
    void set_iterations(const std::size_t iterations) {
        this->trials = cumulative_bits(shape_, iterations);
    }
    void set_trials(const std::size_t trials) { this->trials = trials; }

    [[nodiscard]] static std::size_t trials_per_iteration(const event::sample_shape<std::size_t> &shape) {
        return cumulative_bits(shape, 1);
    }

    [[nodiscard]] static std::size_t iterations_from_trials(const std::size_t trials, const std::size_t trials_per_iteration) {
        return static_cast<std::size_t>(std::ceil(static_cast<std::double_t>(trials) / static_cast<std::double_t>(trials_per_iteration)));
    }
private:
    event::sample_shape<std::size_t> shape_;
};

template <typename DataT>
struct tracked_pair {
    DataT current;
    DataT target;
};

template <typename DataT>
struct tracked_triplet {
    DataT current;
    DataT target;
    DataT remaining;
};

}

namespace scram::mc::scheduler {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
class convergence_controller {
  public:
    using index_t_ = std::int32_t;

    /**
     * @param mgr        Reference to a fully initialised layer_manager.
     * @param evt_idx    Index of the event whose probability we track.
     * @param settings   Settings
     */
    convergence_controller(queue::layer_manager<bitpack_t_, prob_t_, size_t_> &mgr,
                           const index_t_ evt_idx,
                           const core::Settings &settings)
        : manager_(mgr), evt_idx_(evt_idx), settings_(settings) {

        // --- parametrize δ as a function of ε ------------------------------------------------------------
        // ε = |µ - p̂|
        // ε = δ•p̂
        // we don't have a target ε yet since it's a value that gets computed and updated after we run some initial
        // trials. So, we make the target postpone convergence checks until the pilot phase finishes.
        interval_ = {
            .current = {
                .half_width_epsilon = std::numeric_limits<std::double_t>::infinity(), // not been computed so far, max
                .two_sided_confidence_level = std::numeric_limits<std::double_t>::quiet_NaN(), // not needed, not tracked for now.
                .normal_quantile_two_sided = std::numeric_limits<std::double_t>::quiet_NaN(),
            },
            .target = {
                .half_width_epsilon = settings_.ci_rel_margin_error() * std::max(0.0, stats::DELTA_EPSILON),
                .two_sided_confidence_level = settings_.ci_confidence(), // from settings
                .normal_quantile_two_sided = stats::normal_quantile_two_sided(settings_.ci_confidence()), // compute once
            },
        };
        progress_.initialize(this, settings.watch_mode());
    }

    void update_stats(const event::tally<bitpack_t_> &new_tally) {

        const std::double_t &target_z = target_zscore();

        // The half_width helper expects the *Z*-score. We pre-computed the
        // two-sided normal quantile when initialising `targets_`, so use it
        // here instead of the confidence level itself.
        interval_.current.half_width_epsilon = stats::half_width(new_tally, target_z);

        // set the target epsilon as a fraction of the estimated mean
        interval_.target.half_width_epsilon = settings_.ci_rel_margin_error() * std::max(new_tally.mean, stats::DELTA_EPSILON);

        assert(current_tally_.total_bits == trials_completed_so_far() && "mismatch in number of computed vs tracked trials");

        const iteration_shape target = estimate_trials_for_ci(current_tally().mean, target_state());
    }

    void update_targets() {

    }
    void update_projections() {

    }

    void process_tally(const event::tally<bitpack_t_> &new_tally) {
        current_tally_ = new_tally;
        update_stats(new_tally);
    }

    [[nodiscard]] bool check_convergence() const {
        return check_epsilon_bounded();
    }

    [[nodiscard]] bool check_epsilon_bounded() const {
        return current_.half_width_epsilon <= targets_.half_width_epsilon && current_.half_width_epsilon > 0.0;
    }

    /** Execute exactly one additional iteration on the device. */
    [[nodiscard]] bool step() {

        // don't step anymore, just return that we didn't take a step.
        if (converged_ && settings_.early_stop()) {
            return false;
        }

        // out of iterations, return that we didn't take a step.
        // evals to false if max_iterations_ is 0, which means keep going.
        if (iteration_limit_reached()) {
            progress_.mark_iterations_complete();
            return false;
        }

        // still have iterations remaining
        // get the tally
        process_tally(manager_.single_pass_and_tally(evt_idx_));

        // for now, this is our convergence criteria
        // if converged now, set convergence_ sticky to true
        if (!converged_ && check_convergence()) {
            converged_ = true;
            progress_.mark_converged();
        }

        // since we did step, update the iteration count
        return ++iteration_;
    }

    /**
     * Execute exactly one additional asynchronous iteration on the device, don't worry about getting the tallies yet
     * since they are accumulating on device. The iteration count keeps track of how many trials have been run on device
     * so far.
     */
    [[nodiscard]] bool pilot_step() {

        // iteration_ now shows that enough tasks have been queued on device such that by the time they finish,
        // pilot trials would be complete. So, don't queue up any additional work, just return saying you're done taking
        // pilot steps.
        if (pilot_trials_complete()) {
            return false;
        }

        process_tally(manager_.single_pass_and_tally(evt_idx_));

        // since we did step, update the iteration count
        return ++iteration_;
    }

    /**
     * Run until the stopping criterion is met (or until the original plan is
     * exhausted).  Returns the final tally.
     */
    [[nodiscard]] event::tally<bitpack_t_> run_to_convergence() {
        // queue up pilot trials, but dont check for convergence.
        while(pilot_step()) {
            progress_.tick(this);
        }
        while (step()) {
            progress_.tick(this);
        }
        return current_tally();
    }

private:
    queue::layer_manager<bitpack_t_, prob_t_, size_t_> &manager_;
    const index_t_ evt_idx_;

    const core::Settings &settings_;

    tracked_pair<stats::ci> interval_;

    tracked_pair<iteration_shape> steps_;

    //stats::ci targets_{};
    //stats::ci current_{};

    //std::double_t ground_truth_;

    // User-supplied convergence parameters.

    // Derived constants.
    //bool stop_on_convergence_ = false;
    //std::size_t max_iterations_ = 0;

    // State bookkeeping.
    // std::size_t iteration_ = 0;
    bool converged_ = false;

    // std::vector<event::tally<bitpack_t_>> tallies_;
    event::tally<bitpack_t_> current_tally_{};

    //std::size_t trials_per_iteration_ = 0;
    //std::size_t max_trials_ = 0;

    // --- Relative ε state -------------------------------------------------
    //double       rel_epsilon_  = -1.0;  ///< δ: relative half-width requested.
    //std::size_t  pilot_trials_ = 0;     ///< free pilot iterations.

    progress<bitpack_t_, prob_t_, size_t_> progress_;

public:
    [[nodiscard]] bool diagnostics_enabled() const { return settings_.oracle_p() >= 0.0; }
    [[nodiscard]] std::double_t ground_truth() const { return settings_.oracle_p(); }


    [[nodiscard]] const iteration_shape &current_steps() const { return steps_.current; }
    [[nodiscard]] const iteration_shape &projected_steps() const { return steps_.target; }
    [[nodiscard]] const iteration_shape &remaining_steps() const {
        return {
            .iterations = projected_steps().iterations - current_steps().iterations,
            .trials = projected_steps().trials - current_steps().trials,
        };
    }

    [[nodiscard]] bool stop_on_convergence() const { return settings_.early_stop(); }
    [[nodiscard]] bool converged() const { return converged_; }
    [[nodiscard]] tracked_triplet<iteration_shape> convergence_status() const {
        return tracked_triplet{
            .current = current_steps(),
            .target = projected_steps(),
            .remaining = remaining_steps(),
        };
    }

    [[nodiscard]] const std::double_t &target_zscore() const { return interval_.target.normal_quantile_two_sided; }

    [[nodiscard]] const stats::ci &target_state() const { return interval_.target; }
    [[nodiscard]] const stats::ci &current_state() const { return interval_.current; }

    [[nodiscard]] const event::tally<bitpack_t_> &current_tally() const { return current_tally_; }



    [[nodiscard]] std::optional<stats::AccuracyMetrics> accuracy_metrics() const {
        std::optional<stats::AccuracyMetrics> metrics;
        if (diagnostics_enabled()) {
            metrics = stats::compute_accuracy_metrics(current_tally(), settings_.oracle_p());
        }
        return metrics;
    }
    [[nodiscard]] std::optional<stats::SamplingDiagnostics> sampling_diagnostics() const {
        std::optional<stats::SamplingDiagnostics> sampling_diagnostics;
        if (diagnostics_enabled()) {
            sampling_diagnostics = stats::compute_sampling_diagnostics(current_tally(), settings_.oracle_p(), target_state());
        }
        return sampling_diagnostics;
    }

    [[nodiscard]] std::size_t trials_per_iteration() const {
        return cumulative_bits(manager_.shaper().SAMPLE_SHAPE, 1);
    }

    [[nodiscard]] bool iteration_limit_reached() const {
        return max_iterations() && iteration_ >= max_iterations() ;
    }

    [[nodiscard]] std::size_t trials_completed_so_far() const {
        return cumulative_bits(manager_.shaper().SAMPLE_SHAPE, iteration_);
    }

    [[nodiscard]] bool pilot_trials_complete() const {
        return trials_completed_so_far() >= settings_.ci_burnin_trials();
    }

    // [[nodiscard]] std::size_t max_iterations() const {
    //     return manager_.shaper().TOTAL_ITERATIONS;
    // }
};
} // namespace scram::mc::queue