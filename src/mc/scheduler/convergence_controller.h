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

#include <unistd.h>  // isatty
#include <string>
#include <sstream>
#include <optional>
#include <iomanip>
#include <memory>
#include <vector>
#include <cmath>

#define PRECISION_LOG_SCIENTIFIC_DIGITS 3

namespace scram::mc::scheduler {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
class convergence_controller {
  public:
    using index_t_ = std::int32_t;

    static std::size_t cumulative_bits(const event::sample_shape<std::size_t> &shape, const size_t &iteration) {
        return iteration * shape.num_bitpacks() * sizeof(bitpack_t_) * 8;
    }

    [[nodiscard]] bool iteration_limit_reached() const {
        return max_iterations_ && (iteration_ >= max_iterations_) ;
    }

    /**
     * @param mgr        Reference to a fully initialised layer_manager.
     * @param evt_idx    Index of the event whose probability we track.
     * @param settings   Settings
     */
    convergence_controller(queue::layer_manager<bitpack_t_, prob_t_, size_t_> &mgr,
                           const index_t_ evt_idx,
                           const core::Settings &settings)
        : manager_(mgr), evt_idx_(evt_idx), ground_truth_(settings.true_prob()), max_iterations_(mgr.shaper().TOTAL_ITERATIONS) {

        targets_ = {
            .half_width_epsilon = settings.ci_margin_error(),
            .two_sided_confidence_level = settings.ci_confidence(),
            .normal_quantile_two_sided = mc::stats::normal_quantile_two_sided(settings.ci_confidence()),
        };

        current_ = {
            .half_width_epsilon = MAXFLOAT,
            .two_sided_confidence_level = targets_.two_sided_confidence_level,
            .normal_quantile_two_sided = targets_.normal_quantile_two_sided,
        };

        enable_diagnostics_ = settings.true_prob() >= 0.0;
        stop_on_convergence_ = settings.early_stop();

        trials_per_iteration_ = cumulative_bits(manager_.shaper().SAMPLE_SHAPE, 1);
        max_trials_ = settings.num_trials();

        tallies_.reserve(this->max_iterations() ? this->max_iterations() : 1000);
        progress_.initialize(this);
    }

    void update_stats(const event::tally<bitpack_t_> &new_tally) {
        current_.half_width_epsilon = stats::half_width(new_tally, targets_.two_sided_confidence_level);
    }

    void process_tally(const event::tally<bitpack_t_> &new_tally) {
        tallies_.push_back(new_tally);
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
        if (converged_ && stop_on_convergence_) {
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

        // ------------------------------------------------------------------
        //  Diagnostic metrics (if ground truth provided)
        // ------------------------------------------------------------------
        if (enable_diagnostics_) {
            std::optional<scram::mc::stats::AccuracyMetrics> acc_opt;
            std::optional<scram::mc::stats::SamplingDiagnostics> diag_opt;
            acc_opt  = scram::mc::stats::compute_accuracy_metrics(current_tally(), ground_truth_);
            diag_opt = scram::mc::stats::compute_sampling_diagnostics(current_tally(), ground_truth_, targets_);
        //     // Log progress for this step (single consolidated line)
             log_progress(current_.half_width_epsilon, acc_opt, diag_opt, "iter=" + std::to_string(iteration_));
         }

        // since we did step, update the iteration count
        return ++iteration_;
    }

    /**
     * Run until the stopping criterion is met (or until the original plan is
     * exhausted).  Returns the final tally.
     */
    [[nodiscard]] event::tally<bitpack_t_> run_to_convergence() {

        while (step()) {
            progress_.tick(this);
        }

        //progress_.mark_iterations_complete();
        // Final log with iteration count (no half-width / diagnostics)
        //log_progress(current_.half_width_epsilon, std::nullopt, std::nullopt, "Iterations :: " + std::to_string(iteration_));

        return tallies_.back();
    }

private:
    queue::layer_manager<bitpack_t_, prob_t_, size_t_> &manager_;
    const index_t_ evt_idx_;

    stats::ci targets_{};
    stats::ci current_{};

    bool enable_diagnostics_ = false;

    std::double_t ground_truth_;

    // User-supplied convergence parameters.

    // Derived constants.
    bool stop_on_convergence_ = false;
    std::size_t max_iterations_ = 0;

    // State bookkeeping.
    std::size_t iteration_ = 0;
    bool converged_ = false;

    std::vector<event::tally<bitpack_t_>> tallies_;

    std::size_t trials_per_iteration_ = 0;
    std::size_t max_trials_ = 0;

    progress<bitpack_t_, prob_t_, size_t_> progress_;

    void log_progress(const double half_width,
                      const std::optional<mc::stats::AccuracyMetrics> &acc,
                      const std::optional<mc::stats::SamplingDiagnostics> &diag,
                      const std::string &suffix) {

        // ANSI colours (only if stderr is a TTY)
        constexpr const char *RED   = "\033[31m";
        constexpr const char *GREEN = "\033[32m";
        constexpr const char *YELL  = "\033[33m";
        constexpr const char *CYAN  = "\033[36m";
        constexpr const char *RESET = "\033[0m";

        const bool colorize = isatty(fileno(stderr));
        const char *mean_col  = colorize ? (converged_ ? GREEN : RED) : "";
        const char *std_col   = colorize ? YELL  : "";
        const char *ci_col    = colorize ? CYAN  : "";
        const char *reset_col = colorize ? RESET : "";

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS);

        oss << "tally[" << evt_idx_ << "] :: ["
            << ci_col << current_tally().ci[2] << reset_col << ", "
            << ci_col << current_tally().ci[0] << reset_col << ", "
            << mean_col << current_tally().mean << reset_col << ", "
            << ci_col << current_tally().ci[1] << reset_col << ", "
            << ci_col << current_tally().ci[3] << reset_col << "] :: ";

        // add requested CI info
        oss << "CI(" << targets_.two_sided_confidence_level * 100.0 << ") :: ";

        // half-width
        if (half_width >= 0.0) {
            oss << "ε[" << half_width << "] :: ";
        }

        // accuracy metrics
        if (acc) {
            oss << *acc << " :: ";
        }

        // diagnostics
        if (diag) {
            oss << *diag << " :: ";
        }

        oss << suffix;

        LOG(DEBUG2) << oss.str();
    }

public:
    [[nodiscard]] bool diagnostics_enabled() const { return enable_diagnostics_; }
    [[nodiscard]] std::double_t ground_truth() const { return ground_truth_; }
    [[nodiscard]] bool stop_on_convergence() const { return stop_on_convergence_; }
    [[nodiscard]] std::size_t iterations() const { return iteration_; }
    [[nodiscard]] bool converged() const { return converged_; }
    [[nodiscard]] std::size_t trials_per_iteration() const { return trials_per_iteration_; }
    [[nodiscard]] std::size_t max_trials() const { return max_trials_; }
    [[nodiscard]] stats::ci targets() const { return targets_; }
    [[nodiscard]] stats::ci current() const { return current_; }
    [[nodiscard]] std::size_t max_iterations() const { return max_iterations_; }
    [[nodiscard]] const event::tally<bitpack_t_> &current_tally() const { return tallies_.back(); }
};
} // namespace scram::mc::queue