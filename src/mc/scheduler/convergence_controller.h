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

#include <unistd.h>  // isatty
#include <string>
#include <sstream>
#include <optional>
#include <iomanip>

namespace scram::mc::queue {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
class convergence_controller {
  public:
    using index_t_ = std::int32_t;

    static std::size_t cumulative_bits(const event::sample_shape<std::size_t> &shape, const size_t &iteration) {
        return iteration * shape.num_bitpacks() * sizeof(bitpack_t_) * 8;
    }

    [[nodiscard]] bool iteration_limit_reached() const {
        return iteration_ >= max_iterations_ && (max_iterations_ > 0);
    }

    /**
     * @param mgr        Reference to a fully initialised layer_manager.
     * @param evt_idx    Index of the event whose probability we track.
     * @param settings   Settings
     */
    convergence_controller(layer_manager<bitpack_t_, prob_t_, size_t_> &mgr,
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

        trials_per_iteration_ = cumulative_bits(manager_.sample_shape_, 1);
        max_trials_ = settings.num_trials();
        trials_complete_ = 0;
    }

    // ---------------------------------------------------------------------------------
    //  Logging helper – prints *one* status line with all diagnostics.
    // ---------------------------------------------------------------------------------
    void log_progress(const double half_width,
                      const std::optional<mc::stats::AccuracyMetrics> &acc,
                      const std::optional<mc::stats::SamplingDiagnostics> &diag,
                      const std::string &suffix) const {

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
        oss << std::setprecision(6);

        oss << "tally[" << evt_idx_ << "] :: ["
            << ci_col << current_tally_.ci[2] << reset_col << ", "
            << ci_col << current_tally_.ci[0] << reset_col << ", "
            << mean_col << current_tally_.mean << reset_col << ", "
            << ci_col << current_tally_.ci[1] << reset_col << ", "
            << ci_col << current_tally_.ci[3] << reset_col << "] :: ";

        // add requested CI info
        oss << "CI(" << targets_.two_sided_confidence_level * 100.0 << ") :: ";

        // half-width
        if (half_width >= 0.0) {
            oss << "ε[" << half_width << "] :: ";
        }

        // accuracy metrics
        if (acc) {
            oss << "|Δ|=" << acc->abs_error << ", rel_err=" << acc->rel_error;
        }

        // diagnostics
        if (diag) {
            oss << ", z=" << diag->z_score << " :: p=" << diag->p_value
                << " :: CI(95)=" << (diag->ci95_covered ? "✔" : "✘")
                << " :: CI(99)=" << (diag->ci99_covered ? "✔" : "✘");
            if (!std::isnan(diag->n_ratio)) {
                oss << " :: n_ratio=" << diag->n_ratio;
            }
            oss << " :: ";
        }

        oss << suffix;

        LOG(DEBUG2) << oss.str();
    }

    /** Execute exactly one additional iteration on the device. */
    [[nodiscard]] bool step() {

        // don't step anymore, just return that we didn't take a step.
        if (converged_ && stop_on_convergence_) {
            return false;
        }

        // out of iterations, return that we didn't take a step.
        if (iteration_limit_reached()) {
            return false;
        }

        // still have iterations remaining
        // get the tally
        current_tally_ = manager_.single_pass_and_tally(evt_idx_);

        // ---------------------------------------------------------------------
        //  Host-side statistical post-processing
        // ---------------------------------------------------------------------
        // The Monte-Carlo kernel only updates `num_one_bits` and `total_bits`.
        // We complete the statistics on the host so that the device kernel does
        // no redundant work (especially when several work-groups process the
        // same tally).

        stats::populate_point_estimates(current_tally_);

        current_.half_width_epsilon = stats::half_width(current_tally_, current_.normal_quantile_two_sided);

        // for now, this is our convergence criteria
        const bool epsilon_bounded = current_.half_width_epsilon <= targets_.half_width_epsilon && current_.half_width_epsilon > 0.0;

        // if converged now, set convergence_ sticky to true
        if (!converged_ && epsilon_bounded) {
            converged_ = true;
        }

        // ------------------------------------------------------------------
        //  Diagnostic metrics (if ground truth provided)
        // ------------------------------------------------------------------
        if (enable_diagnostics_) {
            std::optional<scram::mc::stats::AccuracyMetrics> acc_opt;
            std::optional<scram::mc::stats::SamplingDiagnostics> diag_opt;
            acc_opt  = scram::mc::stats::compute_accuracy_metrics(current_tally_, ground_truth_);
            diag_opt = scram::mc::stats::compute_sampling_diagnostics(current_tally_, ground_truth_, targets_);
            // Log progress for this step (single consolidated line)
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
        }

        // Final log with iteration count (no half-width / diagnostics)
        log_progress(current_.half_width_epsilon, std::nullopt, std::nullopt, "Iterations :: " + std::to_string(iteration_));
        return current_tally_;
    }

    [[nodiscard]] bool converged() const { return converged_; }

    [[nodiscard]] std::size_t iterations_completed() const { return iteration_; }

    [[nodiscard]] const event::tally<bitpack_t_> &current_tally() const { return current_tally_; }

  private:

    layer_manager<bitpack_t_, prob_t_, size_t_> &manager_;
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
    event::tally<bitpack_t_> current_tally_{};

    std::size_t trials_per_iteration_ = 0;
    std::size_t max_trials_ = 0;
    std::size_t trials_complete_ = 0;
};

} // namespace scram::mc::queue