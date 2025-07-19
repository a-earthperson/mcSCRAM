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
#include <unistd.h>  // isatty
#include <string>
#include <sstream>

namespace scram::mc::queue {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
class convergence_controller {
  public:
    using index_t_ = std::int32_t;

    static std::size_t cumulative_bits(const event::sample_shape<std::size_t> &shape, const size_t &iteration) {
        return iteration * shape.num_bitpacks() * sizeof(bitpack_t_) * 8;
    }

    bool iteration_limit_reached() const {
        return iteration_ >= max_iterations_ && (max_iterations_ > 0);
    }

    /**
     * @param mgr        Reference to a fully initialised layer_manager.
     * @param evt_idx    Index of the event whose probability we track.
     * @param settings   Settings
     */
    convergence_controller(layer_manager<bitpack_t_, prob_t_, size_t_> &mgr, const index_t_ evt_idx, const core::Settings &settings)
        : manager_(mgr), evt_idx_(evt_idx), max_iterations_(mgr.shaper().TOTAL_ITERATIONS) {

        target_epsilon_ = std::abs(settings.ci_margin_error());                    // half-width ε
        confidence_ = std::clamp(settings.ci_confidence(), 0.0, 1.0);       // confidence level (two-sided)
        z_score_ = scram::mc::stats::z_score(confidence_);
        stop_on_convergence_ = settings.early_stop();                              // stop on convergence
    }

    // (colour constants defined in the private helper below)

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

        const double half_width = stats::half_width(current_tally_, z_score_);

        // for now, this is our convergence criteria
        const bool is_normal = stats::normal_approx_ok(current_tally_);
        const bool epsilon_bounded = half_width <= target_epsilon_;
        const bool converged_now = is_normal && epsilon_bounded;

        // if converged now, set convergence_ sticky to true
        if (!converged_ && converged_now) {
            converged_ = true;
        }

        // Log progress for this step.
        {
            std::ostringstream tag;
            tag << "ε[" << half_width << "]";
            log_status(tag.str());
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

        // Final log with iteration count
        {
            std::ostringstream tag;
            tag << "Iterations :: " << iteration_;
            log_status(tag.str());
        }
        return current_tally_;
    }

    [[nodiscard]] bool converged() const { return converged_; }

    [[nodiscard]] std::size_t iterations_completed() const { return iteration_; }

    [[nodiscard]] const event::tally<bitpack_t_> &current_tally() const { return current_tally_; }

  private:
    // Helper that prints the coloured status line.  The suffix argument lets the
    // caller append context-specific information (e.g. ε-half-width or number
    // of iterations).
    void log_status(const std::string &suffix) const {
        // ANSI colours (only if stderr is a TTY)
        constexpr const char *RED   = "\033[31m";
        constexpr const char *GREEN = "\033[32m";
        constexpr const char *YELL  = "\033[33m";
        constexpr const char *CYAN  = "\033[36m";
        constexpr const char *RESET = "\033[0m";

        const bool colourise = isatty(fileno(stderr));
        const char *mean_col  = colourise ? (converged_ ? GREEN : RED) : "";
        const char *std_col   = colourise ? YELL  : "";
        const char *ci_col    = colourise ? CYAN  : "";
        const char *reset_col = colourise ? RESET : "";

        LOG(DEBUG1) << "tally[" << evt_idx_ << "] :: "
                     << "[std_err] :: " << std_col << current_tally_.std_err << reset_col << " :: "
                     << "[p01, p05, mean, p95, p99] :: ["
                     << ci_col << current_tally_.ci[2] << reset_col << ", "
                     << ci_col << current_tally_.ci[0] << reset_col << ", "
                     << mean_col << current_tally_.mean << reset_col << ", "
                     << ci_col << current_tally_.ci[1] << reset_col << ", "
                     << ci_col << current_tally_.ci[3] << reset_col << "] :: "
                     << "CI(" << confidence_ * 100.0 << "% ) :: " << suffix;

        // If weighted stats are present, print a second bracketed block
        if (current_tally_.total_weight > 0.0) {
            LOG(DEBUG1) << "tally[" << evt_idx_ << "] :: WEIGHTED :: [std_err] :: "
                        << std_col << current_tally_.weighted_std_err << reset_col << " :: ["
                        << ci_col << current_tally_.weighted_ci[2] << reset_col << ", "
                        << ci_col << current_tally_.weighted_ci[0] << reset_col << ", "
                        << mean_col << current_tally_.weighted_mean << reset_col << ", "
                        << ci_col << current_tally_.weighted_ci[1] << reset_col << ", "
                        << ci_col << current_tally_.weighted_ci[3] << reset_col << "] :: "
                        << "CI(" << confidence_ * 100.0 << "% ) :: " << suffix;
        }
    }

    layer_manager<bitpack_t_, prob_t_, size_t_> &manager_;
    const index_t_ evt_idx_;

    // User-supplied convergence parameters.
    /**
     * Desired half-width of the confidence interval.
     */
    double target_epsilon_;
    /**
     * confidence level (two-sided) requested by the user
     */
    double confidence_;

    // Derived constants.
    bool stop_on_convergence_ = false;
    double z_score_ = 0.0; // z-score for the requested confidence
    std::size_t max_iterations_ = 0;

    // State bookkeeping.
    std::size_t iteration_ = 0;
    bool converged_ = false;
    event::tally<bitpack_t_> current_tally_{};
};

} // namespace scram::mc::queue