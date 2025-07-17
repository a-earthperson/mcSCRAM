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

namespace scram::mc::queue {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
class convergence_controller {
  public:
    using index_t_ = std::int32_t;

    /**
     * @param mgr        Reference to a fully initialised layer_manager.
     * @param evt_idx    Index of the event whose probability we track.
     * @param eps        Desired half-width of the confidence interval.  If 0 the
     *                   controller will *not* early-stop.
     * @param confidence Confidence level (two-sided) requested by the user –
     *                   e.g. 0.95 for a 95 % CI.
     */
    convergence_controller(layer_manager<bitpack_t_, prob_t_, size_t_> &mgr, const index_t_ evt_idx, const double eps,
                           const double confidence)
        : manager_(mgr), evt_idx_(evt_idx), eps_(eps), confidence_(confidence) {
        use_early_stop_ = (eps_ > 0.0) && (confidence_ > 0.0);
        z_ = scram::mc::stats::z_score(confidence_);
        max_iterations_ = manager_.shaper().TOTAL_ITERATIONS;
    }

    /** Execute exactly one additional iteration on the device. */
    [[nodiscard]] bool step() {
        if (converged_ || iteration_ >= max_iterations_) {
            return converged_;
        }

        // get the tally
        current_tally_ = manager_.single_pass_and_tally(evt_idx_);
        ++iteration_;

        // compute the tally-related metrics
        const double half_width = z_ * current_tally_.std_err;
        const std::size_t bits_so_far = iteration_ * manager_.shaper().SAMPLE_SHAPE.num_bitpacks() * sizeof(bitpack_t_) * 8;

        LOG(WARNING) << "tally[" << evt_idx_ << "] :: [std_err] :: [p05, mean, p95] :: [" << current_tally_.std_err << "] :: ["
            << current_tally_.ci[0] << ", " << current_tally_.mean << ", "
            << current_tally_.ci[1] << "] :: CI(" << confidence_ * 100.0 << "%) :: ε["
            << half_width << "]";

        // stop early if convergence criteria met
        if (use_early_stop_ && half_width <= eps_ && scram::mc::stats::clt_ok(bits_so_far, current_tally_.mean)) {
            converged_ = true;
        }

        // Implicitly converged when we have exhausted all planned iterations.
        if (iteration_ >= max_iterations_) {
            converged_ = true;
        }
        return converged_;
    }

    /**
     * Run until the stopping criterion is met (or until the original plan is
     * exhausted).  Returns the final tally.
     */
    [[nodiscard]] event::tally<bitpack_t_> run_to_convergence() {
        while (!step()) {
        }
        return current_tally_;
    }

    [[nodiscard]] bool converged() const { return converged_; }

    [[nodiscard]] std::size_t iterations_completed() const { return iteration_; }

    [[nodiscard]] const event::tally<bitpack_t_> &current_tally() const { return current_tally_; }

  private:
    layer_manager<bitpack_t_, prob_t_, size_t_> &manager_;
    const index_t_ evt_idx_;

    // User-supplied convergence parameters.
    const double eps_;
    const double confidence_;

    // Derived constants.
    bool use_early_stop_ = false;
    double z_ = 0.0; // z-score for the requested confidence
    std::size_t max_iterations_ = 0;

    // State bookkeeping.
    std::size_t iteration_ = 0;
    bool converged_ = false;
    event::tally<bitpack_t_> current_tally_{};
};

} // namespace scram::mc::queue