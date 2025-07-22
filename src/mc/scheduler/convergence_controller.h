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
#include "mc/scheduler/progressbar.h"
#include "mc/scheduler/iteration_shape.h"
#include "mc/stats/ci_utils.h"
#include "mc/stats/diagnostics.h"

#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>

#define PRECISION_LOG_SCIENTIFIC_DIGITS 3

namespace scram::mc::scheduler {

template <typename bitpack_t_, typename prob_t_ = std::double_t, typename size_t_ = std::uint64_t>
struct progress;

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

        steps_ = {
            .current = iteration_shape<bitpack_t_>(mgr.shaper().SAMPLE_SHAPE, 0),
            .target  = iteration_shape<bitpack_t_>(mgr.shaper().SAMPLE_SHAPE, settings.num_trials()),
        };

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

    void process_tally(const event::tally<bitpack_t_> &tally) {
        current_tally_ = tally;
        // ------------------ update iterations ---------------------------------------
        steps_.current.trials(tally.total_bits);
        // ------------------ update epsilons -----------------------------------------
        const std::double_t &target_z = target_zscore();
        // The half_width helper expects the *Z*-score. We pre-computed the
        // two-sided normal quantile when initialising `targets_`, so use it
        // here instead of the confidence level itself.
        interval_.current.half_width_epsilon = stats::half_width(tally, target_z);
        // set the target epsilon as a fraction of the estimated mean
        const std::double_t target_epsilon = settings_.ci_rel_margin_error() * std::max(tally.mean, stats::DELTA_EPSILON);
        interval_.target.half_width_epsilon = target_epsilon;
        // ------------------ update projected trials ----------------------------------
        const auto N = stats::required_trials_from_normal_quantile_two_sided(tally.mean, target_epsilon, target_z);
        steps_.target.trials(N);
    }

    [[nodiscard]] bool check_convergence() const {
        return check_epsilon_bounded(interval_);
    }

    /** Execute exactly one additional iteration on the device. */
    [[nodiscard]] bool step() {

        // don't step anymore, just return that we didn't take a step.
        if (converged_ && stop_on_convergence()) {
            return false;
        }

        // out of iterations, return that we didn't take a step.
        // evals to false if max_iterations_ is 0, which means keep going.
        if (iteration_limit_reached()) {
            progress_.mark_fixed_iterations_complete(*this);
            return false;
        }

        // still have iterations remaining
        // get the tally
        process_tally(manager_.single_pass_and_tally(evt_idx_));

        // for now, this is our convergence criteria
        // if converged now, set convergence_ sticky to true
        if (!converged_ && check_convergence()) {
            converged_ = true;
            progress_.mark_converged(*this);
        }

        // since we did step, update the iteration count
        return true;
    }

    /**
     * Execute exactly one additional asynchronous iteration on the device, don't worry about getting the tallies yet
     * since they are accumulating on device. The iteration count keeps track of how many trials have been run on device
     * so far.
     */
    [[nodiscard]] bool burn_in_step() {

        // iteration_ now shows that enough tasks have been queued on device such that by the time they finish,
        // pilot trials would be complete. So, don't queue up any additional work, just return saying you're done taking
        // pilot steps.
        if (burn_in_complete()) {
            progress_.mark_burn_in_complete(*this);
            return false;
        }

        process_tally(manager_.single_pass_and_tally(evt_idx_));

        // since we did step, update the iteration count
        return true;
    }

    /**
     * Run until the stopping criterion is met (or until the original plan is
     * exhausted).  Returns the final tally.
     */
    [[nodiscard]] event::tally<bitpack_t_> run_to_convergence() {
        // queue up pilot trials, but dont check for convergence.
        while(burn_in_step()) {
            progress_.tick_burn_in(*this);
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
    tracked_pair<iteration_shape<bitpack_t_>> steps_{};
    event::tally<bitpack_t_> current_tally_{};
    bool converged_ = false;
    progress<bitpack_t_, prob_t_, size_t_> progress_;

public:
    [[nodiscard]] bool diagnostics_enabled() const { return settings_.oracle_p() >= 0.0; }
    [[nodiscard]] std::double_t ground_truth() const { return settings_.oracle_p(); }

    [[nodiscard]] const iteration_shape<bitpack_t_> &current_steps() const { return steps_.current; }
    [[nodiscard]] const iteration_shape<bitpack_t_> &projected_steps() const { return steps_.target; }
    [[nodiscard]] const iteration_shape<bitpack_t_> &remaining_steps() const {
        return {
            .iterations = projected_steps().iterations() - current_steps().iterations(),
            .trials = projected_steps().trials() - current_steps().trials(),
        };
    }

    [[nodiscard]] bool stop_on_convergence() const { return settings_.early_stop(); }
    [[nodiscard]] bool converged() const { return converged_; }
    [[nodiscard]] tracked_triplet<iteration_shape<bitpack_t_>> convergence_status() const {
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

    [[nodiscard]] static bool check_epsilon_bounded(const tracked_pair<stats::ci> &interval) {
        const auto &current = interval.current.half_width_epsilon;
        const auto &target = interval.target.half_width_epsilon;
        return current > 0 && current <= target;
    }

    [[nodiscard]] bool iteration_limit_reached() const {
        const std::size_t max_iterations = manager_.shaper().TOTAL_ITERATIONS;
        return max_iterations && current_steps().iterations() >= max_iterations;
    }

    [[nodiscard]] bool burn_in_complete() const {
        return current_steps().trials() >= settings_.ci_burnin_trials();
    }

    [[nodiscard]] std::size_t burn_in_trials() const {
        return settings_.ci_burnin_trials();
    }

    [[nodiscard]] iteration_shape<bitpack_t_> burn_in_trials_shape() const {
        return iteration_shape<bitpack_t_>(manager_.shaper().SAMPLE_SHAPE, settings_.ci_burnin_trials());
    }

    [[nodiscard]] bool fixed_iterations() const {
        return manager_.shaper().TOTAL_ITERATIONS;
    }

    [[nodiscard]] iteration_shape<bitpack_t_> fixed_iterations_shape() const {
        auto shape = iteration_shape<bitpack_t_>(manager_.shaper().SAMPLE_SHAPE, 0);
        shape.iterations(manager_.shaper().TOTAL_ITERATIONS);
        return shape;
    }
};

template <typename bitpack_t_, typename prob_t_, typename size_t_>
struct progress {

    void initialize(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller, const bool watch_mode) {
        // Initialize timing for throughput tracking
        last_tick_time_ = std::chrono::high_resolution_clock::now();
        first_tick_ = true;

        watch_mode_ = watch_mode;

        // Configure progress bar
        if (!isatty(fileno(stdout)) || !isatty(fileno(stderr))) {
            LOG(WARNING) << "Disabling progressbar since neither STDOUT nor STDERR are TTYs.";
            watch_mode_ = false;
            return;
        }

        if (!watch_mode) {
            LOG(WARNING) << "Disabling progressbar since watch mode is disabled. Enable with --watch flag";
        }

        if (!watch_mode_) {
            return;
        }

        bars_ = std::make_unique<indicators::DynamicProgress<indicators::ProgressBar>>();
        bars_->set_option(indicators::option::HideBarWhenComplete{false});
        indicators::show_console_cursor(false);

        // all-inclusive progress bar, will be dynamically updated
        setup_fixed_bar(*controller);
        setup_burn_in_bar(*controller);
        setup_convergence_bar(*controller);
        setup_estimate(*controller);
        if (controller->diagnostics_enabled()) {
            setup_diagnostics(*controller);
            setup_accuracy_metrics(*controller);
        }
        setup_throughput(*controller);
    }

    void tick(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) {
        tick_fixed_bar(*controller);
        tick_convergence_bar(*controller);
        tick_text(controller);
        if (!watch_mode_) {
            LOG(DEBUG2) << controller->current_tally();
        }
    }

    ~progress() {
        indicators::show_console_cursor(true);
        for (auto &bar: owned_bars_) {
            bar->mark_as_completed();
        }
    }

    void tick_text(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) const {
        tick_estimate_bar(*controller);
        tick_diagnostics(*controller);
        tick_accuracy_metrics(*controller);
        tick_throughput_bar(*controller);
    }

    // Mark the convergence bar as complete once the controller reports success.
    void mark_converged(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        tick_convergence_bar(controller);
        if (convergence_ && !bars_->operator[](*convergence_).is_completed()) {
            bars_->operator[](*convergence_).mark_as_completed();
        }
    }

    // Mark the iterations bar as complete when the planned iterations finish.
    void mark_fixed_iterations_complete(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        tick_fixed_bar(controller);
        if (fixed_iterations_ && !bars_->operator[](*fixed_iterations_).is_completed()) {
            bars_->operator[](*fixed_iterations_).mark_as_completed();
        }
    }

    void mark_burn_in_complete(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        tick_burn_in(controller);
        if (burn_in_ && !bars_->operator[](*burn_in_).is_completed()) {
            bars_->operator[](*burn_in_).mark_as_completed();
        }
    }

    void tick_burn_in(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        tick_fixed_bar(controller);
        tick_text(&controller);
        if (burn_in_ && !bars_->operator[](*burn_in_).is_completed()) {
            auto &bar = bars_->operator[](*burn_in_);
            std::ostringstream tar_ss;
            tar_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.target_state().half_width_epsilon;
            const std::string t = tar_ss.str();
            std::ostringstream cur_ss;
            cur_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.current_state().half_width_epsilon;
            const std::string c = cur_ss.str();
            bar.set_option(indicators::option::PrefixText{"[burn-in]     :: ε= "+c+" | ε₀= "+t+" :: "});
            const auto cur_ite = controller.current_steps().iterations();
            const auto tot_ite = controller.burn_in_trials_shape().iterations();
            const std::string ite = std::to_string(cur_ite) + "/" + std::to_string(tot_ite);
            bar.set_option(indicators::option::PostfixText{"["+ite+"]"});
            bar.set_progress(cur_ite);
        }
    }

  private:
    // Container managed by the indicators library (prints the bars)
    std::unique_ptr<indicators::DynamicProgress<indicators::ProgressBar>> bars_;

    // We must keep the ProgressBar objects alive for as long as the DynamicProgress
    // references them.  Store owning pointers here.
    std::vector<std::unique_ptr<indicators::ProgressBar>> owned_bars_;

    // Indices into `bars_`; use std::optional so that index 0 is not interpreted
    // as *false*.
    std::optional<std::size_t> burn_in_{};
    std::optional<std::size_t> convergence_{};
    std::optional<std::size_t> fixed_iterations_{};
    std::optional<std::size_t> accuracy_metrics_{};
    std::optional<std::size_t> diagnostics_{};
    std::optional<std::size_t> estimate_{};
    std::optional<std::size_t> throughput_{};
    std::optional<std::size_t> all_inclusive_progress_{};

    static constexpr std::uint8_t bar_width_ = 30;
    // Throughput tracking state
    mutable std::chrono::high_resolution_clock::time_point last_tick_time_;
    mutable bool first_tick_ = true;
    mutable std::size_t last_iteration_ = 0;

    bool watch_mode_ = false;

    indicators::ProgressBar &add_text(std::optional<std::size_t> &idx, const std::optional<std::string> &pretext) {
        if (!idx) {
            auto bar_ptr = make_text(pretext);
            idx = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
        return bars_->operator[](*idx);
    }

    indicators::ProgressBar &add_iterations_bar(std::optional<std::size_t> &idx, const std::size_t count = 0) {
        if (!idx) {
            auto bar_ptr = make_iterations_progress_bar(count);
            idx = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
        return bars_->operator[](*idx);
    }

    void setup_burn_in_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        if (controller.burn_in_trials()) {
            auto &bar = add_iterations_bar(burn_in_);
            const auto tot_ite = controller.burn_in_trials_shape().iterations();
            // bar.set_option(indicators::option::PrefixText{"[burn-in]     :: ε= ?         | ε₀= ?         :: "});
            bar.set_option(indicators::option::BarWidth{bar_width_});
            bar.set_option(indicators::option::MaxProgress{tot_ite});
            bar.set_option(indicators::option::ShowPercentage{true});
            bar.set_option(indicators::option::ShowElapsedTime{true});
            bar.set_option(indicators::option::ShowRemainingTime{true});
            bar.set_option(indicators::option::ForegroundColor{indicators::Color::white});
            tick_burn_in(controller);
        }
    }

    void setup_fixed_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        if (controller.fixed_iterations()) {
            auto &bar = add_iterations_bar(fixed_iterations_);
            const auto tot_ite = controller.fixed_iterations_shape().iterations();
            bar.set_option(indicators::option::PrefixText{"[fixed]       :: "});
            bar.set_option(indicators::option::BarWidth{bar_width_});
            bar.set_option(indicators::option::MinProgress{0});
            bar.set_option(indicators::option::MaxProgress{tot_ite});
            bar.set_option(indicators::option::ShowPercentage{true});
            bar.set_option(indicators::option::ShowElapsedTime{true});
            bar.set_option(indicators::option::ShowRemainingTime{true});
            bar.set_option(indicators::option::ForegroundColor{indicators::Color::white});
            tick_fixed_bar(controller);
        }
    }

    void setup_convergence_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        auto &bar = add_iterations_bar(convergence_);
        std::ostringstream tar_ss;
        tar_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.target_state().half_width_epsilon;
        const std::string t = tar_ss.str();
        std::ostringstream cur_ss;
        cur_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.current_state().half_width_epsilon;
        const std::string c = cur_ss.str();
        // bar.set_option(indicators::option::PrefixText{"[convergence] :: ε= "+c+" | ε₀= "+t+" :: "});
        bar.set_option(indicators::option::BarWidth{bar_width_});
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::white});
    }

    void setup_throughput(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        auto &bar = add_text(throughput_, "[throughput]  ::");
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::magenta});
    }

    void setup_estimate(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        auto &bar = add_text(estimate_, "[estimate]    ::");
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::yellow});
    }

    void setup_accuracy_metrics(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        auto &bar = add_text(accuracy_metrics_, "[accuracy]    ::");
        std::ostringstream true_p_ss;
        true_p_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.ground_truth();
        const std::string str_prefix = "[accuracy]    :: true(p)= "+true_p_ss.str()+" |";
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::white});
    }

    void setup_diagnostics(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) {
        auto &bar = add_text(diagnostics_,"[diagnostics] ::");
        bar.set_option(indicators::option::ForegroundColor{indicators::Color::green});
    }

    void tick_convergence_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (convergence_ && !bars_->operator[](*convergence_).is_completed()) {
            auto &bar = bars_->operator[](*convergence_);
            const auto cur_ite = controller.current_steps().iterations();
            const auto projected_ite = controller.projected_steps().iterations();
            const std::string ite = std::to_string(cur_ite) + "/" + std::to_string(projected_ite);
            std::ostringstream tar_ss;
            tar_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.target_state().half_width_epsilon;
            const std::string t = tar_ss.str();
            std::ostringstream cur_ss;
            cur_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.current_state().half_width_epsilon;
            const std::string c = cur_ss.str();
            bar.set_option(indicators::option::PrefixText{"[convergence] :: ε= "+c+" | ε₀= "+t+" :: "});
            bar.set_option(indicators::option::PostfixText{"["+ite+"]"});
            bar.set_option(indicators::option::MinProgress{0});
            bar.set_option(indicators::option::MaxProgress{projected_ite}); // moving target
            bar.set_option(indicators::option::ShowPercentage{true});
            bar.set_option(indicators::option::ShowElapsedTime{true});
            bar.set_option(indicators::option::ShowRemainingTime{true});
            bar.set_progress(cur_ite);
        }
    }

    void tick_fixed_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (fixed_iterations_ && !bars_->operator[](*fixed_iterations_).is_completed()) {
            auto &bar = bars_->operator[](*fixed_iterations_);
            const auto cur_ite = controller.current_steps().iterations();
            const auto tot_ite = controller.fixed_iterations_shape().iterations();
            const std::string ite = std::to_string(cur_ite) + "/" + std::to_string(tot_ite);
            bar.set_option(indicators::option::PostfixText{"["+ite+"]"});
            bar.set_progress(cur_ite);
        }
    }

    void tick_estimate_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (!estimate_) {
            return;
        }
        auto &bar = bars_->operator[](*estimate_);
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << controller.current_tally();
        bar.set_option(indicators::option::PostfixText{ss.str()});
    }

    void tick_diagnostics(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (!diagnostics_) {
            return;
        }
        auto &bar = bars_->operator[](*diagnostics_);
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << *controller.sampling_diagnostics();
        bar.set_option(indicators::option::PostfixText{ss.str()});
    }

    void tick_accuracy_metrics(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (!accuracy_metrics_) {
            return;
        }
        auto &bar = bars_->operator[](*accuracy_metrics_);
        std::ostringstream metrics_ss;
        metrics_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << *controller.accuracy_metrics();
        bar.set_option(indicators::option::PostfixText{metrics_ss.str()});
    }


    void tick_throughput_bar(const convergence_controller<bitpack_t_, prob_t_, size_t_> &controller) const {
        if (!throughput_) {
            return;
        }

        auto &bar = bars_->operator[](*throughput_);

        const auto current_time = std::chrono::high_resolution_clock::now();
        const auto current_iteration = controller.current_steps().iterations();

        if (first_tick_) {
            first_tick_ = false;
            last_tick_time_ = current_time;
            last_iteration_ = current_iteration;
            bar.set_option(indicators::option::PostfixText{"initializing..."});
            return;
        }

        const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - last_tick_time_).count();
        const auto iterations_delta = current_iteration - last_iteration_;

        if (elapsed_time <= 0.0 || iterations_delta == 0) {
            return; // Avoid division by zero
        }

        // Calculate iterations per second or seconds per iteration
        const double iterations_per_sec = static_cast<double>(iterations_delta) / elapsed_time;
        std::string iter_rate_str;
        if (iterations_per_sec >= 1.0) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << iterations_per_sec << " iter/s";
            iter_rate_str = ss.str();
        } else {
            const double sec_per_iteration = elapsed_time / static_cast<double>(iterations_delta);
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << sec_per_iteration << " s/iter";
            iter_rate_str = ss.str();
        }

       //  Calculate bits per iteration with appropriate units
        const auto bits_per_iteration = controller.fixed_iterations_shape().trials_per_iteration();
         std::string bits_iter_str;
         if (bits_per_iteration >= 1024 * 1024) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << static_cast<double>(bits_per_iteration) / (1024.0 * 1024.0) << " Mbit/iter";
             bits_iter_str = ss.str();
         } else if (bits_per_iteration >= 1024) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << static_cast<double>(bits_per_iteration) / 1024.0 << " kbit/iter";
             bits_iter_str = ss.str();
         } else {
             std::ostringstream ss;
             ss << bits_per_iteration << " bit/iter";
             bits_iter_str = ss.str();
         }

         // Calculate bits per second with appropriate units
         const double bits_per_sec = static_cast<double>(bits_per_iteration * iterations_delta) / elapsed_time;
         std::string bits_sec_str;
         if (bits_per_sec >= 1024.0 * 1024.0) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << bits_per_sec / (1024.0 * 1024.0) << " Mbit/s";
             bits_sec_str = ss.str();
         } else if (bits_per_sec >= 1024.0) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << bits_per_sec / 1024.0 << " kbit/s";
             bits_sec_str = ss.str();
         } else {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << bits_per_sec << " bit/s";
             bits_sec_str = ss.str();
         }

         // Combine all metrics into a single line
         const std::string throughput_text = bits_iter_str + " | " + iter_rate_str + " | " + bits_sec_str;
         bar.set_option(indicators::option::PostfixText{throughput_text});

         // Update state for next tick
         last_tick_time_ = current_time;
         last_iteration_ = current_iteration;
    }


    static std::unique_ptr<indicators::ProgressBar> make_iterations_progress_bar(const std::size_t max_iterations = 0) {
        //const std::string str_ite = "0/" + max_iterations;
        //const std::string str_prefix = "["+str_ite+"] :: ";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{bar_width_},
            indicators::option::Start{"["},
            indicators::option::Fill{"■"},
            indicators::option::Lead{"■"},
            indicators::option::Remainder{"-"},
            indicators::option::End{"]"});
    }

    static std::unique_ptr<indicators::ProgressBar> make_text(const std::optional<std::string> &pretext) {
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::PrefixText{pretext ? *pretext : ""},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::ForegroundColor{indicators::Color::white},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false});
    }

};
} // namespace scram::mc::queue