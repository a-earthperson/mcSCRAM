#pragma once


#include <indicators/cursor_control.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <iomanip>
#include <sstream>
#include <vector>
#include <optional>
#include <chrono>
#include "mc/stats/ci_utils.h"
#include "mc/stats/diagnostics.h"

#define PRECISION_LOG_SCIENTIFIC_DIGITS 3

namespace scram::mc::scheduler {

template <typename bitpack_t_, typename prob_t_, typename size_t_>
class convergence_controller;

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
        const std::double_t bits_per_iteration_whole_graph = static_cast<double_t>(controller.fixed_iterations_shape().trials_per_iteration());
        const std::double_t nodes_per_graph = static_cast<double_t>(controller.node_count());
        const std::double_t bits_per_iteration = bits_per_iteration_whole_graph;
         std::string bits_iter_str;
         if (bits_per_iteration >= 1024 * 1024) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << bits_per_iteration / (1024.0 * 1024.0) << " Mbit/iter";
             bits_iter_str = ss.str();
         } else if (bits_per_iteration >= 1024) {
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(2) << bits_per_iteration / 1024.0 << " kbit/iter";
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
} // namespace scram::mc::scheduler