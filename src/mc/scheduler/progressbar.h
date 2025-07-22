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

        add_iterations_bar(controller->max_iterations());
        add_throughput_bar();
        add_convergence_bar(controller->targets());
        add_estimate_bar();
        if (controller->diagnostics_enabled()) {
            add_accuracy_metrics_bar(controller->ground_truth());
            add_diagnostics_bar();
        }
    }

    void tick(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) {
        const auto tally = controller->current_tally();
        const auto stats_tar = controller->targets();
        const auto stats_cur = controller->current();
        const auto trials_per_ite = controller->trials_per_iteration();
        const auto cur_ite = controller->iterations();
        const auto max_ite = controller->max_iterations();
        const std::size_t current_trials = tally.total_bits;
        const std::size_t tot_conv_trials = stats::required_trials(tally, stats_tar);
        const std::size_t cur_conv_ite = std::ceil(static_cast<std::double_t>(current_trials) / static_cast<std::double_t>(trials_per_ite));
        const std::size_t tot_conv_ite = std::ceil(static_cast<std::double_t>(tot_conv_trials) / static_cast<std::double_t>(trials_per_ite));
        if (!watch_mode_) {
            LOG(DEBUG2) << tally;
        }
        if (iterations_) {
            if (!max_ite) {
                tick_iterations_progress_bar(bars_->operator[](*iterations_), cur_conv_ite, tot_conv_ite);
            } else {
                tick_iterations_progress_bar(bars_->operator[](*iterations_), cur_ite, max_ite);
            }
        }
        if (convergence_) {
            tick_convergence_progress_bar(bars_->operator[](*convergence_), tally,stats_tar, stats_cur);
        }
        if (estimate_) {
            tick_estimate_bar(bars_->operator[](*estimate_), tally);
        }
        if (throughput_) {
            tick_throughput_bar(bars_->operator[](*throughput_), controller);
        }
        if (accuracy_metrics_) {
            tick_accuracy_metrics_bar(bars_->operator[](*accuracy_metrics_), controller->accuracy_metrics());
        }
        if (diagnostics_) {
            tick_diagnostics_bar(bars_->operator[](*diagnostics_), controller->sampling_diagnostics());
        }
    }

    ~progress() {
        indicators::show_console_cursor(true);
        for (auto &bar: owned_bars_) {
            bar->mark_as_completed();
        }
    }

    // Mark the convergence bar as complete once the controller reports success.
    void mark_converged() const {
        if (convergence_) {
            // `mark_as_completed` forces the bar to 100 % and prints the final line.
            bars_->operator[](*convergence_).mark_as_completed();
        }
    }

    // Mark the iterations bar as complete when the planned iterations finish.
    void mark_iterations_complete() const {
        if (iterations_) {
            bars_->operator[](*iterations_).mark_as_completed();
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
    std::optional<std::size_t> convergence_{};
    std::optional<std::size_t> iterations_{};
    std::optional<std::size_t> accuracy_metrics_{};
    std::optional<std::size_t> diagnostics_{};
    std::optional<std::size_t> estimate_{};
    std::optional<std::size_t> throughput_{};

    // Throughput tracking state
    mutable std::chrono::high_resolution_clock::time_point last_tick_time_;
    mutable bool first_tick_ = true;
    mutable std::size_t last_iteration_ = 0;

    bool watch_mode_ = false;

    void add_convergence_bar(const stats::ci &targets) {
        if (!convergence_) {
            auto bar_ptr = make_convergence_progress_bar(targets);
            convergence_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    void add_accuracy_metrics_bar(const std::double_t ground_truth) {
        if (!accuracy_metrics_) {
            auto bar_ptr = make_accuracy_metrics_bar(ground_truth);
            accuracy_metrics_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    void add_diagnostics_bar() {
        if (!diagnostics_) {
            auto bar_ptr = make_diagnostics_bar();
            diagnostics_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    void add_estimate_bar() {
        if (!estimate_) {
            auto bar_ptr = make_estimate_bar();
            estimate_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    void add_throughput_bar() {
        if (!throughput_) {
            auto bar_ptr = make_throughput_bar();
            throughput_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    void add_iterations_bar(const std::size_t max_iterations) {
        if (!iterations_) {
            auto bar_ptr = make_iterations_progress_bar(max_iterations);
            iterations_ = bars_->push_back(*bar_ptr);
            owned_bars_.push_back(std::move(bar_ptr));
        }
    }

    // marks the progress/march towards convergence, stopping at the iteration # when convergence was achieved.
    static std::unique_ptr<indicators::ProgressBar> make_convergence_progress_bar(const stats::ci &targets) {
        std::ostringstream eps_ss;
        eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << targets.half_width_epsilon;
        std::ostringstream ci_ss;
        ci_ss << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS + 3) << targets.two_sided_confidence_level;
        const std::string epsilon = eps_ss.str();
        const std::string conf_lv = "ci[" + ci_ss.str() + "]";
        const std::string str_prefix_pre = "[convergence] :: "+conf_lv+" tar(ε)= " + eps_ss.str() + " | ";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::PrefixText{str_prefix_pre},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::ForegroundColor{indicators::Color::yellow},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false},
            indicators::option::FontStyles{}
        );
    }

    static void tick_convergence_progress_bar(indicators::ProgressBar &bar,
                                              const event::tally<bitpack_t_> &tally, const stats::ci &target,
                                              const stats::ci &current) {
        std::ostringstream eps_ss;
        eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << target.half_width_epsilon;
        std::ostringstream ci_ss;
        ci_ss << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS + 3) << target.two_sided_confidence_level;
        const std::string epsilon = eps_ss.str();
        const std::string conf_lv = "ci[" + ci_ss.str() + "]";
        const std::string str_prefix_pre = "[convergence] :: "+conf_lv+" tar(ε)= " + eps_ss.str() + " | ";
        std::ostringstream cur_eps_ss;
        cur_eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << current.half_width_epsilon;
        const std::string cur_eps = "ε= " + cur_eps_ss.str();
        const std::double_t delta_epsilon = std::abs(target.half_width_epsilon - current.half_width_epsilon);
        std::ostringstream del_eps_ss;
        del_eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << delta_epsilon;
        const std::string del_eps = "| |Δε|: " + del_eps_ss.str();
        const std::string str_prefix_post = cur_eps + " " + del_eps;
        const std::string str_prefix = str_prefix_pre + str_prefix_post;
        // set the values
        bar.set_option(indicators::option::PrefixText{str_prefix});
    }

    static std::unique_ptr<indicators::ProgressBar> make_iterations_progress_bar(const std::size_t max_iterations) {
        const std::string str_ite = "0/" + max_iterations;
        const std::string str_prefix = "["+str_ite+"] :: ";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{str_prefix},
            indicators::option::Start{"["},
            indicators::option::Fill{"■"},
            indicators::option::Lead{"■"},
            indicators::option::Remainder{"-"},
            indicators::option::End{"]"},
            indicators::option::ForegroundColor{indicators::Color::magenta},
            indicators::option::FontStyles{std::vector{indicators::FontStyle::bold}});
    }

    static void tick_iterations_progress_bar(indicators::ProgressBar &bar,
                                             const std::size_t current_iterations,
                                             const std::size_t max_iterations) {
        const std::string str_ite = std::to_string(current_iterations)+"/" + std::to_string(max_iterations);
        const std::string str_prefix = "["+str_ite+"] :: ";
        // set the values
        bar.set_option(indicators::option::PrefixText{str_prefix});
        bar.set_option(indicators::option::MaxProgress{max_iterations});
        bar.set_progress(current_iterations);
    }

    static void toggle_timing_progress_bar(indicators::ProgressBar &bar, const bool toggle) {
        // set the values
        bar.set_option(indicators::option::ShowPercentage{toggle});
        bar.set_option(indicators::option::ShowElapsedTime{toggle});
        bar.set_option(indicators::option::ShowRemainingTime{toggle});
    }



    static std::unique_ptr<indicators::ProgressBar> make_accuracy_metrics_bar(const std::double_t ground_truth) {
        std::ostringstream true_p_ss;
        true_p_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << ground_truth;
        const std::string str_prefix = "[accuracy]    :: true(p)= "+true_p_ss.str()+" |";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::PrefixText{str_prefix},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false});

    }

    static void tick_accuracy_metrics_bar(indicators::ProgressBar &bar, const std::optional<stats::AccuracyMetrics> &metrics) {
        if (!metrics) {
            return;
        }
        std::ostringstream metrics_ss;
        metrics_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << *metrics;
        bar.set_option(indicators::option::PostfixText{metrics_ss.str()});
    }

    static std::unique_ptr<indicators::ProgressBar> make_diagnostics_bar() {
        const std::string str_prefix = "[sampler]     ::";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::PrefixText{str_prefix},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false});
    }

    static void tick_diagnostics_bar(indicators::ProgressBar &bar, const std::optional<stats::SamplingDiagnostics> &diagnostics) {
        if (!diagnostics) {
            return;
        }
        std::ostringstream diagnostics_ss;
        diagnostics_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << *diagnostics;
        bar.set_option(indicators::option::PostfixText{diagnostics_ss.str()});
    }


    static std::unique_ptr<indicators::ProgressBar> make_estimate_bar() {
        const std::string str_prefix = "[estimate]    :: [p01 p05 mean p95 p99] :: ";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::PrefixText{str_prefix},
            indicators::option::ForegroundColor{indicators::Color::white},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false});

    }

    static void tick_estimate_bar(indicators::ProgressBar &bar, const event::tally<bitpack_t_> &tally) {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << tally;
        bar.set_option(indicators::option::PostfixText{ss.str()});
    }

    static std::unique_ptr<indicators::ProgressBar> make_throughput_bar() {
        const std::string str_prefix = "[throughput]  ::";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{0},
            indicators::option::Start{""},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::PrefixText{str_prefix},
            indicators::option::ForegroundColor{indicators::Color::white},
            indicators::option::ShowPercentage{false},
            indicators::option::ShowElapsedTime{false},
            indicators::option::ShowRemainingTime{false});
    }

    void tick_throughput_bar(indicators::ProgressBar &bar, const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) {
        const auto current_time = std::chrono::high_resolution_clock::now();
        const auto current_iteration = controller->iterations();
        
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

        // Calculate bits per iteration with appropriate units
        const auto trials_per_iteration = controller->trials_per_iteration();
        const auto bits_per_iteration = trials_per_iteration; // Each trial is a bit
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

};
} // namespace scram::mc::scheduler