#pragma once


#include <indicators/cursor_control.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <iomanip>
#include <sstream>
#include <vector>
#include <optional>

#define PRECISION_LOG_SCIENTIFIC_DIGITS 3

namespace scram::mc::scheduler {

template <typename bitpack_t_, typename prob_t_, typename size_t_>
class convergence_controller;

template <typename bitpack_t_, typename prob_t_, typename size_t_>
struct progress {

    void initialize(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) {
        // Configure progress bar
        if (!isatty(fileno(stdout)) || !isatty(fileno(stderr))) {
            LOG(WARNING) << "Disabling progressbar since STDOUT is not a TTY.";
            return;
        }
        bars_ = std::make_unique<indicators::DynamicProgress<indicators::ProgressBar>>();
        bars_->set_option(indicators::option::HideBarWhenComplete{false});
        indicators::show_console_cursor(false);

        add_iterations_bar(controller->max_iterations());
        add_convergence_bar(controller->targets());
    }

    void tick(const convergence_controller<bitpack_t_, prob_t_, size_t_> *controller) {
        if (convergence_) {
            tick_convergence_progress_bar(bars_->operator[](*convergence_), controller->current_tally(),
                                         controller->targets(), controller->current());
        }
        if (iterations_) {
            tick_iterations_progress_bar(bars_->operator[](*iterations_), controller->iterations(),
                                        controller->max_iterations());
        }
    }

    ~progress() {
        indicators::show_console_cursor(true);
    }

    void mark_converged() const {
        if (convergence_) {
            //bars_.get()[convergence_].print_progress();
        }
    }

    void mark_iterations_complete() const {
        if (iterations_) {
            //iterations_->mark_as_completed();
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

    void add_convergence_bar(const stats::ci &targets) {
        if (!convergence_) {
            auto bar_ptr = make_convergence_progress_bar(targets);
            convergence_ = bars_->push_back(*bar_ptr);
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
        ci_ss << std::scientific << std::setprecision(1) << targets.two_sided_confidence_level;
        const std::string epsilon = "ε: " + eps_ss.str();
        const std::string conf_lv = "CI(" + ci_ss.str() + ")";
        const std::string str_target_epsilon = "[convergence] target " + epsilon + ", " + conf_lv;
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{100},
            indicators::option::PrefixText{str_target_epsilon},
            indicators::option::PostfixText{"ε: +inf, |Δε|: +inf"},
            indicators::option::ForegroundColor{indicators::Color::yellow},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}
            // indicators::option::FontStyles{std::vector{indicators::FontStyle::bold}}
        );
    }

    static void tick_convergence_progress_bar(indicators::ProgressBar &bar,
                                              const event::tally<bitpack_t_> &tally, const stats::ci &target,
                                              const stats::ci &current) {
        std::ostringstream cur_eps_ss;
        cur_eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << current.half_width_epsilon;
        const std::string cur_eps = "ε: " + cur_eps_ss.str();

        const std::double_t delta_epsilon = std::abs(target.half_width_epsilon - current.half_width_epsilon);
        std::ostringstream del_eps_ss;
        del_eps_ss << std::scientific << std::setprecision(PRECISION_LOG_SCIENTIFIC_DIGITS) << delta_epsilon;
        const std::string del_eps = "|Δε|: " + del_eps_ss.str();
        const std::string postfix = cur_eps + ", " + del_eps;
        // current trials
        const std::size_t current_trials = tally.total_bits;
        // total trials
        const std::size_t total_trials_needed_for_convergence = stats::required_trials(tally, target);
        // set the values
        bar.set_option(indicators::option::PostfixText{postfix});
        bar.set_option(indicators::option::MaxProgress{total_trials_needed_for_convergence});
        bar.set_progress(current_trials);
    }

    static std::unique_ptr<indicators::ProgressBar> make_iterations_progress_bar(const std::size_t max_iterations) {
        const std::string str_postfix = "iteration 0/" + max_iterations;
        const std::string str_prefix = "[iterations] ";
        return std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"■"},
            indicators::option::Lead{"■"},
            indicators::option::Remainder{"-"},
            indicators::option::End{" ]"},
            indicators::option::PrefixText{str_prefix},
            indicators::option::PostfixText{str_postfix},
            indicators::option::ForegroundColor{indicators::Color::white},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::FontStyles{std::vector{indicators::FontStyle::bold}});
    }

    static void tick_iterations_progress_bar(indicators::ProgressBar &bar,
                                             const std::size_t current_iterations, const std::size_t max_iterations) {
        const std::string str_postfix =
            "iteration " + std::to_string(current_iterations) + "/" + std::to_string(max_iterations);
        // set the values
        bar.set_option(indicators::option::PostfixText{str_postfix});
        bar.set_option(indicators::option::MaxProgress{max_iterations});
        bar.set_progress(current_iterations);
    }
};
} // namespace scram::mc::scheduler