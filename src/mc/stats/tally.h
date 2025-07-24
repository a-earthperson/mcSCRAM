#pragma once

namespace scram::mc::stats {
// to be used by probability_analysis
struct tally {
    /// @brief Count of positive outcomes (1-bits) across all samples
    std::size_t num_one_bits = 0;

    /// @brief Count of total outcomes evaluated so far
    std::size_t total_bits = 0;

    /// @brief Estimated mean probability based on sample proportion
    std::double_t mean = 0.;

    /// @brief Standard error of the probability estimate
    std::double_t std_err = 0.;

    std::array<double_t, 4> ci = {0., 0., 0., 0.};
};
} // namespace scram::mc::stats