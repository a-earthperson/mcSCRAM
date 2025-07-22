#pragma once
#include <cstddef>
#include "mc/event/node.h"

namespace scram::mc::scheduler {

template<typename bitpack_t_>
class iteration_shape {
public:
    explicit iteration_shape(const event::sample_shape<std::size_t> &shape = {}, const std::size_t trials = 0)   // canonical state = trials
        : shape_{shape}, trials_{trials} {}

    // ---- getters -------------------------------------------------------
    [[nodiscard]] std::size_t trials()     const noexcept { return trials_; }
    [[nodiscard]] std::size_t iterations() const noexcept {
        return (trials_ + trials_per_iteration() - 1) / trials_per_iteration();
    }

    // ---- setters -------------------------------------------------------
    void trials(const std::size_t t) noexcept { trials_ = t; }
    void iterations(const std::size_t it) noexcept {
        trials_ = it * trials_per_iteration();
    }

    // convenience operators
    iteration_shape& operator++()            { iterations(iterations() + 1); return *this; }
    iteration_shape& operator--()            { iterations(iterations() - 1); return *this; }
    iteration_shape& operator+=(const std::size_t i){ iterations(iterations() + i); return *this; }

    [[nodiscard]] std::size_t trials_per_iteration() const noexcept {
        return shape_.num_bitpacks() * sizeof(bitpack_t_) * 8;
    }
private:


    event::sample_shape<std::size_t> shape_{};
    std::size_t                      trials_{};      // single source of truth
};

// template<typename bitpack_t_>
// struct iteration_shape {
// public:
//     std::size_t iterations{};
//     std::size_t trials;
//     template<event::sample_shape<std::size_t> shape>
//     [[nodiscard]] iteration_shape(const std::size_t trials)
//         : trials(trials), shape_(shape) {}
//
//     [[nodiscard]] static std::size_t cumulative_bits(const event::sample_shape<std::size_t> &shape,
//                                                      const size_t &iteration = 1) {
//         return iteration * shape.num_bitpacks() * sizeof(bitpack_t_) * 8;
//     }
//     void set_iterations(const std::size_t iterations) {
//         this->trials = cumulative_bits(shape_, iterations);
//     }
//     void set_trials(const std::size_t trials) { this->trials = trials; }
//
//     [[nodiscard]] static std::size_t trials_per_iteration(const event::sample_shape<std::size_t> &shape) {
//         return cumulative_bits(shape, 1);
//     }
//
//     [[nodiscard]] static std::size_t iterations_from_trials(const std::size_t trials, const std::size_t trials_per_iteration) {
//         return static_cast<std::size_t>(std::ceil(static_cast<std::double_t>(trials) / static_cast<std::double_t>(trials_per_iteration)));
//     }
// private:
//     event::sample_shape<std::size_t> shape_;
// };

template <typename DataT>
struct tracked_pair {
    DataT current{};
    DataT target{};
};

template <typename DataT>
struct tracked_triplet {
    DataT current{};
    DataT target{};
    DataT remaining{};
};
}