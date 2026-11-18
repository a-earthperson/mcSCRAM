/**
 * @file sample_shaper.h
 * @brief Chooses the per-pass sampling geometry for the Monte-Carlo engine.
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details
 * The sample shaper decides how many Bernoulli trials each node samples per
 * device pass (an "iteration").  The result is a 1-D @ref event::sample_shape
 * with @c batch_size == 1 and @c bitpacks_per_batch == B, where each work-item
 * processes one bit-pack (e.g. 64 trials for @c uint64_t).
 *
 * The single free variable is
 *
 *     B  :=  bitpacks_per_batch   (bit-packs sampled, per node, per pass)
 *
 * which is chosen by @ref sample_shaper::solve_bitpacks_per_pass subject to a
 * small set of inequalities (device memory, total-work, optimistic throughput,
 * cache-line alignment, and power-of-2 sizing).  See that method for the full
 * derivation.
 *
 * Historically a fuzzy @c overhead_ratio knob inflated the per-node memory
 * footprint as a proxy for "use less device memory".  It had no precise meaning
 * and was abused to indirectly tune throughput.  It has been removed in favour
 * of an explicit, semantically meaningful @c requested_trials_per_pass target
 * (optimistic: the realized value may be smaller after constraints are applied).
 *
 * @copyright Copyright (C) 2025 Arjun Earthperson
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "mc/event/node.h"
#include "working_set.h"

#include "logger.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

// Forward declare the sample_shaper template so we can reference it before its full definition.
namespace scram::mc::queue {
    template<typename bitpack_t_> struct sample_shaper;
}

namespace scram::log::sample_shaper {
    // Forward declaration so that friend declaration inside mc::queue::sample_shaper
    // can find the template before it is defined in log_sample_shaper.h.
    template<typename bitpack_t_>
    std::vector<std::pair<std::string, std::string>>
    csv_pairs(const scram::mc::queue::sample_shaper<bitpack_t_>&);
}

/// Hard ceiling on a single device allocation we are willing to request, in bytes.
/// The realized budget is min(this, device max_mem_alloc_size).
static constexpr std::size_t MAX_ALLOCATABLE_DEVICE_BYTES = 256ULL * 1024ULL * 1024ULL;

namespace scram::mc::queue {
template<typename bitpack_t_>
struct sample_shaper {

    /// Number of device passes required to reach the requested total (0 → unbounded).
    std::size_t TOTAL_ITERATIONS = 0;
    /// The chosen per-pass geometry (batch_size == 1, bitpacks_per_batch == B).
    event::sample_shape<std::size_t> SAMPLE_SHAPE{};

    /// Default per-node trial budget assumed when the run is unbounded
    /// (requested_num_trials == 0) purely so the per-pass geometry can be sized.
    static constexpr std::size_t DEFAULT_UNBOUNDED_TRIALS = 64ull * 1024 * 1024;

    struct description {
        std::size_t requested_num_trials = 0;
        std::size_t rounded_num_trials = 0;
        std::size_t total_bits_to_sample = 0;
        std::size_t target_bits_per_iteration = 0;   ///< per-node memory budget, in bits
        std::size_t bits_per_iteration = 0;          ///< realized trials per node per pass
        std::size_t total_iterations = 0;
        std::size_t bits_per_bitpack = 0;
        std::size_t requested_trials_per_pass = 0;   ///< optimistic target (0 = auto)
        std::size_t realized_trials_per_pass = 0;    ///< == bits_per_iteration
        std::size_t memory_budget_bitpacks_per_node = 0;
        std::size_t num_nodes = 0;
        std::size_t max_device_bytes = 0;
        std::size_t max_device_bits = 0;
        std::size_t cache_line_bitpacks = 0;
        bool unlimited_iterations = false;
        std::string shape_strategy = "uninitialized";
        event::sample_shape<std::size_t> sample_shape{};
    };

    explicit sample_shaper() = default;

    /**
     * @param queue                     SYCL queue whose device drives the limits.
     * @param requested_num_trials      Total trials to sample per node (0 → unbounded).
     * @param num_nodes                 Number of nodes sharing the device memory.
     * @param requested_trials_per_pass Optimistic per-node trials-per-pass target
     *                                  (0 → automatic: fill the memory budget).
     */
    sample_shaper(const sycl::queue &queue,
                  const std::size_t requested_num_trials,
                  const std::size_t num_nodes,
                  const std::size_t requested_trials_per_pass)
        : requested_num_trials_(requested_num_trials),
          num_nodes_(num_nodes),
          requested_trials_per_pass_(requested_trials_per_pass) {

        const sycl::device device = queue.get_device();

        // ---- 1) device budget -------------------------------------------------
        const device_budget budget = query_device_budget(device);
        max_device_bytes_    = budget.max_alloc_bytes;
        max_device_bits_     = max_device_bytes_ * static_cast<std::size_t>(8);
        cache_line_bitpacks_ = budget.cache_line_bitpacks;

        // ---- 2) total work, rounded up to whole bit-packs ---------------------
        //  An unbounded run (requested == 0) is still given a finite figure so the
        //  per-pass geometry can be sized; TOTAL_ITERATIONS stays 0 (run forever).
        unlimited_iterations_ = (requested_num_trials_ == 0);
        const std::size_t effective_request =
            unlimited_iterations_ ? DEFAULT_UNBOUNDED_TRIALS : requested_num_trials_;
        num_trials_           = round_up_to_multiple(effective_request, bits_in_bitpack_);
        total_bits_to_sample_ = num_trials_;

        // per-node memory budget (informational + the (C1) bound below)
        memory_budget_bitpacks_per_node_ = memory_budget_bitpacks(max_device_bytes_, num_nodes_);
        target_bits_per_iteration_       = memory_budget_bitpacks_per_node_ * bits_in_bitpack_;

        // ---- 3) solve for B = bitpacks-per-pass under all constraints ----------
        const std::size_t bitpacks_per_pass = solve_bitpacks_per_pass(
            requested_trials_per_pass_,
            num_nodes_,
            total_bits_to_sample_,
            unlimited_iterations_,
            max_device_bytes_,
            cache_line_bitpacks_);

        SAMPLE_SHAPE    = event::sample_shape<std::size_t>{/*batch_size=*/1, bitpacks_per_pass};
        shape_strategy_ = build_strategy_label(requested_trials_per_pass_, cache_line_bitpacks_);

        // ---- 4) iteration accounting -----------------------------------------
        bits_per_iteration_ = SAMPLE_SHAPE.num_bitpacks() * bits_in_bitpack_;
        num_iterations_     = ceil_div(total_bits_to_sample_, bits_per_iteration_);
        TOTAL_ITERATIONS    = unlimited_iterations_ ? 0 : num_iterations_;

        LOG(DEBUG2) << working_set<std::size_t, bitpack_t_>(queue, num_nodes_, SAMPLE_SHAPE);
    }

    [[nodiscard]] description describe() const {
        return {
            .requested_num_trials = requested_num_trials_,
            .rounded_num_trials = num_trials_,
            .total_bits_to_sample = total_bits_to_sample_,
            .target_bits_per_iteration = target_bits_per_iteration_,
            .bits_per_iteration = bits_per_iteration_,
            .total_iterations = TOTAL_ITERATIONS,
            .bits_per_bitpack = bits_in_bitpack_,
            .requested_trials_per_pass = requested_trials_per_pass_,
            .realized_trials_per_pass = bits_per_iteration_,
            .memory_budget_bitpacks_per_node = memory_budget_bitpacks_per_node_,
            .num_nodes = num_nodes_,
            .max_device_bytes = max_device_bytes_,
            .max_device_bits = max_device_bits_,
            .cache_line_bitpacks = cache_line_bitpacks_,
            .unlimited_iterations = unlimited_iterations_,
            .shape_strategy = shape_strategy_,
            .sample_shape = SAMPLE_SHAPE,
        };
    }

    /**
     * @brief Formatted output operator for the chosen sampling geometry.
     */
    friend std::ostream &operator<<(std::ostream &os, const sample_shaper &sched) {
        os << "requested_num_trials: " << sched.requested_num_trials_ << std::endl
           << "requested_trials_per_pass: " << sched.requested_trials_per_pass_ << std::endl
           << "num_nodes: " << sched.num_nodes_ << std::endl
           << "------------------------------------------------" << std::endl
           << "max_device_bytes: " << sched.max_device_bytes_ << std::endl
           << "max_device_bits: " << sched.max_device_bits_ << std::endl
           << "bits_in_bitpack: " << sched.bits_in_bitpack_ << std::endl
           << "cache_line_bitpacks: " << sched.cache_line_bitpacks_ << std::endl
           << "memory_budget_bitpacks_per_node: " << sched.memory_budget_bitpacks_per_node_ << std::endl
           << "------------------------------------------------" << std::endl
           << "num_trials (rounded): " << sched.num_trials_ << std::endl
           << "total_bits_to_sample: " << sched.total_bits_to_sample_ << std::endl
           << "target_bits_per_iteration: " << sched.target_bits_per_iteration_ << std::endl
           << "bits_per_iteration (realized trials/pass): " << sched.bits_per_iteration_ << std::endl
           << "num_iterations: " << sched.num_iterations_ << std::endl
           << "shape_strategy: " << sched.shape_strategy_ << std::endl
           << "------------------------------------------------" << std::endl
           << "SAMPLE_SHAPE.batch_size: " << sched.SAMPLE_SHAPE.batch_size << std::endl
           << "SAMPLE_SHAPE.bitpacks_per_batch: " << sched.SAMPLE_SHAPE.bitpacks_per_batch << std::endl
           << "SAMPLE_SHAPE.num_bitpacks(): " << sched.SAMPLE_SHAPE.num_bitpacks() << std::endl
           << "TOTAL_ITERATIONS: " << sched.TOTAL_ITERATIONS << std::endl;
        return os;
    }

    // Friend declaration for logging function
    template<typename U>
    friend std::vector<std::pair<std::string, std::string>> scram::log::sample_shaper::csv_pairs(const sample_shaper<U>&);

    // =======================================================================
    //  Constraint model
    // =======================================================================

    /// Device-derived bounds relevant to per-pass sizing.
    struct device_budget {
        std::size_t max_alloc_bytes = 0;     ///< min(MAX_ALLOCATABLE_DEVICE_BYTES, device limit)
        std::size_t cache_line_bitpacks = 0; ///< cache line size expressed in bit-packs (0 ⇒ unknown)
    };

    [[nodiscard]] static device_budget query_device_budget(const sycl::device &device) {
        device_budget budget;
        budget.max_alloc_bytes = std::min<std::size_t>(
            MAX_ALLOCATABLE_DEVICE_BYTES,
            static_cast<std::size_t>(device.get_info<sycl::info::device::max_mem_alloc_size>()));
        const auto cache_line_bytes =
            static_cast<std::size_t>(device.get_info<sycl::info::device::global_mem_cache_line_size>());
        budget.cache_line_bitpacks = cache_line_bytes / sizeof(bitpack_t_); // 0 if device reports 0
        return budget;
    }

    /// (C1) per-node memory budget, in bit-packs:
    ///      B_mem = floor(max_alloc_bytes / (num_nodes · sizeof(bitpack))), floored at 1.
    [[nodiscard]] static std::size_t memory_budget_bitpacks(const std::size_t max_alloc_bytes,
                                                            const std::size_t num_nodes) {
        const std::size_t nodes = num_nodes ? num_nodes : 1;
        const std::size_t budget = max_alloc_bytes / (nodes * sizeof(bitpack_t_));
        return std::max<std::size_t>(budget, 1);
    }

    /**
     * @brief Solve for @c B (bit-packs per node per pass) under the constraints.
     *
     * Let @c P = bits_in_bitpack_ (bits per work-item).  We choose a single
     * integer @c B ≥ 1 that is as close as possible to an optimistic target
     * while respecting hard device/limit bounds:
     *
     *   (C1) Memory budget (hard upper bound):
     *          num_nodes · B · sizeof(bitpack) ≤ max_alloc_bytes
     *        ⇒ B ≤ B_mem = floor(max_alloc_bytes / (num_nodes · sizeof(bitpack)))
     *
     *   (C2) Total-work cap (only when the run is bounded): a single pass should
     *        never sample more than the whole requested budget.
     *          B ≤ B_total = ceil(total_bits_to_sample / P)
     *        (inactive — treated as +∞ — when @p unlimited).
     *
     *   (C3) Optimistic throughput target (the objective):
     *          B_req = (requested_trials_per_pass > 0)
     *                    ? ceil(requested_trials_per_pass / P)   // honour the request
     *                    : B_mem                                 // 0 ⇒ "auto": fill memory
     *
     *   (C4) Cache-line alignment (memory coalescing), applied only when it does
     *        not force over-sampling (i.e. when B ≥ one cache line):
     *          B ← B − (B mod C),   C = cache-line size in bit-packs
     *
     *   (C5) Positivity:  B ≥ 1.
     *
     *   (C6) Power-of-2 work sizing (GPU-friendly local ranges):
     *          B ← largest power of 2 ≤ B
     *
     * Objective:  B = floor_pow2( clamp_align( min(B_req, B_mem, B_total) ) ).
     * Because of (C4) and (C6) the realized trials-per-pass (B · P) may be
     * < requested_trials_per_pass — hence the request is *optimistic*.
     */
    [[nodiscard]] static std::size_t solve_bitpacks_per_pass(const std::size_t requested_trials_per_pass,
                                                            const std::size_t num_nodes,
                                                            const std::size_t total_bits_to_sample,
                                                            const bool unlimited,
                                                            const std::size_t max_alloc_bytes,
                                                            const std::size_t cache_line_bitpacks) {
        // (C1)
        const std::size_t b_mem = memory_budget_bitpacks(max_alloc_bytes, num_nodes);
        // (C2)
        const std::size_t b_total = unlimited
            ? std::numeric_limits<std::size_t>::max()
            : std::max<std::size_t>(ceil_div(total_bits_to_sample, bits_in_bitpack_), 1);
        // (C3)
        const std::size_t b_req = (requested_trials_per_pass > 0)
            ? std::max<std::size_t>(ceil_div(requested_trials_per_pass, bits_in_bitpack_), 1)
            : b_mem;

        // optimistic target, clamped by the hard upper bounds (C1) and (C2)
        std::size_t b = std::min(b_req, std::min(b_mem, b_total));
        b = std::max<std::size_t>(b, 1); // (C5)

        // (C4) round DOWN to a whole cache line, but never below one cache line and
        //      never to zero (skip when the target is already sub-cache-line).
        if (cache_line_bitpacks > 0 && b >= cache_line_bitpacks) {
            b -= (b % cache_line_bitpacks);
        }

        // (C6) round DOWN to a power of 2 (1 when b == 1).
        b = floor_power_of_2(b);
        return std::max<std::size_t>(b, 1); // (C5)
    }

    // =======================================================================
    //  Small integer helpers
    // =======================================================================
    /// Largest power of 2 not exceeding @p n (returns 1 for @p n ≤ 1).
    [[nodiscard]] static std::size_t floor_power_of_2(const std::size_t n) {
        return n <= 1 ? 1 : std::bit_floor(n);
    }

    [[nodiscard]] static std::size_t ceil_div(const std::size_t numerator, const std::size_t denominator) {
        return denominator ? (numerator + denominator - 1) / denominator : 0;
    }

    [[nodiscard]] static std::size_t round_up_to_multiple(const std::size_t value, const std::size_t multiple) {
        return multiple ? ceil_div(value, multiple) * multiple : value;
    }

    [[nodiscard]] static std::string build_strategy_label(const std::size_t requested_trials_per_pass,
                                                          const std::size_t cache_line_bitpacks) {
        std::string label = requested_trials_per_pass ? "requested_trials_per_pass" : "auto_memory_budget";
        if (cache_line_bitpacks > 0) {
            label += "_cacheline_aligned";
        }
        return label;
    }

protected:
    // Input parameters
    std::size_t requested_num_trials_ = 0;
    std::size_t num_nodes_ = 0;
    std::size_t requested_trials_per_pass_ = 0;

    // Device capabilities
    std::size_t max_device_bytes_ = 0;
    std::size_t max_device_bits_ = 0;

    // Computed constants
    static constexpr std::size_t bits_in_bitpack_ = sizeof(bitpack_t_) * static_cast<std::size_t>(8);

    // Intermediate calculations
    std::size_t num_trials_ = 0;
    std::size_t total_bits_to_sample_ = 0;
    std::size_t target_bits_per_iteration_ = 0;
    std::size_t bits_per_iteration_ = 0;
    std::size_t num_iterations_ = 0;
    std::size_t cache_line_bitpacks_ = 0;
    std::size_t memory_budget_bitpacks_per_node_ = 0;
    bool unlimited_iterations_ = false;
    std::string shape_strategy_ = "uninitialized";
};
}
