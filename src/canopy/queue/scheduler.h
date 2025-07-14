/**
 * @file scheduler.h
 * @brief SYCL-based MC-scheduling helper for the compute graph layers.
 * @author Arjun Earthperson
 * @date 2025
 *
 *
 * @copyright Copyright (C) 2025 Arjun Earthperson
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "canopy/event/node.h"
#include "canopy/working_set.h"

#include "logger.h"

#include <sycl/sycl.hpp>

namespace scram::canopy::queue {
template<typename bitpack_t_>
struct scheduler {

    std::size_t TOTAL_ITERATIONS = 0;
    event::sample_shape<std::size_t> SAMPLE_SHAPE{};

    explicit scheduler(){}

    scheduler(const sycl::queue &queue, const std::size_t requested_num_trials, const std::size_t num_nodes) {

        const sycl::device device = queue.get_device();
        const std::size_t max_device_bytes = device.get_info<sycl::info::device::max_mem_alloc_size>();
        const std::size_t max_device_bits = max_device_bytes * static_cast<std::size_t>(8);

        // round number of sampled bits to nearest multiple of bits in bitpack_t_
        constexpr std::size_t bits_in_bitpack = sizeof(bitpack_t_) * static_cast<std::size_t>(8);
        const std::size_t num_trials = requested_num_trials + bits_in_bitpack - (requested_num_trials % bits_in_bitpack);

        // num_trials is now the number of bits to sample, over all iterations, for each node.
        const std::size_t total_bits_to_sample = num_trials;

        // but the resident memory will need to be split between each node's outputs
        const std::size_t target_bits_per_iteration = max_device_bits / num_nodes;

        // compute the optimal sample shape for each node's output per iteration
        const event::sample_shape<std::size_t> sample_shape = compute_optimal_sample_shape_for_bits(device, target_bits_per_iteration);

        // the actual number of bits per sample shape per iteration
        const std::size_t bits_per_iteration = sample_shape.num_bitpacks() * bits_in_bitpack;

        // so, it will take these many iterations to collect the total samples
        const std::size_t num_iterations = total_bits_to_sample / bits_per_iteration;

        TOTAL_ITERATIONS = num_iterations;
        SAMPLE_SHAPE = sample_shape;

        LOG(DEBUG2) << working_set<std::size_t, bitpack_t_>(queue, num_nodes, sample_shape);
    }

    static event::sample_shape<std::size_t> compute_optimal_sample_shape_for_bitpacks(const sycl::device &device,
                                                                                      const std::size_t bitpack_count) {
        // Heuristic search balancing Y (batch_size) and Z (bitpacks_per_batch)
        // while respecting device limits and maximizing utilized bitpacks.
        event::sample_shape<std::size_t> shape{1, 1};

        // ---------------------------------------------------------------------
        // 1) Query hardware limits that constrain the split
        // ---------------------------------------------------------------------
        const auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
        const std::size_t limit_y = max_sizes[1]; // Y-dimension (batch_size)
        const std::size_t limit_z = max_sizes[2]; // Z-dimension (bitpacks_per_batch)

        // Largest available sub-group (warp/wavefront) size – may be empty on CPUs.
        std::vector<std::size_t> sg_sizes;
        if (device.has(sycl::aspect::gpu)) {
            sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        }
        const std::size_t subgroup = sg_sizes.empty() ? 1 : *std::max_element(sg_sizes.begin(), sg_sizes.end());

        // ---------------------------------------------------------------------
        // 2) Helper: highest power-of-two ≤ v  (0 → 0)
        // ---------------------------------------------------------------------
        const auto highest_pow2_le = [](std::size_t v) -> std::size_t {
            if (v == 0)
                return 0;
            std::size_t p = 1;
            while ((p << 1) <= v)
                p <<= 1;
            return p;
        };

        // ---------------------------------------------------------------------
        // 3) Enumerate candidate batch sizes (powers of two, multiples of subgroup)
        // ---------------------------------------------------------------------
        std::size_t best_bs = 1;
        std::size_t best_ss = 1;
        std::size_t best_product = 0;

        const std::size_t max_bs_start = std::min(limit_y, bitpack_count);
        std::size_t bs = subgroup ? subgroup : 1;
        if ((bs & (bs - 1)) != 0) { // round up to next power of two if needed
            bs = highest_pow2_le(bs) << 1;
        }

        for (; bs && bs <= max_bs_start; bs <<= 1) {
            if (bs % subgroup)
                continue; // honour SIMD alignment

            const std::size_t max_ss = std::min(limit_z, bitpack_count / bs);
            const std::size_t ss = highest_pow2_le(max_ss);
            if (ss == 0)
                continue;

            const std::size_t product = bs * ss;
            if (product > best_product) {
                best_product = product;
                best_bs = bs;
                best_ss = ss;
                if (product == bitpack_count)
                    break; // perfect utilization
            }
        }

        // ---------------------------------------------------------------------
        // 4) Fallback for very small totals (or pathological limits)
        // ---------------------------------------------------------------------
        if (best_product == 0) {
            best_bs = std::min<std::size_t>(subgroup, limit_y);
            best_ss = highest_pow2_le(std::min(limit_z, bitpack_count / best_bs));
            best_ss = std::max<std::size_t>(best_ss, 1);
        }

        shape.batch_size = best_bs;
        shape.bitpacks_per_batch = best_ss;

        // Sanity: never exceed requested capacity
        assert(shape.num_bitpacks() <= bitpack_count);
        return shape;
    }

    static event::sample_shape<std::size_t> compute_optimal_sample_shape_for_bits(const sycl::device &device, const std::size_t bit_count) {
        constexpr std::size_t bits_in_bitpack = sizeof(bitpack_t_) * static_cast<std::size_t>(8);
        const std::size_t rounded_bit_count = bit_count + bits_in_bitpack - (bit_count % bits_in_bitpack);
        const std::size_t rounded_bitpack_count = rounded_bit_count / bits_in_bitpack;
        return compute_optimal_sample_shape_for_bitpacks(device, rounded_bitpack_count);
    }
};
}

