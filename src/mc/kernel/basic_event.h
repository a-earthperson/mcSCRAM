/**
 * @file basic_event.h
 * @brief SYCL kernel implementation for parallel basic event sampling using Philox PRNG
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details This file implements a high-performance SYCL kernel for generating random samples
 * from basic events in probabilistic analysis. It uses the Philox counter-based pseudorandom
 * number generator to ensure reproducible, high-quality random numbers suitable for Monte
 * Carlo simulations in parallel computing environments.
 *
 * The implementation provides efficient bit-packed sampling with configurable bit widths
 * and supports massive parallelization across GPU threads. Each basic event is sampled
 * independently using its failure probability, with results stored in bit-packed format
 * for memory efficiency and computational performance.
 *
 * Key features:
 * - Philox 4x32-10 PRNG for cryptographic-quality randomness
 * - Memory-efficient bit-packed storage
 * - Reproducible results with deterministic seeding
 * - Optimal SYCL kernel dispatch and work-group sizing
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
#include "mc/prng/philox128.h"
#include "mc/prng/state128.h"
#include "mc/prng/xorshift128.h"
#include "prng/sfc64.h"
#include "prng/wyrand.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace scram::mc::kernel {

    /**
     * @brief Device layout of the per-launch high-word tie records.
     *
     * The sampler kernel writes one record per slot on every launch, with no
     * data-dependent control flow; the fix-up kernel scans the records in
     * 8-byte chunks and recomputes only the tied work-items.
     *
     * - `slot_bytes == 1` (sub-group size 1, e.g. the OpenMP host device):
     *   one byte per work-item at `group * slots_per_group + local_linear_id`,
     *   non-zero iff that work-item drew a high-word tie.
     * - `slot_bytes == 4`: one `uint32_t` per sub-group at
     *   `group * slots_per_group + sub_group_id`, packing
     *   `(min_lid + 1) | (max_lid + 1) << 16` over the tied work-items'
     *   local linear ids (0 = no tie). The fix-up rechecks every lid in
     *   `[min_lid, max_lid]`, so the record is exact for one tie and
     *   conservative for several.
     */
    struct tie_slot_layout {
        std::uint8_t *data = nullptr;
        std::uint32_t slots_per_group = 0;
        std::uint32_t slot_bytes = 0;
        std::size_t num_chunks = 0;

        static constexpr std::size_t kChunkBytes = sizeof(std::uint64_t);
    };

    template<typename prob_t_, typename bitpack_t_, typename size_t_>
    class basic_event {

        using be_block = event::basic_event_block<prob_t_, bitpack_t_>;
        be_block basic_events_block_;

        using sample_shape = event::sample_shape<size_t_>;
        /// @brief Configuration for sample batch dimensions and bit-packing
        const sample_shape sample_shape_;

        tie_slot_layout tie_slots_;

    public:
        basic_event(
                const be_block &basic_events_block,
                const sample_shape &sample_shape,
                const tie_slot_layout &tie_slots)
                : basic_events_block_(basic_events_block),
                  sample_shape_(sample_shape),
                  tie_slots_(tie_slots) {}

        struct positional_counter  {
            /// @brief Basic event position in block
            /// range: [0, basic_events_block_.count)
            /// constraints: [[ event_idx < basic_events_block_.count ]]
            /// @warning typically, pdag_idx and event_idx could be correlated
            uint32_t event_idx;

            /// @brief Unique index identifier in the pdag
            /// range: [2, ?)
            uint32_t pdag_idx;

            /// @brief Batch index/position:
            /// range: [0, sample_shape_.batch_size)
            /// constraints: [[ batch_idx < sample_shape_.batch_size ]]
            uint32_t batch_idx;

            /// @brief Bitpack index within the current batch
            /// range: [0, sample_shape_.bitpacks_per_batch)
            /// constraints: [[ bitpack_idx < sample_shape_.bitpacks_per_batch ]]
            uint32_t bitpack_idx;

            /// @brief linearized index
            /// batch_idx * bitpacks_per_batch + bitpack_idx
            uint32_t sample_idx;

            /// @brief iteration index
            /// (iteration + 1) << 6,
            uint32_t iteration_idx;

            bool is_out_of_bounds;

            [[gnu::always_inline]] explicit positional_counter(const sycl::nd_item<3> &item,
                                                               const be_block &event_block,
                                                               const sample_shape &sample_shape,
                                                               const uint32_t &iteration)
                : positional_counter(static_cast<uint32_t>(item.get_global_id(0)),
                                     static_cast<uint32_t>(item.get_global_id(1)),
                                     static_cast<uint32_t>(item.get_global_id(2)),
                                     event_block, sample_shape, iteration) {}

            [[gnu::always_inline]] explicit positional_counter(const uint32_t global_x,
                                                               const uint32_t global_y,
                                                               const uint32_t global_z,
                                                               const be_block &event_block,
                                                               const sample_shape &sample_shape,
                                                               const uint32_t &iteration) {
                event_idx     = global_x;
                batch_idx     = global_y;
                bitpack_idx   = global_z;
                pdag_idx      = static_cast<uint32_t>(event_block[event_idx].index);
                sample_idx    = batch_idx * sample_shape.bitpacks_per_batch + bitpack_idx;
                iteration_idx = iteration;
                is_out_of_bounds = event_idx >= event_block.count || batch_idx >= sample_shape.batch_size || bitpack_idx >= sample_shape.bitpacks_per_batch;
            }

            [[gnu::always_inline]] [[nodiscard]] static prng::state128 fill(const positional_counter &args) {
                const prng::state128 to_fill = {
                    .x = {
                        args.pdag_idx + 1,
                        args.event_idx + 1,
                        args.sample_idx + 1,
                        (args.iteration_idx + 1) << 6, // spare 6 bits to store generation count (i)
                    }
                };
                return to_fill;
            }
        };

        void operator()(const sycl::nd_item<3> &item, const uint32_t iteration) const {

            const positional_counter args(item, basic_events_block_, sample_shape_, iteration);

            // Out-of-bounds work-items skip sampling but still reach the
            // sub-group collective in `record_tie`.
            bool any_tie = false;
            if (!args.is_out_of_bounds) {
                const prng::state128 seed_base = positional_counter::fill(args);

                const auto &p_threshold = basic_events_block_[args.event_idx].probability_threshold;
                const bitpack_t_ bitpack_value =
                    prng::philox::pack_bernoulli_draws_hi<bitpack_t_>(seed_base, p_threshold, any_tie);

                // Store the bitpacked samples into the buffer
                bitpack_t_ *output = basic_events_block_[args.event_idx].buffer;
                output[args.sample_idx] = bitpack_value;
            }
            record_tie(item, any_tie);
        }

        /// Unconditional, branch-free tie record (see `tie_slot_layout`).
        /// The sub-group size is a JIT-time constant, so only one arm survives.
        [[gnu::always_inline]] void record_tie(const sycl::nd_item<3> &item, const bool any_tie) const {
            const std::size_t group_base =
                static_cast<std::size_t>(item.get_group_linear_id()) * tie_slots_.slots_per_group;
            tie_slots_.data[group_base + item.get_local_linear_id()] = static_cast<std::uint8_t>(any_tie);
            if (true) return;
            const auto sg = item.get_sub_group();
            const auto lid = static_cast<std::uint32_t>(item.get_local_linear_id());
            const std::uint32_t lo = sycl::reduce_over_group(sg, any_tie ? lid + 1u : 0xffffffffu,
                                                             sycl::minimum<std::uint32_t>());
            const std::uint32_t hi = sycl::reduce_over_group(sg, any_tie ? lid + 1u : 0u,
                                                             sycl::maximum<std::uint32_t>());
            if (sg.leader()) {
                auto *slots = reinterpret_cast<std::uint32_t *>(tie_slots_.data);
                slots[group_base + sg.get_group_linear_id()] = hi == 0u ? 0u : (lo | (hi << 16));
            }
        }

        /**
         * @brief Allocate and zero the tie records for one `basic_event` launch shape.
         *
         * `slot_bytes` follows the device's sub-group sizes: 1 when the only
         * size is 1 (per-work-item flag), otherwise 4 (per-sub-group lid range).
         * Slots per group assume the smallest sub-group size, so every
         * sub-group of every legal size has its own slot.
         */
        static tie_slot_layout allocate_tie_slots(const sycl::queue &queue, const sycl::nd_range<3> &nd_range) {
            const auto sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
            std::size_t min_sg = 1;
            std::size_t max_sg = 1;
            if (!sizes.empty()) {
                min_sg = *std::min_element(sizes.begin(), sizes.end());
                max_sg = *std::max_element(sizes.begin(), sizes.end());
            }
            const auto local = nd_range.get_local_range();
            const auto global = nd_range.get_global_range();
            const std::size_t local_size = local.size();
            const std::size_t num_groups = global.size() / local_size;

            tie_slot_layout layout;
            layout.slot_bytes = max_sg == 1 ? 1u : 4u;
            layout.slots_per_group = static_cast<std::uint32_t>(
                layout.slot_bytes == 1 ? local_size : (local_size + min_sg - 1) / min_sg);
            const std::size_t used_bytes = num_groups * layout.slots_per_group * layout.slot_bytes;
            layout.num_chunks = (used_bytes + tie_slot_layout::kChunkBytes - 1) / tie_slot_layout::kChunkBytes;
            const std::size_t alloc_bytes = layout.num_chunks * tie_slot_layout::kChunkBytes;

            auto q = queue;
            layout.data = static_cast<std::uint8_t *>(
                sycl::aligned_alloc_device(tie_slot_layout::kChunkBytes, alloc_bytes, q));
            q.memset(layout.data, 0, alloc_bytes).wait_and_throw();
            return layout;
        }

        static sycl::nd_range<3> get_range(const size_t_ num_events,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &sample_shape_) {

            size_t global_size_x = num_events;
            size_t global_size_y = sample_shape_.batch_size;
            size_t global_size_z = sample_shape_.bitpacks_per_batch;

            // Round up to the next multiple of the local size in each dimension
            global_size_x = ((global_size_x + local_range[0] - 1) / local_range[0]) * local_range[0];
            global_size_y = ((global_size_y + local_range[1] - 1) / local_range[1]) * local_range[1];
            global_size_z = ((global_size_z + local_range[2] - 1) / local_range[2]) * local_range[2];

            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);
            LOG(INFO) << "kernel::basic_event::\tlocal_range{x,y,z}:(" << local_range[0] <<", " << local_range[1] <<", " << local_range[2] <<")\tglobal_range{x,y,z}:("<< global_range[0] <<", " << global_range[1] <<", " << global_range[2] <<")\tnum_events:" << num_events << " | " << sample_shape_;
            return {global_range, local_range};
        }
    };

    /**
     * @brief Fix-up pass for `basic_event`: rewrites every work-item whose
     * tie record is set with the exact two-word sample.
     *
     * One work-item per 8-byte chunk of tie records. A sampler work-item
     * ties with probability ≈ 64 · 2^-32 = 2^-26 (64-bit packs), so almost
     * every chunk is zero and its work-item exits after one load. The
     * recompute is out of line so neither the host vectorizer nor the GPU
     * register allocator sees the second Philox on the scan path.
     * Must run after the `basic_event` launch it fixes and with the same
     * `iteration`.
     */
    template<typename prob_t_, typename bitpack_t_, typename size_t_>
    class basic_event_tie_fixup {
        using sampler = basic_event<prob_t_, bitpack_t_, size_t_>;
        using positional_counter = typename sampler::positional_counter;
        using be_block = event::basic_event_block<prob_t_, bitpack_t_>;
        using sample_shape = event::sample_shape<size_t_>;

        be_block basic_events_block_;
        sample_shape sample_shape_;
        tie_slot_layout tie_slots_;
        std::uint32_t local_[3]{};
        std::uint32_t groups_[3]{};
        std::size_t num_groups_ = 0;

    public:
        basic_event_tie_fixup(const be_block &basic_events_block,
                              const sample_shape &sample_shape,
                              const tie_slot_layout &tie_slots,
                              const sycl::nd_range<3> &sampler_range)
            : basic_events_block_(basic_events_block),
              sample_shape_(sample_shape),
              tie_slots_(tie_slots) {
            const auto local = sampler_range.get_local_range();
            const auto global = sampler_range.get_global_range();
            for (int d = 0; d < 3; ++d) {
                local_[d] = static_cast<std::uint32_t>(local[d]);
                groups_[d] = static_cast<std::uint32_t>(global[d] / local[d]);
            }
            num_groups_ = static_cast<std::size_t>(groups_[0]) * groups_[1] * groups_[2];
        }

        void operator()(const sycl::nd_item<1> &item, const uint32_t iteration) const {
            const std::size_t chunk = item.get_global_id(0);
            if (chunk >= tie_slots_.num_chunks) {
                return;
            }
            const std::uint64_t word = reinterpret_cast<const std::uint64_t *>(tie_slots_.data)[chunk];
            if (word != 0) [[unlikely]] {
                fix_chunk(chunk, word, iteration);
            }
        }

        static sycl::nd_range<1> get_range(const tie_slot_layout &tie_slots, const std::size_t max_work_group_size) {
            const std::size_t local = std::min<std::size_t>(256, std::max<std::size_t>(1, max_work_group_size));
            const std::size_t global = ((std::max<std::size_t>(tie_slots.num_chunks, 1) + local - 1) / local) * local;
            return {sycl::range<1>(global), sycl::range<1>(local)};
        }

    private:
        [[gnu::noinline]] void fix_chunk(const std::size_t chunk, const std::uint64_t word, const uint32_t iteration) const {
            const std::uint32_t slots_per_chunk = tie_slot_layout::kChunkBytes / tie_slots_.slot_bytes;
            for (std::uint32_t k = 0; k < slots_per_chunk; ++k) {
                const std::size_t slot = chunk * slots_per_chunk + k;
                const std::size_t group = slot / tie_slots_.slots_per_group;
                if (group >= num_groups_) {
                    return;
                }
                std::uint32_t first_lid = 0;
                std::uint32_t last_lid = 0;
                if (tie_slots_.slot_bytes == 1) {
                    if (((word >> (8u * k)) & 0xffu) == 0) {
                        continue;
                    }
                    first_lid = last_lid = static_cast<std::uint32_t>(slot % tie_slots_.slots_per_group);
                } else {
                    const auto record = static_cast<std::uint32_t>(word >> (32u * k));
                    if (record == 0) {
                        continue;
                    }
                    first_lid = (record & 0xffffu) - 1u;
                    last_lid = (record >> 16) - 1u;
                }
                for (std::uint32_t lid = first_lid; lid <= last_lid; ++lid) {
                    fix_work_item(group, lid, iteration);
                }
            }
        }

        /// SYCL linear ids are row-major with dimension 2 fastest, for both
        /// the group and the local id.
        void fix_work_item(const std::size_t group, const std::uint32_t lid, const uint32_t iteration) const {
            const auto gz = static_cast<std::uint32_t>(group % groups_[2]);
            const auto gy = static_cast<std::uint32_t>((group / groups_[2]) % groups_[1]);
            const auto gx = static_cast<std::uint32_t>(group / (static_cast<std::size_t>(groups_[2]) * groups_[1]));
            const std::uint32_t lz = lid % local_[2];
            const std::uint32_t ly = (lid / local_[2]) % local_[1];
            const std::uint32_t lx = lid / (local_[2] * local_[1]);

            const std::uint32_t x = gx * local_[0] + lx;
            const std::uint32_t y = gy * local_[1] + ly;
            const std::uint32_t z = gz * local_[2] + lz;
            if (x >= basic_events_block_.count || y >= sample_shape_.batch_size || z >= sample_shape_.bitpacks_per_batch) {
                return;
            }
            const positional_counter args(x, y, z, basic_events_block_, sample_shape_, iteration);
            const prng::state128 seed_base = positional_counter::fill(args);
            const auto &p_threshold = basic_events_block_[x].probability_threshold;
            basic_events_block_[x].buffer[args.sample_idx] =
                prng::philox::pack_bernoulli_draws<bitpack_t_>(seed_base, p_threshold);
        }
    };
}// namespace scram::mc::kernel