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
#include "mc/prng/state128.h"
#include "mc/prng/xorshift128.h"
#include "mc/prng/philox128.h"

#include <sycl/sycl.hpp>

namespace scram::mc::kernel {

    template<typename prob_t_, typename bitpack_t_, typename size_t_>
    class basic_event {

        event::basic_event_block<prob_t_, bitpack_t_> basic_events_block_;

        /// @brief Configuration for sample batch dimensions and bit-packing
        const event::sample_shape<size_t_> sample_shape_;

    public:
        basic_event(
                const event::basic_event_block<prob_t_, bitpack_t_> &basic_events_block,
                const event::sample_shape<size_t_> &sample_shape)
                : basic_events_block_(basic_events_block),
                  sample_shape_(sample_shape) {}

        struct positional_counter  {
            /// @brief Unique index identifier for the event (used for seeding)
            uint32_t index_id;

            /// @brief Event identifier within the current batch
            uint32_t event_id;

            /// @brief Batch identifier for the current sampling round
            uint32_t batch_id;

            /// @brief Bitpack index within the current batch
            uint32_t bitpack_idx;
        };

        void operator()(const sycl::nd_item<3> &item, const uint32_t iteration) const {
            const auto blk_idx = static_cast<uint32_t>(item.get_global_id(0));
            positional_counter args = {
                .index_id    = static_cast<uint32_t>(basic_events_block_[blk_idx].index),
                .event_id    = static_cast<uint32_t>(item.get_global_id(0)),
                .batch_id    = static_cast<uint32_t>(item.get_global_id(1)),
                .bitpack_idx = static_cast<uint32_t>(item.get_global_id(2)),
            };

            // Bounds checking
            if (args.event_id >= basic_events_block_.count ||
                args.batch_id >= sample_shape_.batch_size ||
                args.bitpack_idx >= sample_shape_.bitpacks_per_batch) {
                return;
            }

            const prng::state128 seed_base = {
                .x = {
                    args.index_id + 1,
                    args.event_id + 1,
                    args.batch_id + 1,
                    (args.bitpack_idx + iteration + 1) << 6,  // spare 6 bits to store generation count (i)
                },
            };

            const auto &p_threshold = basic_events_block_[blk_idx].probability_threshold;
            const bitpack_t_ bitpack_value = prng::philox::pack_bernoulli_draws<bitpack_t_>(seed_base, p_threshold);

            // Store the bitpacked samples into the buffer
            bitpack_t_ *output = basic_events_block_[args.event_id].buffer;
            // Calculate the index within the generated_samples buffer
            const size_t_ index = args.batch_id * sample_shape_.bitpacks_per_batch + args.bitpack_idx;
            output[index] = bitpack_value;
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
            LOG(DEBUG3) << "kernel::basic_event:: local_range{x,y,z}:(" << local_range[0] <<", " << local_range[1] <<", " << local_range[2] <<")\t global_range{x,y,z}:(" << "events:"<< global_size_x <<", batch_size:"<< global_size_y <<", sample_shape_.bitpacks_per_batch:"<<global_size_z<<")";
            return {global_range, local_range};
        }
    };
}// namespace scram::mc::kernel