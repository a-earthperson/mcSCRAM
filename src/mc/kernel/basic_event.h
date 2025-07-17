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

#include <sycl/sycl.hpp>
#include <functional> // for std::bit_or used in subgroup reduction

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

        /// @name Philox PRNG Constants
        /// @{

        /// @brief Philox round constant A for 32-bit operations
        static constexpr uint32_t PHILOX_W32A = 0x9E3779B9;

        /// @brief Philox round constant B for 32-bit operations
        static constexpr uint32_t PHILOX_W32B = 0xBB67AE85;

        /// @brief Philox multiplication constant A for 4x32 variant
        static constexpr uint32_t PHILOX_M4x32A = 0xD2511F53;

        /// @brief Philox multiplication constant B for 4x32 variant
        static constexpr uint32_t PHILOX_M4x32B = 0xCD9E8D57;

        /// @}

        struct philox128_state {
            /// @brief Four 32-bit state values forming the complete 128-bit state
            uint32_t x[4];
        };

        template<typename T>
        struct Vec2 {
            T A;
            T B;
        };

        template<typename T>
        struct Vec4 {
            T X;
            T Y;
            T Z;
            T W;
        };

        template<typename DW = std::uint64_t, typename W = std::uint32_t>
        [[gnu::always_inline]] static inline W mulhilo(const W a, const W b, W* hi) {
            DW product = static_cast<DW>(a) * static_cast<DW>(b);
            *hi = product >> (sizeof(W) * 8);
            return static_cast<W>(product);
        }

        [[gnu::always_inline]] static void philox_round(philox128_state &counters, Vec2<uint32_t> &key) {
            static constexpr Vec2<uint32_t> PHILOX_M4x32 = {
                .A = 0xD2511F53u,
                .B = 0xCD9E8D57u,
            };
            // Split into high and low parts
            Vec2<uint32_t> hi{};

            const Vec2<uint32_t> lo = {
                .A = mulhilo(PHILOX_M4x32.A, counters.x[0], &hi.A),
                .B = mulhilo(PHILOX_M4x32.B, counters.x[2], &hi.B),
            };

            // Mix in the key
            counters.x[0] = hi.B ^ counters.x[1] ^ key.A;
            counters.x[1] = lo.B;
            counters.x[2] = hi.A ^ counters.x[3] ^ key.B;
            counters.x[3] = lo.B;

            static constexpr Vec2<uint32_t> PHILOX_W32 = {
                .A = 0x9E3779B9u,
                .B = 0xBB67AE85u,
            };

            // Bump the key
            key.A += PHILOX_W32.A;
            key.B += PHILOX_W32.B;
        }

        [[gnu::always_inline]] static void philox128_generate(const philox128_state *seeds, philox128_state *results, const uint8_t generation) {
            // Key as Vec2
            Vec2<uint32_t> key = {
                .A = 382307844u,
                .B = 293830103u,
            };

            // Counter
            philox128_state counters = *seeds;
            counters.x[3] += generation;

            #define PHILOX4x32_DEFAULT_ROUNDS 10
            #pragma unroll
            for(auto i=0; i < PHILOX4x32_DEFAULT_ROUNDS; i++){
                philox_round(counters, key);
            }
            *results = counters;
        }

        [[gnu::always_inline]] static bitpack_t_ sample(const philox128_state *seeds, const uint32_t &threshold, const std::uint32_t generation) {
            philox128_state results;
            philox128_generate(seeds, &results, generation);

            static constexpr std::uint32_t bernoulli_bits_per_generation = 4;
            const std::uint32_t bernoulli_bits_offset = bernoulli_bits_per_generation * generation;

            using b = bitpack_t_;
            bitpack_t_ out_bits = b(0);

            out_bits |= (results.x[0] < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 0);
            out_bits |= (results.x[1] < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 1);
            out_bits |= (results.x[2] < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 2);
            out_bits |= (results.x[3] < threshold ? b(1) : b(0)) << bitpack_t_(bernoulli_bits_offset + 3);

            return out_bits;
        }

        struct sampler_args {
            /// @brief Unique index identifier for the event (used for seeding)
            uint32_t index_id;

            /// @brief Event identifier within the current batch
            uint32_t event_id;

            /// @brief Batch identifier for the current sampling round
            uint32_t batch_id;

            /// @brief Bitpack index within the current batch
            uint32_t bitpack_idx;

            /// @brief Iteration number for multiple sampling rounds
            uint32_t iteration;

            /// @brief 32-bit fixed-point threshold p·2^32 for Bernoulli sampling
            uint32_t prob_threshold;

            uint32_t generation;
        };

        [[gnu::always_inline]] static bitpack_t_ generate(const sampler_args &args) {
            static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
            static constexpr std::uint8_t num_generations = sizeof(bitpack_t_) * 8 / bernoulli_bits_per_generation;

            const philox128_state seed_base = {
                .x = {
                    args.index_id + 1,
                    args.event_id + 1,
                    args.batch_id + 1,
                    (args.bitpack_idx + args.iteration + 1) << 6,  // spare 6 bits to store generation count (i)
                },
            };

            bitpack_t_ bitpacked_sample = bitpack_t_(0);
            #pragma unroll
            for (std::uint32_t i = 0; i < num_generations; ++i) {
                const bitpack_t_ four_bits = sample(&seed_base, args.prob_threshold, i);
                bitpacked_sample |= four_bits;
            }
            return bitpacked_sample;
        }

        void operator()(const sycl::nd_item<3> &item, const uint32_t iteration) const {
            const auto blk_idx = static_cast<uint32_t>(item.get_global_id(0));
            sampler_args args = {
                .index_id    = static_cast<uint32_t>(basic_events_block_[blk_idx].index),
                .event_id    = static_cast<uint32_t>(item.get_global_id(0)),
                .batch_id    = static_cast<uint32_t>(item.get_global_id(1)),
                .bitpack_idx = static_cast<uint32_t>(item.get_global_id(2)),
                .iteration = iteration,
                .prob_threshold = basic_events_block_[blk_idx].probability_threshold,
            };

            // Bounds checking
            if (args.event_id >= basic_events_block_.count || args.batch_id >= sample_shape_.batch_size || args.bitpack_idx >= sample_shape_.bitpacks_per_batch) {
                return;
            }

            // Calculate the index within the generated_samples buffer
            const size_t_ index = args.batch_id * sample_shape_.bitpacks_per_batch + args.bitpack_idx;

            // Store the bitpacked samples into the buffer
            bitpack_t_ *output = basic_events_block_[args.event_id].buffer;
            const bitpack_t_ bitpack_value = generate(args);
            output[index] = bitpack_value;
        }

        static sycl::nd_range<3> get_range(const size_t_ num_events,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &sample_shape_) {
            // In the revised kernel we launch one work-item per generation.
            // Therefore the Z dimension is (bitpacks_per_batch * local_range[2]).

            size_t global_size_x = num_events;
            size_t global_size_y = sample_shape_.batch_size;
            size_t global_size_z = sample_shape_.bitpacks_per_batch * local_range[2]; // local_range[2] == subgroup size

            // Round up to the next multiple of the local size in each dimension
            global_size_x = ((global_size_x + local_range[0] - 1) / local_range[0]) * local_range[0];
            global_size_y = ((global_size_y + local_range[1] - 1) / local_range[1]) * local_range[1];
            global_size_z = ((global_size_z + local_range[2] - 1) / local_range[2]) * local_range[2];

            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);
            LOG(DEBUG3) << "kernel::basic_event:: local_range{x,y,z}:(" << local_range[0] <<", " << local_range[1] <<", " << local_range[2] <<")\t global_range{x,y,z}:(" << "events:"<< global_size_x <<", batch_size:"<< global_size_y <<", sample_shape_.bitpacks_per_batch * local_range[2]:"<<global_size_z<<")";
            return {global_range, local_range};
        }
    };
}// namespace scram::mc::kernel