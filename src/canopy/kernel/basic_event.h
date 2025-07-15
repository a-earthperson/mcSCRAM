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
 * - Configurable bit-width sampling (1, 2, 4, 8, 16, 32, 64 bits)
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

#include "canopy/event/node.h"

#include <sycl/sycl.hpp>

namespace scram::canopy::kernel {

    /**
     * @class basic_event
     * @brief SYCL kernel class for parallel random sampling of basic events
     * 
     * @details This class implements a high-performance SYCL kernel that generates
     * random samples for basic events using the Philox counter-based pseudorandom
     * number generator. The kernel is designed for massive parallelization across
     * GPU threads, with each thread responsible for generating samples for specific
     * event-batch-bitpack combinations.
     * 
     * The implementation uses the Philox 4x32-10 PRNG algorithm, which provides
     * cryptographic-quality randomness with excellent parallelization properties.
     * Unlike linear congruential generators, Philox is counter-based, allowing
     * any sequence position to be computed directly without iterating through
     * previous values.
     * 
     * Key architectural features:
     * - Counter-based PRNG for perfect parallelization
     * - Deterministic seeding for reproducible results
     * - Bit-packed output for memory efficiency
     * - Configurable sampling bit widths
     * - Optimal work-group sizing for different GPU architectures
     * 
     * @tparam prob_t_ Floating-point type for probability values (typically float or double)
     * @tparam bitpack_t_ Integer type for bit-packed result storage (typically uint32_t or uint64_t)
     * @tparam size_t_ Integer type for indexing and sizes (typically uint32_t)
     * 
     * @note The kernel assumes basic_events_ array is allocated in unified shared memory
     * @note All random number generation is deterministic based on thread indices
     * @warning This class is designed for GPU execution; CPU performance may be suboptimal
     * 
     * @example Basic usage:
     * @code
     * // Create basic events array
     * auto events = create_basic_events<double, uint64_t>(queue, probabilities, indices, 1024);
     * 
     * // Create kernel instance
     * basic_event<double, uint64_t, uint32_t> kernel(events, num_events, sample_shape);
     * 
     * // Submit kernel for execution
     * queue.submit([&](sycl::handler& h) {
     *     auto range = kernel.get_range(num_events, local_range, sample_shape);
     *     h.parallel_for(range, [=](sycl::nd_item<3> item) {
     *         kernel(item, iteration);
     *     });
     * });
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_, typename size_t_>
    class basic_event {

        event::basic_event_block<prob_t_, bitpack_t_> basic_events_block_;

        /// @brief Configuration for sample batch dimensions and bit-packing
        const event::sample_shape<size_t_> sample_shape_;

    public:
        /**
         * @brief Constructs a basic event sampling kernel
         * 
         * @details Initializes the kernel with the basic events array and sampling
         * configuration. The kernel instance can be used multiple times for different
         * iterations of the sampling process.
         * 
         * @param basic_events_block reference to basic events block (must be in unified shared memory)
         * @param sample_shape Configuration defining batch size and bit-packing dimensions
         * 
         * @note The basic_events array must remain valid for the lifetime of the kernel
         * @note All parameters are stored by value/reference and should not be modified after construction
         * 
         * @example
         * @code
         * sample_shape<uint32_t> shape{1024, 16};
         * basic_event<double, uint64_t, uint32_t> kernel(events, num_events, shape);
         * @endcode
         */
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

        /**
         * @struct philox128_state
         * @brief State structure for Philox 4x32 PRNG algorithm
         * 
         * @details Contains the 128-bit state for the Philox pseudorandom number generator.
         * The state consists of four 32-bit values that are transformed through multiple
         * rounds of the Philox algorithm to produce high-quality random numbers.
         * 
         * This state can represent either:
         * - Input seeds/counters for initialization
         * - Intermediate state during round computations
         * - Final output containing 4 independent 32-bit random values
         * 
         * @note The state is designed to be efficiently processed by GPU threads
         * @note All 128 bits of entropy are utilized in the generation process
         * 
         * @example
         * @code
         * philox128_state seeds;
         * seeds.x[0] = event_id;
         * seeds.x[1] = batch_id;
         * seeds.x[2] = bitpack_id;
         * seeds.x[3] = iteration;
         * 
         * philox128_state results;
         * philox_generate(&seeds, &results);
         * @endcode
         */
        struct philox128_state {
            /// @brief Four 32-bit state values forming the complete 128-bit state
            uint32_t x[4];
        };

        /**
         * @brief Performs one round of the Philox 4x32 algorithm
         * 
         * @details Executes a single round of the Philox transformation, which consists
         * of multiplication, bit manipulation, and key mixing operations. The Philox
         * algorithm uses 10 rounds total to achieve cryptographic-quality randomness.
         * 
         * Each round performs:
         * 1. Multiply two state values by the Philox multiplication constants
         * 2. Split the 64-bit products into high and low 32-bit parts
         * 3. Mix the results with the remaining state values and round keys
         * 4. Advance the round keys for the next round
         * 
         * @param k0 [in,out] First round key, incremented after use
         * @param k1 [in,out] Second round key, incremented after use
         * @param counters [in,out] State values to be transformed
         * 
         * @note This is a static inline function for optimal performance
         * @note Round keys are automatically advanced for the next round
         * 
         * @example
         * @code
         * uint32_t k0 = 382307844, k1 = 293830103;
         * philox128_state state = {{seed1, seed2, seed3, seed4}};
         * philox_round(k0, k1, &state);  // Performs one round
         * @endcode
         */
        static inline void philox_round(uint32_t &k0, uint32_t &k1, philox128_state *counters) {
            // Multiply
            const uint64_t product0 = static_cast<uint64_t>(PHILOX_M4x32A) * counters->x[0];
            const uint64_t product1 = static_cast<uint64_t>(PHILOX_M4x32B) * counters->x[2];

            // Split into high and low parts
            philox128_state hi_lo;
            hi_lo.x[0] = static_cast<uint32_t>(product0 >> 32);
            hi_lo.x[1] = static_cast<uint32_t>(product0);
            hi_lo.x[2] = static_cast<uint32_t>(product1 >> 32);
            hi_lo.x[3] = static_cast<uint32_t>(product1);

            // Mix in the key
            counters->x[0] = hi_lo.x[2] ^ counters->x[1] ^ k0;
            counters->x[1] = hi_lo.x[3];
            counters->x[2] = hi_lo.x[0] ^ counters->x[3] ^ k1;
            counters->x[3] = hi_lo.x[1];

            // Bump the key
            k0 += PHILOX_W32A;
            k1 += PHILOX_W32B;
        }

        /**
         * @brief Generates 4 random uint32_t values using Philox 4x32-10 algorithm
         * 
         * @details Implements the complete Philox 4x32-10 pseudorandom number generator,
         * which applies 10 rounds of the Philox transformation to convert input seeds
         * into high-quality random numbers. The algorithm is designed for parallel
         * execution and produces cryptographically secure random values.
         * 
         * The function uses fixed round keys and applies the Philox round function
         * exactly 10 times to ensure proper mixing and randomness quality. The output
         * consists of four independent 32-bit random values that can be used for
         * sampling or further processing.
         * 
         * @param seeds Input seeds/counters for random generation
         * @param results [out] Output structure containing 4 random uint32_t values
         * 
         * @note Uses fixed round keys for consistent results across all threads
         * @note The algorithm is counter-based, allowing direct computation of any sequence position
         * @note All 128 bits of output are statistically independent and high-quality
         * 
         * @example
         * @code
         * philox128_state seeds = {{event_id, batch_id, bitpack_id, iteration}};
         * philox128_state results;
         * philox_generate(&seeds, &results);
         * 
         * // results.x[0..3] now contain 4 independent 32-bit random values
         * double rand_val = static_cast<double>(results.x[0]) / UINT32_MAX;
         * @endcode
         */
        static void philox_generate(const philox128_state *seeds, philox128_state *results) {
            // Key
            uint32_t k0 = 382307844;
            uint32_t k1 = 293830103;

            // Counter
            philox128_state counters = *seeds;

            // Number of rounds; Philox 4x32 uses 10 rounds
            #pragma unroll
            for(auto i=0; i<10;i++){
                philox_round(k0, k1, &counters);
            }
            *results = counters;
        }

        /**
         * @brief Generates bit-packed random samples based on probability threshold
         * 
         * @details Converts the four 32-bit random values from Philox into binary
         * samples by comparing normalized floating-point values against the given
         * probability threshold. Each comparison produces a single bit (0 or 1),
         * and multiple bits are packed together for efficient storage.
         * 
         * The function normalizes each 32-bit random value to the range [0, 1) and
         * compares it against the probability. If the random value is less than the
         * probability, the corresponding bit is set to 1, otherwise 0. This implements
         * the standard inverse transform method for Bernoulli sampling.
         * 
         * @tparam width Bit width specifying how many samples to generate (default: 4 bits)
         * 
         * @param seeds Input seeds for random number generation
         * @param threshold Threshold probability for Bernoulli sampling (range: [0, UINT32MAX])
         * 
         * @return Bit-packed samples with the specified bit width
         * 
         * @note The default 4-bit width utilizes all 4 random values from Philox
         * @note Normalization uses high-precision arithmetic to avoid bias
         * @note Results are deterministic given the same seeds and probability
         * 
         * @example
         * @code
         * philox128_state seeds = {{1, 2, 3, 4}};
         * double prob = 0.1;  // 10% probability
         * 
         * // Generate 4 bit-packed samples
         * auto samples = sample<four_bits>(&seeds, prob);
         * 
         * // Extract individual bits
         * bool bit0 = samples & 1;
         * bool bit1 = (samples >> 1) & 1;
         * bool bit2 = (samples >> 2) & 1;
         * bool bit3 = (samples >> 3) & 1;
         * @endcode
         */
        static bitpack_t_ sample(const philox128_state *seeds, const uint32_t &threshold) {
            philox128_state results;
            philox_generate(seeds, &results);

            bitpack_t_ out_bits = bitpack_t_(0);

            out_bits |= (results.x[0] < threshold ? 1 : 0) << 0;
            out_bits |= (results.x[1] < threshold ? 1 : 0) << 1;
            out_bits |= (results.x[2] < threshold ? 1 : 0) << 2;
            out_bits |= (results.x[3] < threshold ? 1 : 0) << 3;
            return out_bits;
        }

        /**
         * @struct sampler_args
         * @brief Argument structure for the random sample generation process
         * 
         * @details Encapsulates all parameters needed for deterministic random sample
         * generation. These arguments are used to construct unique seeds for the Philox
         * PRNG, ensuring that each thread generates independent random sequences while
         * maintaining reproducibility across runs.
         * 
         * The structure contains both identification parameters (for seeding) and
         * computational parameters (for the sampling process). The identification
         * parameters ensure that each unique combination of event, batch, and bitpack
         * produces a different random sequence.
         * 
         * @note All ID fields should be unique within their respective domains
         * @note The iteration parameter allows for multiple independent sampling rounds
         * 
         * @example
         * @code
         * sampler_args args = {
         *     .index_id = 42,      // Event index
         *     .event_id = 0,       // Event within batch
         *     .batch_id = 1,       // Batch index
         *     .bitpack_idx = 2,    // Bitpack within batch
         *     .iteration = 0,      // Iteration number
         *     .probability = 0.05  // 5% failure probability
         * };
         * auto samples = generate(args);
         * @endcode
         */
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
        };

        /**
         * @brief Generates a complete bitpack of random samples for a single event
         * 
         * @details Produces a full bitpack of random samples by performing multiple
         * rounds of Philox generation and bit-packing. The function automatically
         * calculates the number of rounds needed based on the bitpack type size
         * and the sampling bit width.
         * 
         * Each round generates 4 bits of samples, which are then combined to fill
         * the complete bitpack. The seeding is carefully constructed to ensure
         * each round produces independent random values while maintaining overall
         * sequence reproducibility.
         * 
         * @param args Sampling arguments containing all necessary parameters
         * 
         * @return Complete bitpack containing random samples
         * 
         * @note The function uses compile-time constants for optimal performance
         * @note Seeding incorporates all argument fields to ensure unique sequences
         * @note The loop is unrolled for better GPU performance
         * 
         * @example
         * @code
         * sampler_args args = {
         *     .index_id = 1, .event_id = 0, .batch_id = 0, 
         *     .bitpack_idx = 0, .iteration = 0, .probability = 0.1
         * };
         * uint64_t samples = generate(args);
         * // samples now contains 64 bits of random samples
         * @endcode
         */
        static bitpack_t_ generate(const sampler_args &args) {
            static constexpr std::uint8_t bernoulli_bits_per_generation = 4;
            static constexpr std::uint8_t bits_in_bitpack = sizeof(bitpack_t_) * 8;
            static constexpr std::uint8_t num_generations = bits_in_bitpack / bernoulli_bits_per_generation;

            philox128_state seeds;
            seeds.x[0] = args.index_id + 1;
            seeds.x[1] = args.event_id + 1;
            seeds.x[2] = args.batch_id + 1;
            seeds.x[3] = (args.bitpack_idx + args.iteration + 1) << 6; // spare 6 bits to store generation count (i)

            bitpack_t_ bitpacked_sample = bitpack_t_(0);
            #pragma unroll
            for (auto i = 0; i < num_generations; ++i) {
                seeds.x[3] += i;
                const auto generation_offset = bernoulli_bits_per_generation * i;
                bitpacked_sample |= sample(&seeds, args.prob_threshold) << generation_offset;
            }
            return bitpacked_sample;
        }

        /**
         * @brief SYCL kernel operator for parallel basic event sampling
         * 
         * @details This is the main kernel function executed by each SYCL thread.
         * It extracts thread indices from the nd_item, performs bounds checking,
         * retrieves event parameters, and generates random samples for the assigned
         * event-batch-bitpack combination.
         * 
         * The kernel operates in a 3D thread space:
         * - X dimension: Event index (different basic events)
         * - Y dimension: Batch index (different sample batches)
         * - Z dimension: Bitpack index (different bitpacks within a batch)
         * 
         * Each thread is responsible for generating one bitpack worth of samples
         * for its assigned event-batch-bitpack combination. The results are stored
         * directly in the event's output buffer.
         * 
         * @param item SYCL nd_item providing thread indices and group information
         * @param iteration Iteration number for this sampling round
         * 
         * @note Performs bounds checking to handle over-provisioned thread grids
         * @note Accesses unified shared memory for basic event parameters
         * @note Stores results directly in device memory buffers
         * 
         * @example
         * @code
         * // In SYCL kernel submission:
         * queue.submit([&](sycl::handler& h) {
         *     auto range = kernel.get_range(num_events, local_range, sample_shape);
         *     h.parallel_for(range, [=](sycl::nd_item<3> item) {
         *         kernel(item, 0);  // iteration = 0
         *     });
         * });
         * @endcode
         */
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

        /**
         * @brief Calculates optimal SYCL nd_range for kernel execution
         * 
         * @details Computes the global and local work-group sizes for optimal kernel
         * dispatch. The function ensures that global sizes are multiples of local
         * sizes (required by SYCL) and provides adequate thread coverage for all
         * work items.
         * 
         * The 3D execution space is organized as:
         * - X dimension: Number of basic events
         * - Y dimension: Batch size from sample shape
         * - Z dimension: Bitpacks per batch from sample shape
         * 
         * Global sizes are rounded up to the nearest multiple of local sizes to
         * ensure proper work-group alignment. Excess threads are handled by bounds
         * checking in the kernel operator.
         * 
         * @param num_events Number of basic events to process
         * @param local_range Desired local work-group size (should be optimized for target device)
         * @param sample_shape_ Sample shape configuration defining Y and Z dimensions
         * 
         * @return SYCL nd_range object ready for kernel submission
         * 
         * @note Global sizes are always multiples of corresponding local sizes
         * @note Over-provisioning is handled by kernel bounds checking
         * @note Local range should be tuned for specific GPU architecture
         * 
         * @example
         * @code
         * sycl::range<3> local_range(8, 16, 4);  // Optimized for specific GPU
         * auto nd_range = kernel.get_range(num_events, local_range, sample_shape);
         * 
         * queue.submit([&](sycl::handler& h) {
         *     h.parallel_for(nd_range, [=](sycl::nd_item<3> item) {
         *         kernel(item, iteration);
         *     });
         * });
         * @endcode
         */
        static sycl::nd_range<3> get_range(const size_t_ num_events,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &sample_shape_) {
            // Compute global range
            size_t global_size_x = num_events;
            size_t global_size_y = sample_shape_.batch_size;
            size_t global_size_z = sample_shape_.bitpacks_per_batch;

            // Adjust global sizes to be multiples of the corresponding local sizes
            global_size_x = ((global_size_x + local_range[0] - 1) / local_range[0]) * local_range[0];
            global_size_y = ((global_size_y + local_range[1] - 1) / local_range[1]) * local_range[1];
            global_size_z = ((global_size_z + local_range[2] - 1) / local_range[2]) * local_range[2];

            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);

            return {global_range, local_range};
        }
    };
}// namespace scram::canopy::kernel
