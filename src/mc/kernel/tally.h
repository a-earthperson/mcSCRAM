/**
 * @file tally.h
 * @brief Population-count kernel and statistics reducer for Monte-Carlo tallies
 * @author Arjun Earthperson
 * @date 2025
 *
 * This header defines `scram::mc::kernel::tally`, a SYCL kernel that
 *   1. counts the number of set bits (failures) in a bit-packed buffer, and
 *   2. derives probability estimates plus confidence intervals from those
 *      counts.
 *
 * Execution model
 * ---------------
 * Global ND-range ids are interpreted as `(tally_id, batch_idx, bitpack_id)`.
 * Every work-item touches **exactly one** `bitpack_t_` value:
 *
 *   idx = batch_idx × bitpacks_per_batch + bitpack_id
 *
 * An intra-group reduction collapses the per-item popcounts; the group leader
 * atomically adds the partial sum to `tally.num_one_bits`.
 *
 * Single-shot vs. repeated work
 * -----------------------------
 * • Bit counting is unique – each bit-pack is processed exactly once.
 * • Statistical post-processing (`mean`, `std_err`, `ci`) is executed **once per
 *   work-group** that maps to the same `tally_id`.  When the local Y/Z sizes do
 *   not cover the whole `(batch_size × bitpacks_per_batch)` tile, multiple work-
 *   groups are launched for one tally and this calculation is repeated.
 *   The end result is correct (all groups see the same `num_one_bits`) but the
 *   extra invocations waste GPU cycles and cause benign write races.
 *
 * To avoid the repetition choose a local range such that
 *
 *     local_range = { 1,
 *                     sample_shape.batch_size,
 *                     sample_shape.bitpacks_per_batch }
 *
 * or move the statistics step to a separate, single-work-group kernel.
 */

#pragma once

#include "mc/compat/metal.h"
#include "mc/event/node.h"
#include <sycl/sycl.hpp>

namespace scram::mc::kernel {

    /**
     * @class tally
     * @brief SYCL kernel for parallel statistical tally computation and analysis
     * 
     * @details This class implements a high-performance SYCL kernel that computes
     * statistical tallies from bit-packed simulation results. The kernel performs
     * parallel population counting across massive datasets and computes comprehensive
     * statistical measures including probability estimates, standard errors, and
     * confidence intervals.
     * 
     * The implementation uses a parallel reduction strategy that
     * minimizes atomic operations while maximizing GPU utilization. Each work-group
     * performs local reductions before contributing to global atomic accumulation,
     * significantly improving performance for large-scale statistical computations.
     * 
     * **Algorithm Overview:**
     * 1. **Parallel Population Counting**: Each thread processes one bitpack using
     *    hardware-accelerated popcount operations
     * 2. **Intra-Group Reduction**: Work-groups perform local reductions to minimize
     *    atomic contention on global memory
     * 3. **Atomic Accumulation**: Group leaders perform thread-safe updates to
     *    global bit count accumulators
     * 4. **Statistical Computation**: Final statistical measures are computed using
     *    standard Monte Carlo estimation techniques
     * 
     * **Statistical Measures Computed:**
     * - **Mean**: Sample proportion estimate of underlying probability
     * - **Standard Error**: Measure of estimation uncertainty
     * - **95% Confidence Interval**: Range containing true value with 95% probability
     * - **99% Confidence Interval**: Range containing true value with 99% probability
     * 
     * **Performance Characteristics:**
     * - Time Complexity: O(total_bits / num_threads) for population counting
     * - Space Complexity: O(1) per thread for local computation
     * - Scalability: Linear scaling with number of GPU compute units
     * - Memory Efficiency: Single-pass algorithm with minimal memory overhead
     * 
     * @tparam prob_t_ Floating-point type for probability and statistical calculations
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for indexing and counting operations
     * 
     * @note The kernel assumes tally_nodes_ array is allocated in unified shared memory
     * @note Statistical computation uses double-precision for numerical stability
     * @note Work-group organization is optimized for statistical computation patterns
     * 
     * @example Basic tally kernel usage:
     * @code
     * // Create tally events and kernel
     * auto tally_events = create_tally_events(queue, buffers, initial_counts);
     * tally<double, uint64_t, uint32_t> tally_kernel(tally_events, num_tallies, sample_shape);
     * 
     * // Submit kernel for execution
     * queue.submit([&](sycl::handler& h) {
     *     auto range = tally_kernel.get_range(num_tallies, local_range, sample_shape);
     *     h.parallel_for(range, [=](sycl::nd_item<3> item) {
     *         tally_kernel(item, iteration);
     *     });
     * });
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_, typename size_t_>
    class tally {
        /// @brief Wrapper owning the contiguous tally allocation
        event::tally_block<bitpack_t_> tallies_block_;
        
        /// @brief Configuration for sample batch dimensions and bit-packing
        const event::sample_shape<size_t_> sample_shape_;

    public:
        static_assert(compat::group_sum_fits_u32_v<bitpack_t_>,
                      "Metal uint32 group-sum requires bits(bitpack)*kMaxTallyLocalSize to fit in uint32");

        /**
         * @brief Constructs a statistical tally computation kernel
         * 
         * @details Initializes the kernel with the tally events array and sampling
         * configuration. The kernel is designed for efficient statistical computation
         * across multiple tally events simultaneously.
         * 
         * @param tally_blk Pointer to array of tally events (must be in unified shared memory)
         * @param sample_shape Configuration defining batch size and bit-packing dimensions
         * 
         * @note The tally_nodes array must remain valid for the lifetime of the kernel
         * @note Each tally event maintains its own statistical accumulators
         * @note Sample shape determines the total number of bits processed per tally
         * 
         * @example
         * @code
         * sample_shape<uint32_t> shape{1024, 16};
         * tally<double, uint64_t, uint32_t> tally_kernel(tally_events, num_tallies, shape);
         * @endcode
         */
        tally(const event::tally_block<bitpack_t_> &tally_blk,
              const event::sample_shape<size_t_> &sample_shape)
            : tallies_block_(tally_blk),
              sample_shape_(sample_shape) {}

        /**
         * @brief Calculates optimal SYCL nd_range for tally kernel execution
         * 
         * @details Computes the global and local work-group sizes optimized for
         * statistical computation. The function uses a specialized work-group
         * organization where the X-dimension is constrained to 1 to optimize
         * for the statistical reduction patterns used in tally computation.
         * 
         * The specialized organization improves performance by:
         * - Reducing synchronization overhead for atomic operations
         * - Optimizing memory access patterns for statistical computation
         * - Minimizing contention on global memory accumulators
         * - Enabling efficient intra-group reductions
         * 
         * **3D Execution Space Organization:**
         * - X dimension: Single tally per work-group (local_range[0] = 1)
         * - Y dimension: Batch size from sample shape
         * - Z dimension: Bitpacks per batch from sample shape
         * 
         * @param num_tallies Number of tally events to process
         * @param local_range Desired local work-group size (X dimension will be overridden)
         * @param shape Sample shape configuration defining Y and Z dimensions
         * 
         * @return SYCL nd_range object optimized for statistical computation
         * 
         * @note X dimension is automatically set to 1 for optimal tally computation
         * @note Work-group size affects atomic operation performance
         * @note The returned range ensures proper coverage of all tally operations
         * 
         * @example
         * @code
         * sycl::range<3> local_range(8, 16, 4);  // X will be overridden to 1
         * auto nd_range = tally::get_range(num_tallies, local_range, sample_shape);
         * // Actual local range will be (1, 16, 4)
         * @endcode
         */
        static sycl::nd_range<3> get_range(const size_t_ num_tallies,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &shape) {
            auto new_local_range = local_range;
            new_local_range[0] = 1;
            const size_t global_size_x = (num_tallies + new_local_range[0] - 1) / new_local_range[0] * new_local_range[0];
            const size_t global_size_y = (shape.batch_size + new_local_range[1] - 1) / new_local_range[1] * new_local_range[1];
            const size_t global_size_z = (shape.bitpacks_per_batch + new_local_range[2] - 1) / new_local_range[2] * new_local_range[2];
            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);

            LOG(INFO) << "kernel::tally_event::\tlocal_range{x,y,z}:(" << local_range[0] <<", " << local_range[1] <<", " << local_range[2] <<")\tglobal_range{x,y,z}:("<< global_range[0] <<", " << global_range[1] <<", " << global_range[2] <<")\tnum_tallies:" << num_tallies << " | " << shape;
            return {sycl::range<3>(global_size_x, global_size_y, global_size_z), new_local_range};
        }

        /**
         * @brief SYCL kernel operator for parallel statistical tally computation
         * 
         * @details This is the main kernel function that performs parallel population
         * counting followed by an (optional) statistical reduction.  The popcount part
         * is inherently unique per work-item.  The statistics part is executed by the
         * group leader **of every work-group** that processes the same tally, which
         * means it can run more than once per tally and per iteration.
         *
         * @warning When more than one work-group is launched for a given `tally_id`
         *          the call to `update_tally_stats` is duplicated.  Although the final
         *          values are identical (last writer wins) the extra work costs
         *          performance and introduces benign data races on the derived fields.
         *          Make `local_range[1] == batch_size` and `local_range[2] ==
         *          bitpacks_per_batch`, or move the statistics calculation to a
         *          follow-up kernel, to ensure the computation happens exactly once.
         *
         * **Algorithm Stages:**
         * 
         * **Stage 1: Parallel Population Counting**
         * - Each thread processes one bitpack using hardware popcount
         * - Threads extract bit counts from their assigned bitpack
         * - Local bit counts are computed independently
         * 
         * **Stage 2: Intra-Group Reduction**
         * - Work-groups perform local reductions to aggregate bit counts
         * - Uses SYCL's optimized reduce_over_group operation
         * - Reduces atomic contention on global memory
         * 
         * **Stage 3: Atomic Accumulation**
         * - Group leader performs thread-safe update to global accumulator
         * - Uses relaxed memory ordering for optimal performance
         * - Accumulates partial sums from all work-groups
         * 
         * **Stage 4: Statistical Computation**
         * - Group leader computes final statistical measures
         * - Updates mean, standard error, and confidence intervals
         * - Uses double-precision arithmetic for numerical stability
         * 
         * **Thread Organization:**
         * - 3D thread space: (tally_id, batch_id, bitpack_id)
         * - Each thread processes exactly one bitpack
         * - Work-groups are organized for efficient statistical computation
         * 
         * @param item SYCL nd_item providing thread indices and group information
         * @param iteration Current iteration number for statistical computation
         * 
         * @note Performs bounds checking to handle over-provisioned thread grids
         * @note Uses atomic operations for thread-safe global accumulation
         * @note Computes final statistics only on group leader threads
         * @note Assumes work-group organization optimized for tally computation
         * 
         * @example Kernel execution flow:
         * @code
         * // Thread (tally=0, batch=5, bitpack=10) processes:
         * // 1. Extract bitpack from tally_nodes_[0].buffer[5*16+10]
         * // 2. Count bits using popcount(bitpack)
         * // 3. Participate in group reduction
         * // 4. Group leader updates tally_nodes_[0].num_one_bits atomically
         * // 5. Group leader computes final statistics
         * @endcode
         */
        void operator()(const sycl::nd_item<3> &item, const uint32_t iteration) const {
            // Map global IDs to (tally_idx, batch_idx, bitpack_id)
            const size_t tally_idx = item.get_global_id(0);
            const size_t batch_idx = item.get_global_id(1);
            const size_t bitpack_id = item.get_global_id(2);

            // Bounds check
            if (tally_idx >= tallies_block_.count ||
                batch_idx >= sample_shape_.batch_size ||
                bitpack_id >= sample_shape_.bitpacks_per_batch) {
                return;
            }

            // Each work-item processes exactly one bitpack for this tally
            const std::size_t idx = batch_idx * sample_shape_.bitpacks_per_batch + bitpack_id;
            const std::size_t local_sum = sycl::popcount(tallies_block_.data[tally_idx].buffer[idx]);

            // Intra-group reduction: Metal uint32 (compat::group_sum), native size_t.
            const std::size_t group_sum = compat::group_sum(item.get_group(), local_sum);

            // Have the group leader accumulate bit counts in global memory
            if (item.get_local_linear_id() == 0) {

                static constexpr std::size_t num_bits_in_dtype_ = sizeof(bitpack_t_) * 8;
                const auto total_bits = static_cast<std::size_t>(iteration)
                                  * static_cast<std::size_t>(sample_shape_.batch_size)
                                  * static_cast<std::size_t>(sample_shape_.bitpacks_per_batch)
                                  * static_cast<std::size_t>(num_bits_in_dtype_);


                tallies_block_.data[tally_idx].total_bits = total_bits;
                compat::add_hits(tallies_block_.data[tally_idx].num_one_bits, group_sum);
            }

            // Host-side stats (populate_point_estimates). Device IR must stay
            // free of double: LLVMToMetal rejects it.
        }
    };

    } // namespace scram::mc::kernel
