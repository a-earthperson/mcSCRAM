/**
 * @file gate.h
 * @brief SYCL kernel implementations for parallel logical gate operations
 * @author Arjun Earthperson
 * @date 2025
 * 
 * @details This file implements high-performance SYCL kernels for computing logical
 * gate operations (AND, OR, XOR, NOT, NAND, NOR, ATLEAST) on bit-packed data in
 * parallel. The implementation provides efficient computation of boolean logic across
 * massive datasets using GPU parallelization.
 * 
 * The kernels operate on bit-packed representations where each bitpack contains
 * multiple independent boolean samples. This design enables simultaneous evaluation
 * of thousands of logical expressions with minimal memory overhead and optimal
 * computational throughput.
 * 
 * Key architectural features:
 * - Template-based compile-time optimization for specific gate types
 * - Bit-packed parallel processing for memory efficiency
 * - Support for mixed positive/negative input logic
 * - Specialized implementations for different logical operations
 * - Configurable work-group sizing for different GPU architectures
 * - At-least-k-out-of-n gate support for complex reliability analysis
 * 
 * Performance optimization strategies:
 * - Compile-time operation specialization using template metaprogramming
 * - Efficient bit manipulation with unrolled loops
 * - Optimal memory access patterns for GPU architectures
 * - Work-group level parallelization for large input sets
 * 
 * @note All kernels assume bit-packed input data with consistent formatting
 * @note Gates support both positive and negated inputs through offset indexing
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

#include <sycl/sycl.hpp>

/**
 * @section optimization_strategies Performance Optimization Strategies
 * 
 * @subsection work_item_distribution Split Per-Bit Accumulation Across Work-Items
 * 
 * @details The current implementation processes all inputs sequentially within each
 * work-item, which can become a bottleneck for gates with large numbers of inputs.
 * Advanced optimization strategies can distribute this workload across multiple
 * threads for improved performance:
 * 
 * **Strategy A: Distributed Input Processing**
 * 1. Each work-item processes only a subset of inputs (e.g., 8-16 inputs per thread)
 * 2. Partial results are stored in work-group local memory with per-bit counters
 * 3. Intra-group reduction combines partial bit-counts from all work-items
 * 4. Group leader writes final results or performs threshold comparisons
 * 
 * **Performance Benefits:**
 * - Transforms O(num_inputs) serial loop into O(num_inputs/group_size) parallel operation
 * - Reduces memory latency through better cache utilization
 * - Enables processing of very large gate fan-ins efficiently
 * - Particularly beneficial for at-least-k gates with high input counts
 * 
 * **Implementation Considerations:**
 * - Work-group size must be chosen based on input count distribution
 * - Local memory usage should be optimized for target GPU architecture
 * - Synchronization overhead must be balanced against parallel benefits
 * - Load balancing across work-groups requires careful input distribution
 * 
 * @example Advanced parallel reduction for large gates:
 * @code
 * // Distribute inputs across work-group threads
 * const auto items_per_thread = (num_inputs + group_size - 1) / group_size;
 * const auto start_idx = thread_id * items_per_thread;
 * const auto end_idx = sycl::min(start_idx + items_per_thread, num_inputs);
 * 
 * // Process subset of inputs
 * for (auto i = start_idx; i < end_idx; ++i) {
 *     // Process input[i] and accumulate partial results
 * }
 * 
 * // Reduce partial results across work-group
 * auto group_result = sycl::reduce_over_group(item.get_group(), partial_result, sycl::plus<>());
 * @endcode
 */
namespace scram::canopy::kernel {

    /**
     * @class op
     * @brief Templated SYCL kernel for logical gate operations with compile-time optimization
     * 
     * @details This class template implements high-performance SYCL kernels for various
     * logical operations (AND, OR, XOR, NOT, NAND, NOR) on bit-packed data. The template
     * uses compile-time specialization to generate optimized code for each specific
     * operation type, eliminating runtime branching and maximizing GPU throughput.
     * 
     * The kernel operates in a 3D execution space where each thread processes one
     * gate-batch-bitpack combination. The bit-packed representation allows simultaneous
     * evaluation of multiple boolean expressions within a single bitpack operation.
     * 
     * **Key Design Features:**
     * - Compile-time operation specialization for zero-overhead abstraction
     * - Support for mixed positive/negative input logic
     * - Efficient bit manipulation with minimal branching
     * - Configurable initialization based on operation type
     * - Optimal memory access patterns for GPU architectures
     * 
     * **Operation Types Supported:**
     * - AND: All inputs must be true
     * - OR: Any input must be true  
     * - XOR: Odd number of inputs must be true
     * - NOT: Logical negation of single input
     * - NAND: NOT AND operation
     * - NOR: NOT OR operation
     * - NULL: Pass-through operation
     * 
     * @tparam OpType Logical operation type from core::Connective enum
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for indexing and sizes
     * 
     * @note The class uses CRTP-style template specialization for performance
     * @note All operations support both positive and negated inputs
     * @warning Template specialization requires compile-time constant operation types
     * 
     * @example Basic gate kernel usage:
     * @code
     * // Create AND gate kernel
     * op<core::Connective::kAnd, uint64_t, uint32_t> and_kernel(gates, num_gates, sample_shape);
     * 
     * // Submit kernel for execution
     * queue.submit([&](sycl::handler& h) {
     *     auto range = and_kernel.get_range(num_gates, local_range, sample_shape);
     *     h.parallel_for(range, [=](sycl::nd_item<3> item) {
     *         and_kernel(item);
     *     });
     * });
     * @endcode
     */
    template<core::Connective OpType, typename bitpack_t_, typename size_t_>
    class op {
    protected:
        /// @brief Contiguous block of gates (and associated buffers)
        const event::gate_block<bitpack_t_, size_t_> gates_block_;
        
        /// @brief Configuration for sample batch dimensions and bit-packing
        const event::sample_shape<size_t_> sample_shape_;

    public:
        /**
         * @brief Constructs a logical gate operation kernel
         * 
         * @details Initializes the kernel with the gates array and sampling configuration.
         * The kernel instance can be used multiple times for different execution contexts.
         * 
         * @param gates_block Pointer to array of gates (must be in unified shared memory)
         * @param sample_shape Configuration defining batch size and bit-packing dimensions
         * 
         * @note The gates array must remain valid for the lifetime of the kernel
         * @note All parameters are stored by reference and should not be modified after construction
         * 
         * @example
         * @code
         * sample_shape<uint32_t> shape{1024, 16};
         * op<core::Connective::kOr, uint64_t, uint32_t> or_kernel(gates, num_gates, shape);
         * @endcode
         */
        op(const event::gate_block<bitpack_t_, size_t_> &gates_block, const event::sample_shape<size_t_> &sample_shape)
            : gates_block_(gates_block),
              sample_shape_(sample_shape) {}

        /**
         * @brief Calculates optimal SYCL nd_range for gate kernel execution
         * 
         * @details Computes the global and local work-group sizes for optimal kernel
         * dispatch. The function ensures that global sizes are multiples of local
         * sizes (required by SYCL) and provides adequate thread coverage for all
         * gate operations.
         * 
         * The 3D execution space is organized as:
         * - X dimension: Number of gates
         * - Y dimension: Batch size from sample shape
         * - Z dimension: Bitpacks per batch from sample shape
         * 
         * @param num_gates Number of gates to process
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
         * auto nd_range = op::get_range(num_gates, local_range, sample_shape);
         * @endcode
         */
        static sycl::nd_range<3> get_range(const size_t_ num_gates,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &sample_shape_) {
            // Compute global range
            auto global_size_x = static_cast<size_t>(num_gates);
            auto global_size_y = static_cast<size_t>(sample_shape_.batch_size);
            auto global_size_z = static_cast<size_t>(sample_shape_.bitpacks_per_batch);

            // Adjust global sizes to be multiples of the corresponding local sizes
            global_size_x = ((global_size_x + local_range[0] - 1) / local_range[0]) * local_range[0];
            global_size_y = ((global_size_y + local_range[1] - 1) / local_range[1]) * local_range[1];
            global_size_z = ((global_size_z + local_range[2] - 1) / local_range[2]) * local_range[2];

            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);

            return {global_range, local_range};
        }

        /**
         * @brief Initializes bitpack value based on the logical operation type
         * 
         * @details Provides compile-time initialization of bitpack values based on
         * the operation type. AND-type operations (AND, NAND) start with all bits
         * set to 1, while OR-type operations (OR, NOR, XOR) start with all bits
         * set to 0. This initialization ensures correct logical accumulation.
         * 
         * **Operation-specific initialization:**
         * - AND/NAND: Start with all 1s (0xFFFFFFFF...) - any 0 input yields 0
         * - OR/NOR/XOR: Start with all 0s (0x00000000...) - any 1 input affects result
         * - NOT/NULL: Initialization doesn't matter as result is directly assigned
         * 
         * @return Initial bitpack value for the operation type
         * 
         * @note This is a compile-time constant expression for optimal performance
         * @note Template specialization eliminates runtime branching
         * 
         * @example
         * @code
         * // AND operation initialization
         * constexpr auto and_init = op<core::Connective::kAnd, uint64_t, uint32_t>::init_bitpack();
         * // and_init == 0xFFFFFFFFFFFFFFFF
         * 
         * // OR operation initialization  
         * constexpr auto or_init = op<core::Connective::kOr, uint64_t, uint32_t>::init_bitpack();
         * // or_init == 0x0000000000000000
         * @endcode
         */
        static constexpr bitpack_t_ init_bitpack() {
            return (OpType == core::Connective::kAnd || OpType == core::Connective::kNand) ? ~bitpack_t_(0) : 0;
        }

        /**
         * @brief SYCL kernel operator for parallel logical gate computation
         * 
         * @details This is the main kernel function executed by each SYCL thread.
         * It processes one gate-batch-bitpack combination, performing the specified
         * logical operation on all inputs and storing the result in the gate's output buffer.
         * 
         * **Algorithm Overview:**
         * 1. Extract thread indices and perform bounds checking
         * 2. Initialize result bitpack based on operation type
         * 3. Process positive inputs with the specified logical operation
         * 4. Process negated inputs with the same logical operation
         * 5. Apply final negation if required (NAND, NOR, NOT)
         * 6. Store result in the gate's output buffer
         * 
         * **Input Processing:**
         * - Positive inputs: indices [0, negated_inputs_offset)
         * - Negated inputs: indices [negated_inputs_offset, num_inputs)
         * - Each input contributes to the logical operation result
         * 
         * @param item SYCL nd_item providing thread indices and group information
         * 
         * @note Performs bounds checking to handle over-provisioned thread grids
         * @note Accesses unified shared memory for gate parameters and input data
         * @note Uses compile-time template specialization for optimal performance
         * 
         * @example Operation-specific behavior:
         * @code
         * // AND gate: result = input1 & input2 & input3 & ~input4
         * // OR gate:  result = input1 | input2 | input3 | ~input4
         * // XOR gate: result = input1 ^ input2 ^ input3 ^ ~input4
         * @endcode
         */
        void operator()(const sycl::nd_item<3> &item) const {
            const auto gate_id = static_cast<size_t_>(item.get_global_id(0));
            const auto batch_id = static_cast<size_t_>(item.get_global_id(1));
            const auto bitpack_idx = static_cast<size_t_>(item.get_global_id(2));

            // Bounds checking
            if (gate_id >= this->gates_block_.count || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
                return;
            }

            // Compute the linear index into the buffer
            const size_t_ index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;

            // Get gate
            const auto &g = gates_block_[gate_id];
            const size_t_ num_inputs = g.num_inputs;
            const size_t_ negations_offset = g.negated_inputs_offset;
            // ---------------------------------------------------------------------
            // 1) Initialize, depending on the base operation
            //    (AND-type ops start with all bits=1, OR/XOR-type ops start with 0)
            // ---------------------------------------------------------------------
            bitpack_t_ result = init_bitpack();

            // ---------------------------------------------------------------------
            // 2) Do the base operation, looping over one word from each input
            // ---------------------------------------------------------------------
            for (size_t_ i = 0; i < negations_offset; ++i) {
                const bitpack_t_ val = g.inputs[i][index];
                if constexpr (OpType == core::Connective::kOr || OpType == core::Connective::kNor) {
                    result |= val;
                } else if constexpr (OpType == core::Connective::kAnd || OpType == core::Connective::kNand) {
                    result &= val;
                } else if constexpr (OpType == core::Connective::kXor)// OpType == core::Connective::kXnor
                {
                    result ^= val;
                } else if constexpr (OpType == core::Connective::kNull || OpType == core::Connective::kNot) {
                    result = val;
                }
            }

            for (size_t_ i = negations_offset; i < num_inputs; ++i) {
                const bitpack_t_ val = ~(g.inputs[i][index]);
                if constexpr (OpType == core::Connective::kOr || OpType == core::Connective::kNor) {
                    result |= val;
                } else if constexpr (OpType == core::Connective::kAnd || OpType == core::Connective::kNand) {
                    result &= val;
                } else if constexpr (OpType == core::Connective::kXor)// OpType == core::Connective::kXnor
                {
                    result ^= val;
                } else if constexpr (OpType == core::Connective::kNull || OpType == core::Connective::kNot) {
                    result = val;
                }
            }

            // ---------------------------------------------------------------------
            // 3) If this is a negated op (NOT, NAND, NOR, XNOR), invert the result
            // ---------------------------------------------------------------------
            if constexpr (OpType == core::Connective::kNand || OpType == core::Connective::kNor || OpType == core::Connective::kNot) {
                result = ~result;
            }

            // 4) Write final result into the gate's output buffer
            g.buffer[index] = result;
        }
    };

    /**
     * @class op<core::Connective::kAtleast, bitpack_t_, size_t_>
     * @brief Specialized SYCL kernel for at-least-k-out-of-n gate operations
     * 
     * @details This template specialization implements high-performance at-least-k-out-of-n
     * gate operations, where the output is true if at least k out of n inputs are true.
     * This is a fundamental building block for reliability analysis, voting systems,
     * and redundancy modeling.
     * 
     * The implementation uses a bit-counting algorithm that processes
     * all bits in parallel. For each bit position across all inputs, it counts how
     * many inputs have that bit set, then compares against the threshold to determine
     * the output bit value.
     * 
     * **Algorithm Complexity:**
     * - Time: O(num_inputs × num_bits) per gate
     * - Space: O(num_bits) for accumulation arrays
     * - Parallel: All bits processed simultaneously
     * 
     * **Special Cases:**
     * - k = 0: Always outputs true (trivial case)
     * - k = 1: Equivalent to OR gate
     * - k = n: Equivalent to AND gate  
     * - k > n: Always outputs false (impossible case)
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for indexing and sizes
     * 
     * @note This specialization handles atleast_gate structures instead of basic gates
     * @note The algorithm is optimized for parallel bit processing on GPUs
     * @warning Large input counts may require work-group level parallelization
     * 
     * @example At-least gate applications:
     * @code
     * // 2-out-of-3 voting system
     * atleast_gate<uint64_t, uint32_t> voting_gate;
     * voting_gate.at_least = 2;
     * voting_gate.num_inputs = 3;
     * 
     * // Triple redundancy system (all must work)
     * atleast_gate<uint64_t, uint32_t> redundancy_gate;
     * redundancy_gate.at_least = 3;
     * redundancy_gate.num_inputs = 3;
     * 
     * // Majority vote (more than half)
     * atleast_gate<uint64_t, uint32_t> majority_gate;
     * majority_gate.at_least = 3;  // 3 out of 5
     * majority_gate.num_inputs = 5;
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    class op<core::Connective::kAtleast, bitpack_t_, size_t_> {
    protected:
        /// @brief Pointer to array of at-least gates to be processed
        event::atleast_gate<bitpack_t_, size_t_> *gates_;
        
        /// @brief Number of at-least gates in the array
        const size_t_ num_gates_;
        
        /// @brief Configuration for sample batch dimensions and bit-packing
        const event::sample_shape<size_t_> sample_shape_;

    public:
        /**
         * @brief Constructs an at-least gate operation kernel
         * 
         * @details Initializes the kernel with the at-least gates array and sampling
         * configuration. The kernel is specifically designed for processing gates
         * with at-least-k-out-of-n logic.
         * 
         * @param gates Pointer to array of at-least gates (must be in unified shared memory)
         * @param num_gates Number of at-least gates in the gates array
         * @param sample_shape Configuration defining batch size and bit-packing dimensions
         * 
         * @note The gates array must remain valid for the lifetime of the kernel
         * @note At-least gates contain additional threshold information beyond basic gates
         * 
         * @example
         * @code
         * sample_shape<uint32_t> shape{1024, 16};
         * op<core::Connective::kAtleast, uint64_t, uint32_t> atleast_kernel(gates, num_gates, shape);
         * @endcode
         */
        op(event::atleast_gate<bitpack_t_, size_t_> *gates, const size_t_ &num_gates, const event::sample_shape<size_t_> &sample_shape)
            : gates_(gates),
              num_gates_(num_gates),
              sample_shape_(sample_shape) {}

        /**
         * @brief Calculates optimal SYCL nd_range for at-least gate kernel execution
         * 
         * @details Computes the global and local work-group sizes for optimal kernel
         * dispatch. The function is identical to the base gate implementation but
         * provided for consistency and potential future specialization.
         * 
         * @param num_gates Number of at-least gates to process
         * @param local_range Desired local work-group size (should be optimized for target device)
         * @param sample_shape_ Sample shape configuration defining Y and Z dimensions
         * 
         * @return SYCL nd_range object ready for kernel submission
         * 
         * @note At-least gates may benefit from different work-group sizes due to higher complexity
         * @note Consider input count distribution when choosing local range
         * 
         * @example
         * @code
         * sycl::range<3> local_range(4, 16, 8);  // Smaller X for complex gates
         * auto nd_range = op::get_range(num_gates, local_range, sample_shape);
         * @endcode
         */
        static sycl::nd_range<3> get_range(const size_t_ num_gates,
                                           const sycl::range<3> &local_range,
                                           const event::sample_shape<size_t_> &sample_shape_) {
            // Compute global range
            auto global_size_x = static_cast<size_t>(num_gates);
            auto global_size_y = static_cast<size_t>(sample_shape_.batch_size);
            auto global_size_z = static_cast<size_t>(sample_shape_.bitpacks_per_batch);

            // Adjust global sizes to be multiples of the corresponding local sizes
            global_size_x = ((global_size_x + local_range[0] - 1) / local_range[0]) * local_range[0];
            global_size_y = ((global_size_y + local_range[1] - 1) / local_range[1]) * local_range[1];
            global_size_z = ((global_size_z + local_range[2] - 1) / local_range[2]) * local_range[2];

            sycl::range<3> global_range(global_size_x, global_size_y, global_size_z);

            return {global_range, local_range};
        }

        /**
         * @brief SYCL kernel operator for parallel at-least gate computation
         * 
         * @details This is the main kernel function for at-least-k-out-of-n operations.
         * It implements a bit-counting algorithm that processes all bit
         * positions in parallel, counting how many inputs have each bit set and
         * comparing against the threshold.
         * 
         * **Algorithm Steps:**
         * 1. Extract thread indices and perform bounds checking
         * 2. Initialize per-bit accumulation counters to zero
         * 3. Process positive inputs, accumulating bit counts for each position
         * 4. Process negated inputs, accumulating bit counts for each position
         * 5. Compare accumulated counts against threshold for each bit position
         * 6. Construct result bitpack from threshold comparisons
         * 7. Store result in the gate's output buffer
         * 
         * **Bit Processing:**
         * - Each bit position is processed independently
         * - Accumulation counters track how many inputs have each bit set
         * - Threshold comparison determines output bit value
         * - Unrolled loops optimize for GPU execution
         * 
         * @param item SYCL nd_item providing thread indices and group information
         * 
         * @note Uses sycl::marray for efficient per-bit accumulation
         * @note Unrolled loops provide optimal GPU performance
         * @note Supports both positive and negated inputs through offset indexing
         * 
         * @example At-least computation flow:
         * @code
         * // For 2-out-of-3 gate with inputs [1010, 1100, 0110]:
         * // Bit 0: count=1, 1>=2? false -> output bit 0 = 0
         * // Bit 1: count=2, 2>=2? true  -> output bit 1 = 1  
         * // Bit 2: count=2, 2>=2? true  -> output bit 2 = 1
         * // Bit 3: count=1, 1>=2? false -> output bit 3 = 0
         * // Result: 0110
         * @endcode
         */
        void operator()(const sycl::nd_item<3> &item) const {
            const auto gate_id = static_cast<std::uint32_t>(item.get_global_id(0));
            const auto batch_id = static_cast<std::uint32_t>(item.get_global_id(1));
            const auto bitpack_idx = static_cast<std::uint32_t>(item.get_global_id(2));

            // Bounds checking
            if (gate_id >= this->num_gates_ || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
                return;
            }

            // Compute the linear index into the buffer
            const std::uint32_t index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;

            // Get gate
            const auto &g = gates_[gate_id];
            const auto num_inputs = g.num_inputs;
            const auto negations_offset = g.negated_inputs_offset;

            static constexpr bitpack_t_ NUM_BITS = sizeof(bitpack_t_) * 8;
            sycl::marray<bitpack_t_, NUM_BITS> accumulated_counts(0);

            // for each input, accumulate the counts for each bit-position
            for (auto i = 0; i < negations_offset; ++i) {
                const bitpack_t_ val = g.inputs[i][index];
                #pragma unroll
                for (auto idx = 0; idx < NUM_BITS; ++idx) {
                    accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
                }
            }

            for (auto i = negations_offset; i < num_inputs; ++i) {
                const bitpack_t_ val = ~(g.inputs[i][index]);
                #pragma unroll
                for (auto idx = 0; idx < NUM_BITS; ++idx) {
                    accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
                }
            }

            // at_least = 0   -> always one
            // at_least = 1   -> or gate
            // at_least = k   -> k of n
            // at_least = n   -> and gate
            // at_least = n+1 -> always zero
            const auto threshold = g.at_least;

            bitpack_t_ result = 0;

            #pragma unroll
            for (auto idx = 0; idx < NUM_BITS; ++idx) {
                result |= ((accumulated_counts[idx] >= threshold ? 1 : 0) << idx);
            }

            g.buffer[index] = result;
        }
    };
}
            //     const auto gate_id     = static_cast<size_t_>(item.get_global_id(0));
            //     const auto batch_id    = static_cast<size_t_>(item.get_global_id(1));
            //     const auto bitpack_idx = static_cast<size_t_>(item.get_global_id(2));
            //
            //     // Bounds checking
            //     if (gate_id >= this->num_gates_ || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
            //         return;
            //     }
            //
            //     // Compute the linear index into the buffer
            //     const size_t_ index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;
            //
            //     // This single gate might have many input buffers to combine ("k-of-n" logic)
            //     const auto& g = gates_[gate_id];
            //     const auto num_inputs = g.num_inputs;
            //     const auto threshold  = g.at_least;
            //
            //
            //     const auto grp        = item.get_group();         // sycl::group<3>
            //     const auto local_id   = item.get_local_linear_id();
            //     const auto group_size = grp.get_local_range().size();
            //
            //     // We step over the inputs in increments of group_size.
            //     // e.g. thread local_id processes i, i+group_size, i+2group_size, ...
            //     // until i >= num_inputs
            //     std::uint8_t private_counts[64];
            //     for(int b = 0; b < 64; b++)
            //         private_counts[b] = 0;
            //
            //     for(std::uint32_t i = local_id; i < num_inputs; i += group_size)
            //     {
            //         // Read input
            //         const std::uint64_t val = g.inputs[i][index];
            //         // Accumulate bits
            //         for(int b = 0; b < 64; b++) {
            //             private_counts[b] += (val >> b) & 1ULL;
            //         }
            //     }
            //
            //     // Now use sycl::reduce_over_group() to sum each bit's counts across the group
            //     for(int b = 0; b < 64; b++) {
            //         private_counts[b] = sycl::reduce_over_group(grp, private_counts[b], sycl::plus<>());
            //     }
            //     // Only one thread in each work-group should do the threshold check + write
            //     if(item.get_local_linear_id() == 0)
            //     {
            //         std::uint64_t final_result = 0ULL;
            //         for(int b = 0; b < 64; b++) {
            //             if(private_counts[b] >= threshold) {
            //                 final_result |= (1ULL << b);
            //             }
            //         }
            //         g.buffer[index] = final_result;
            //     }
            // }
        // };
        //
        // template<>
        // inline void op<core::Connective::kAtleast, std::uint8_t, std::uint32_t>::operator()(const sycl::nd_item<3> &item) const {
        //     const auto gate_id     = static_cast<std::uint32_t>(item.get_global_id(0));
        //     const auto batch_id    = static_cast<std::uint32_t>(item.get_global_id(1));
        //     const auto bitpack_idx = static_cast<std::uint32_t>(item.get_global_id(2));
        //
        //     // Bounds checking
        //     if (gate_id >= this->num_gates_ || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
        //         return;
        //     }
        //
        //     // Compute the linear index into the buffer
        //     const std::uint32_t index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;
        //
        //     // Get gate
        //     const auto& g = gates_[gate_id];
        //     const auto num_inputs = g.num_inputs;
        //     const auto negations_offset = g.negated_inputs_offset;
        //     //sycl::marray<std::uint8_t, 8> accumulated_counts = {0, 0, 0, 0, 0, 0, 0, 0};
        //     sycl::uchar8 accumulated_counts = {0, 0, 0, 0, 0, 0, 0, 0};
        //
        //     // for each input, accumulate the counts for each bit-position
        //     for (auto i = 0; i < negations_offset; ++i) {
        //
        //         const std::uint8_t val = g.inputs[i][index];
        //
        //         accumulated_counts[0] += (val & 0b00000001 ? 1 : 0);
        //         accumulated_counts[1] += (val & 0b00000010 ? 1 : 0);
        //         accumulated_counts[2] += (val & 0b00000100 ? 1 : 0);
        //         accumulated_counts[3] += (val & 0b00001000 ? 1 : 0);
        //         accumulated_counts[4] += (val & 0b00010000 ? 1 : 0);
        //         accumulated_counts[5] += (val & 0b00100000 ? 1 : 0);
        //         accumulated_counts[6] += (val & 0b01000000 ? 1 : 0);
        //         accumulated_counts[7] += (val & 0b10000000 ? 1 : 0);
        //     }
        //
        //     for (auto i = negations_offset; i < num_inputs; ++i) {
        //
        //         const std::uint8_t val = ~(g.inputs[i][index]);
        //
        //         accumulated_counts[0] += (val & 0b00000001 ? 1 : 0);
        //         accumulated_counts[1] += (val & 0b00000010 ? 1 : 0);
        //         accumulated_counts[2] += (val & 0b00000100 ? 1 : 0);
        //         accumulated_counts[3] += (val & 0b00001000 ? 1 : 0);
        //         accumulated_counts[4] += (val & 0b00010000 ? 1 : 0);
        //         accumulated_counts[5] += (val & 0b00100000 ? 1 : 0);
        //         accumulated_counts[6] += (val & 0b01000000 ? 1 : 0);
        //         accumulated_counts[7] += (val & 0b10000000 ? 1 : 0);
        //     }
        //
        //     // at_least = 0   -> always one
        //     // at_least = 1   -> or gate
        //     // at_least = k   -> k of n
        //     // at_least = n   -> and gate
        //     // at_least = n+1 -> always zero
        //     const auto threshold = g.at_least;
        //
        //     std::uint8_t result = accumulated_counts[0] >= threshold ? 1 : 0;
        //     result |= (accumulated_counts[1] >= threshold ? 1 : 0) << 1;
        //     result |= (accumulated_counts[2] >= threshold ? 1 : 0) << 2;
        //     result |= (accumulated_counts[3] >= threshold ? 1 : 0) << 3;
        //     result |= (accumulated_counts[4] >= threshold ? 1 : 0) << 4;
        //     result |= (accumulated_counts[5] >= threshold ? 1 : 0) << 5;
        //     result |= (accumulated_counts[6] >= threshold ? 1 : 0) << 6;
        //     result |= (accumulated_counts[7] >= threshold ? 1 : 0) << 7;
        //
        //     g.buffer[index] = result;
        // }

        // template<>
        // inline void op<core::Connective::kAtleast, std::uint8_t, std::uint32_t>::operator()(const sycl::nd_item<3> &item) const {
        //     const auto gate_id     = static_cast<std::uint32_t>(item.get_global_id(0));
        //     const auto batch_id    = static_cast<std::uint32_t>(item.get_global_id(1));
        //     const auto bitpack_idx = static_cast<std::uint32_t>(item.get_global_id(2));
        //
        //     // Bounds checking
        //     if (gate_id >= this->num_gates_ || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
        //         return;
        //     }
        //
        //     // Compute the linear index into the buffer
        //     const std::uint32_t index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;
        //
        //     // Get gate
        //     const auto& g = gates_[gate_id];
        //     const auto num_inputs = g.num_inputs;
        //     const auto negations_offset = g.negated_inputs_offset;
        //
        //     using bitpack_t_ = std::uint8_t;
        //     static constexpr bitpack_t_ NUM_BITS = 8;
        //     sycl::marray<bitpack_t_, NUM_BITS> accumulated_counts(0);
        //
        //     // for each input, accumulate the counts for each bit-position
        //     for (auto i = 0; i < negations_offset; ++i) {
        //         const bitpack_t_ val = g.inputs[i][index];
        //         #pragma unroll
        //         for (auto idx = 0; idx < NUM_BITS; ++idx) {
        //             accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
        //         }
        //     }
        //
        //     for (auto i = negations_offset; i < num_inputs; ++i) {
        //         const bitpack_t_ val = ~(g.inputs[i][index]);
        //         #pragma unroll
        //         for (auto idx = 0; idx < NUM_BITS; ++idx) {
        //             accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
        //         }
        //     }
        //
        //     // at_least = 0   -> always one
        //     // at_least = 1   -> or gate
        //     // at_least = k   -> k of n
        //     // at_least = n   -> and gate
        //     // at_least = n+1 -> always zero
        //     const auto threshold = g.at_least;
        //
        //     bitpack_t_ result = 0;
        //
        //     #pragma unroll
        //     for (auto idx = 0; idx < NUM_BITS; ++idx) {
        //         result |= ((accumulated_counts[idx] >= threshold ? 1 : 0) << idx);
        //     }
        //
        //     g.buffer[index] = result;
        // }
        //
        // template<>
        // inline void op<core::Connective::kAtleast, std::uint64_t, std::uint32_t>::operator()(const sycl::nd_item<3> &item) const {
        //     const auto gate_id     = static_cast<std::uint32_t>(item.get_global_id(0));
        //     const auto batch_id    = static_cast<std::uint32_t>(item.get_global_id(1));
        //     const auto bitpack_idx = static_cast<std::uint32_t>(item.get_global_id(2));
        //
        //     // Bounds checking
        //     if (gate_id >= this->num_gates_ || batch_id >= this->sample_shape_.batch_size || bitpack_idx >= this->sample_shape_.bitpacks_per_batch) {
        //         return;
        //     }
        //
        //     // Compute the linear index into the buffer
        //     const std::uint32_t index = batch_id * sample_shape_.bitpacks_per_batch + bitpack_idx;
        //
        //     // Get gate
        //     const auto& g = gates_[gate_id];
        //     const auto num_inputs = g.num_inputs;
        //     const auto negations_offset = g.negated_inputs_offset;
        //
        //     sycl::marray<std::uint8_t, 64> accumulated_counts(0);
        //
        //     // for each input, accumulate the counts for each bit-position
        //     for (auto i = 0; i < negations_offset; ++i) {
        //         const std::uint64_t val = g.inputs[i][index];
        //         #pragma unroll
        //         for (auto idx = 0; idx < 64; ++idx) {
        //             accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
        //         }
        //     }
        //
        //     for (auto i = negations_offset; i < num_inputs; ++i) {
        //         const std::uint64_t val = ~(g.inputs[i][index]);
        //         #pragma unroll
        //         for (auto idx = 0; idx < 64; ++idx) {
        //             accumulated_counts[idx] += (val & (1 << i) ? 1 : 0);
        //         }
        //     }
        //
        //     // at_least = 0   -> always one
        //     // at_least = 1   -> or gate
        //     // at_least = k   -> k of n
        //     // at_least = n   -> and gate
        //     // at_least = n+1 -> always zero
        //     const auto threshold = g.at_least;
        //
        //     std::uint64_t result = 0;
        //
        //     #pragma unroll
        //     for (auto idx = 0; idx < 64; ++idx) {
        //         result |= ((accumulated_counts[idx] >= threshold ? 1 : 0) << idx);
        //     }
        //
        //     g.buffer[index] = result;
        // }
    //}
    //
    // // We will accumulate partial sums for each bit in a 64-bit block:
    // // partial_counters[b] (for b=0..63)
    // // Then reduce them across the group.
    // // We'll store each thread’s partial sums in a private array,
    // // then do a group-level reduction in local memory.
    //
    // // 1) Each thread accumulates partial sums for a subset of the inputs:
    // //    We'll divide "num_inputs" among all threads in this group.
    // // ------------------------------------------------------------------------
    // const size_t group_size = item.get_local_range(0) *
    //                           item.get_local_range(1) *
    //                           item.get_local_range(2);
    // const size_t local_id   = item.get_local_linear_id();
    //
    // // Compute chunk bounds
    // const size_t chunk      = (num_inputs + group_size - 1) / group_size;
    // const size_t start      = local_id * chunk;
    // const size_t end        = sycl::min(start + chunk, static_cast<size_t>(num_inputs));
    //
    // // Private counters for each bit of a 64-bit pack
    // // Using 32-bit or 16-bit counters depends on max num_inputs
    // // Here, 32-bit is safer if num_inputs can be large
    // std::uint8_t local_counters[64];
    // for (int b = 0; b < 64; b++) {
    //     local_counters[b] = 0;
    // }
    //
    // // Accumulate partial sums
    // for (size_t i = start; i < end; ++i)
    // {
    //     // Read one 64-bit word from input i
    //     const std::uint64_t val = g.inputs[i][index];
    //
    //     // For each bit, add 1 if set
    //     // A common optimization is to enumerate set bits, but we’ll keep it direct:
    //     for (int b = 0; b < 64; b++) {
    //         local_counters[b] += static_cast<std::uint8_t>((val >> b) & 1ULL);
    //     }
    // }
    //
    // // 2) Store partial sums in local memory, then reduce them across the group
    // // ------------------------------------------------------------------------
    // // We'll store each thread’s 64 counters in local memory at an offset
    // // of local_id*64. Then do a parallel reduction.
    // sycl::group<3>  grp = item.get_group();
    // sycl::range<3>  lrange = item.get_local_range();
    //
    // // A 2D local_accessor: [group_size, 64]
    // //   local_sums[lid][bit_index]
    // sycl::local_accessor<std::uint32_t, 2> local_sums(sycl::range<2>(group_size, 64), grp);
    //
    // // Write private counters to local memory
    // for (int b = 0; b < 64; b++) {
    //     local_sums[local_id][b] = local_counters[b];
    // }
    // item.barrier(sycl::access::fence_space::local_space);
    //
    // // Parallel reduction in local memory.
    // // Double‐tree approach: stride = group_size/2 down to 1
    // // At each step, only threads with local_id < stride add in values from local_id+stride
    // for (size_t stride = group_size / 2; stride > 0; stride /= 2)
    // {
    //     if (local_id < stride)
    //     {
    //         for (int b = 0; b < 64; b++)
    //         {
    //             local_sums[local_id][b] += local_sums[local_id + stride][b];
    //         }
    //     }
    //     // Barrier after each pass
    //     item.barrier(sycl::access::fence_space::local_space);
    // }
    //
    // // Now local_sums[0][b] holds the sum for all threads in group for bit b.
    // // 3) Compare sums to threshold, produce final 64-bit mask
    // //    We do this for exactly one thread (group leader, local_id=0).
    // if (local_id == 0)
    // {
    //     std::uint64_t final_mask = 0ULL;
    //     for (int b = 0; b < 64; b++)
    //     {
    //         if (local_sums[0][b] >= threshold) {
    //             final_mask |= (1ULL << b);
    //         }
    //     }
    //     g.buffer[index] = final_mask;
    // }
