/**
 * @file node.h
 * @brief SYCL-based node structures for parallel graph computation
 * @author Arjun Earthperson
 * @date 2025
 * 
 * @details This file defines the core data structures used in SYCL-based parallel
 * computation of probabilistic directed acyclic graphs (PDAGs). It provides template
 * structures for nodes, basic events, gates, and tally events, along with factory
 * functions for efficient device memory allocation and management.
 * 
 * The structures are designed for optimal performance on SYCL devices, using
 * bit-packed representations for efficient memory utilization and parallel processing.
 * All memory allocations are performed using SYCL unified shared memory for seamless
 * host-device data access.
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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>

namespace scram::canopy::event {

    /**
     * @struct node
     * @brief Base structure for all computation nodes in the graph
     * 
     * @details This is the fundamental building block for all node types in the
     * computation graph. It provides a device-accessible buffer for storing
     * computation results in a bit-packed format for memory efficiency.
     * 
     * @tparam bitpack_t_ Integer type used for bit-packed data storage
     * 
     * @note This structure is designed for SYCL device access
     * @note All derived structures inherit this buffer-based design
     * 
     * @example
     * @code
     * // Basic node with 64-bit packing
     * node<std::uint64_t> basic_node;
     * basic_node.buffer = sycl::malloc_device<std::uint64_t>(1024, queue);
     * @endcode
     */
    template<typename bitpack_t_>
    struct node {
        /// @brief Device-accessible buffer for storing computation results in bit-packed format
        bitpack_t_ *buffer;
    };

    /**
     * @struct basic_event
     * @brief Represents a basic event with associated probability and identification
     * 
     * @details Basic events are the leaf nodes of the computation graph and serve as inputs
     * to higher-level gate operations. They are processed in parallel using SYCL
     * kernels to generate random samples based on their failure probabilities.
     * 
     * @tparam prob_t_ Floating-point type for probability values
     * @tparam bitpack_t_ Integer type for bit-packed result storage
     * * @tparam index_t_ Signed integer type for the node index.
     * 
     * @note Probability values should be in the range [0.0, 1.0]
     * @note Index values must be unique within the computation graph
     * 
     * @example
     * @code
     * // Create a basic event with 0.01 failure probability
     * basic_event<double, std::uint64_t> pump_failure;
     * pump_failure.probability = 0.01;
     * pump_failure.index = 42;
     * pump_failure.buffer = sycl::malloc_device<std::uint64_t>(num_bitpacks, queue);
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_, typename index_t_ = int32_t>
    struct basic_event : node<bitpack_t_> {
        /// @brief Unique identifier for this event within the computation graph
        index_t_ index;

        std::uint32_t probability_threshold;
    };

    /**
     * @struct tally
     * @brief Accumulates and stores statistical results from Monte Carlo sampling
     * 
     * @details A tally event collects statistical information from multiple simulation
     * runs, computing mean probability estimates, standard errors, and confidence
     * intervals. This structure is used to aggregate results from parallel computations
     * and provide robust statistical measures for reliability analysis.
     * 
     * The tally process counts positive outcomes across all bit-packed samples and
     * computes statistical measures including confidence intervals at multiple levels
     * (95% and 99%). This enables quantification of uncertainty in probability estimates.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @note Statistical measures are computed using standard Monte Carlo methods
     * @note Confidence intervals use normal approximation for large sample sizes
     * 
     * @example
     * @code
     * // Process tally results
     * tally_event<std::uint64_t> tally_result;
     * if (tally_result.mean > 0.0) {
     *     std::cout << "Probability: " << tally_result.mean 
     *               << " ± " << tally_result.std_err << std::endl;
     *     std::cout << "95% CI: [" << tally_result.ci[0] << ", " << tally_result.ci[1] << "]" << std::endl;
     * }
     * @endcode
     */
    template<typename bitpack_t_>
    struct tally : node<bitpack_t_> {
        /// @brief Count of positive outcomes (1-bits) across all samples
        std::size_t num_one_bits = 0;
        
        /// @brief Estimated mean probability based on sample proportion
        std::double_t mean = 0.;
        
        /// @brief Standard error of the probability estimate
        std::double_t std_err = 0.;
        
        /// @brief Confidence intervals: [lower_95, upper_95, lower_99, upper_99]
        sycl::double4 ci = {0., 0., 0., 0.};
    };

    /**
     * @brief Factory function for creating device-allocated tally events
     * 
     * @details Creates an array of tally_event structures with unified shared memory
     * allocation for efficient host-device access. Each tally event is initialized
     * with the corresponding buffer and initial count values.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @param queue SYCL queue for memory allocation and device context
     * @param buffers Vector of device buffer pointers to associate with each tally
     * @param initial_values Vector of initial count values for each tally event
     * 
     * @return Pointer to array of allocated tally events
     * 
     * @note Uses unified shared memory for seamless host-device access
     * @note Caller is responsible for calling destroy_tally_event() for cleanup
     * 
     * @example
     * @code
     * std::vector<std::uint64_t*> buffers = {buffer1, buffer2, buffer3};
     * std::vector<std::size_t> initial_counts = {0, 0, 0};
     * auto tallies = create_tally_events(queue, buffers, initial_counts);
     * // Use tallies...
     * destroy_tally_event(queue, tallies);
     * @endcode
     */
    template<typename bitpack_t_>
    tally<bitpack_t_> *create_tally_events(const sycl::queue &queue, const std::vector<bitpack_t_ *> &buffers) {
        const auto num_tallies = buffers.size();
        tally<bitpack_t_> *tallies = sycl::malloc_shared<tally<bitpack_t_>>(num_tallies, queue);
        for (auto i = 0; i < num_tallies; ++i) {
            tallies[i].buffer = buffers[i];
            tallies[i].num_one_bits = 0;
            tallies[i].mean = 0.0;
            tallies[i].std_err = 0.0;
        }
        return tallies;
    }

    /**
     * @brief Destroys a tally event and frees associated device memory
     * 
     * @details Properly releases device memory allocated for a tally event structure.
     * This function should be called for all tally events created with create_tally_events()
     * to prevent memory leaks.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @param queue SYCL queue for memory deallocation
     * @param event Pointer to tally event to destroy
     * 
     * @note This function only frees the tally event structure, not the associated buffer
     * @note Buffers should be freed separately if they were allocated independently
     * 
     * @example
     * @code
     * auto tally = create_tally_events(queue, buffers, counts);
     * // Use tally...
     * destroy_tally_event(queue, tally);
     * @endcode
     */
    template<typename bitpack_t_>
    void destroy_tally_event(const sycl::queue queue, tally<bitpack_t_> *event) {
        sycl::free(event, queue);
    }

    /**
     * @struct gate
     * @brief Represents a logical gate with multiple inputs and configurable logic
     * 
     * @details A gate performs logical operations on multiple input signals, supporting
     * both positive and negated inputs. The gate structure is designed for efficient
     * parallel processing of boolean operations in pdags.
     * 
     * Gates can handle mixed positive and negative logic by specifying an offset where
     * negative inputs begin. This allows for efficient representation of complex logical
     * expressions without requiring separate NOT gates.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @note Input buffers are stored as an array of pointers for indirect access
     * @note The negated_inputs_offset defines the boundary between positive and negative inputs
     * 
     * @example
     * @code
     * // Gate with 3 positive inputs and 2 negated inputs
     * gate<std::uint64_t, std::uint32_t> or_gate;
     * or_gate.num_inputs = 5;
     * or_gate.negated_inputs_offset = 3;  // inputs[0-2] positive, inputs[3-4] negated
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    struct gate : node<bitpack_t_> {
        /// @brief Array of pointers to input buffers from other nodes
        bitpack_t_ **inputs;
        
        /// @brief Total number of input connections to this gate
        size_t_ num_inputs;
        
        /// @brief Offset where negated inputs begin (inputs[0..offset-1] are positive, inputs[offset..num_inputs-1] are negated)
        size_t_ negated_inputs_offset;
    };

    /**
     * @struct atleast_gate
     * @brief Specialized gate implementing at-least-k-out-of-n logic
     * 
     * @details An at-least gate (also known as a k-out-of-n gate) outputs true when
     * at least k out of n inputs are true. This is a generalization of AND gates
     * (k=n) and OR gates (k=1), commonly used in reliability analysis for redundant
     * systems and voting logic.
     * 
     * The gate supports mixed positive and negative logic through the inherited
     * negated_inputs_offset mechanism, allowing for complex logical expressions
     * like "at least 2 out of (A, B, NOT C, NOT D)".
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @note at_least value must be in range [0, num_inputs]
     * @note at_least = 0 always outputs true, at_least > num_inputs always outputs false
     * 
     * @example
     * @code
     * // 2-out-of-3 voting gate
     * atleast_gate<std::uint64_t, std::uint32_t> voting_gate;
     * voting_gate.num_inputs = 3;
     * voting_gate.at_least = 2;
     * voting_gate.negated_inputs_offset = 3;  // all inputs positive
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    struct atleast_gate : gate<bitpack_t_, size_t_> {
        /// @brief Minimum number of inputs that must be true for the gate to output true
        std::uint8_t at_least = 0;
    };

    /**
     * @brief Factory function for creating device-allocated at-least gates
     * 
     * @details Creates an array of atleast_gate structures with optimized memory layout
     * for parallel processing. Allocates both the gate structures and their associated
     * buffers in contiguous memory blocks, and sets up input pointer arrays for each gate.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @param queue SYCL queue for memory allocation and device context
     * @param inputs_per_gate Vector of input configurations, each containing input buffers and negated input count
     * @param atleast_per_gate Vector of at-least thresholds for each gate
     * @param num_bitpacks Number of bitpacks to allocate per gate output buffer
     * 
     * @return Pointer to array of allocated at-least gates
     * 
     * @throws std::bad_alloc if device memory allocation fails
     * 
     * @note Gate buffers are allocated as a single contiguous block for efficiency
     * @note Input pointer arrays are allocated separately for each gate
     * 
     * @example
     * @code
     * std::vector<std::pair<std::vector<std::uint64_t*>, std::uint32_t>> inputs = {
     *     {{buffer1, buffer2, buffer3}, 1},  // 3 inputs, 1 negated
     *     {{buffer4, buffer5}, 0}            // 2 inputs, 0 negated
     * };
     * std::vector<std::uint32_t> thresholds = {2, 1};  // 2-out-of-3, 1-out-of-2
     * auto gates = create_atleast_gates(queue, inputs, thresholds, 1024);
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    atleast_gate<bitpack_t_, size_t_> *create_atleast_gates(const sycl::queue &queue, const std::vector<std::pair<std::vector<bitpack_t_ *>, size_t_>> &inputs_per_gate, const std::vector<size_t_> &atleast_per_gate, const std::size_t num_bitpacks) {
        const auto num_gates = inputs_per_gate.size();
        // allocate all the gate objects contiguously
        atleast_gate<bitpack_t_, size_t_> *gates = sycl::malloc_shared<atleast_gate<bitpack_t_, size_t_>>(num_gates, queue);
        bitpack_t_* buffers = sycl::malloc_device<bitpack_t_>(num_gates * num_bitpacks, queue);

        for (auto i = 0; i < num_gates; ++i) {
            const auto gate_input_buffers = inputs_per_gate[i].first;
            const auto num_inputs = gate_input_buffers.size();
            const auto num_negated_inputs = inputs_per_gate[i].second;
            gates[i].num_inputs = static_cast<size_t_>(num_inputs);
            gates[i].negated_inputs_offset = static_cast<size_t_>(num_inputs - num_negated_inputs);
            gates[i].inputs = sycl::malloc_shared<bitpack_t_*>(num_inputs, queue);
            gates[i].buffer = buffers + i * num_bitpacks;
            for (auto j = 0; j < num_inputs; ++j) {
                gates[i].inputs[j] = gate_input_buffers[j];
            }
            gates[i].at_least = static_cast<size_t_>(atleast_per_gate[i]);
        }
        return gates;
    }

    /**
     * @brief Factory function for creating device-allocated standard gates
     * 
     * @details Creates an array of standard gate structures with optimized memory layout
     * for parallel processing. These gates perform basic logical operations (AND, OR, etc.)
     * without the specialized at-least-k logic.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @param queue SYCL queue for memory allocation and device context
     * @param inputs_per_gate Vector of input configurations, each containing input buffers and negated input count
     * @param num_bitpacks Number of bitpacks to allocate per gate output buffer
     * 
     * @return Pointer to array of allocated standard gates
     * 
     * @throws std::bad_alloc if device memory allocation fails
     * 
     * @note Gate buffers are allocated as a single contiguous block for efficiency
     * @note Input pointer arrays are allocated separately for each gate
     * 
     * @example
     * @code
     * std::vector<std::pair<std::vector<std::uint64_t*>, std::uint32_t>> inputs = {
     *     {{buffer1, buffer2}, 0},           // AND gate with 2 positive inputs
     *     {{buffer3, buffer4, buffer5}, 2}   // OR gate with 1 positive, 2 negated inputs
     * };
     * auto gates = create_gates(queue, inputs, 1024);
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    gate<bitpack_t_, size_t_> *create_gates(const sycl::queue &queue, const std::vector<std::pair<std::vector<bitpack_t_ *>, size_t_>> &inputs_per_gate, const std::size_t num_bitpacks) {
        const auto num_gates = inputs_per_gate.size();
        // allocate all the gate objects in a contiguous (shared) block
        gate<bitpack_t_, size_t_> *gates = sycl::malloc_shared<gate<bitpack_t_, size_t_>>(num_gates, queue);

        // allocate the actual buffers in a contiguous (device) block
        bitpack_t_* buffers = sycl::malloc_device<bitpack_t_>(num_gates * num_bitpacks, queue);

        for (auto i = 0; i < num_gates; ++i) {
            const std::vector<bitpack_t_ *> gate_input_buffers = inputs_per_gate[i].first;
            const auto num_inputs = gate_input_buffers.size();
            const auto num_negated_inputs = inputs_per_gate[i].second;
            assert(num_negated_inputs <= num_inputs);
            gates[i].num_inputs = static_cast<size_t_>(num_inputs);
            gates[i].negated_inputs_offset = static_cast<size_t_>(num_inputs - num_negated_inputs);
            gates[i].inputs = sycl::malloc_shared<bitpack_t_*>(num_inputs, queue);
            gates[i].buffer = buffers + i * num_bitpacks;
            for (auto j = 0; j < num_inputs; ++j) {
                gates[i].inputs[j] = gate_input_buffers[j];
            }
        }
        return gates;
    }

    /**
     * @brief Destroys a gate and frees all associated device memory
     * 
     * @details Properly releases device memory allocated for a gate structure,
     * including its input pointer array and output buffer. This function handles
     * the complete cleanup of gate-related memory allocations.
     * 
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @param queue SYCL queue for memory deallocation
     * @param gate_ptr Pointer to gate structure to destroy
     * 
     * @note This function frees the inputs array, buffer, and gate structure itself
     * @note Input buffers referenced by the inputs array are NOT freed (they belong to other nodes)
     * 
     * @example
     * @code
     * auto gates = create_gates(queue, inputs, 1024);
     * // Use gates...
     * for (int i = 0; i < num_gates; ++i) {
     *     destroy_gate(queue, &gates[i]);
     * }
     * @endcode
     */
    template<typename bitpack_t_, typename size_t_>
    void destroy_gate(const sycl::queue &queue, gate<bitpack_t_, size_t_> *gate_ptr) {
        sycl::free(gate_ptr->inputs, queue);// Free the inputs array
        sycl::free(gate_ptr->buffer, queue);// Free the output array
        sycl::free(gate_ptr, queue);        // Free the gate object
    }

    /**
     * @struct sample_shape
     * @brief Configuration structure for sampling dimensions and batch processing
     * 
     * @details Defines the shape and organization of sample data for parallel processing.
     * This structure encapsulates the batch size and bit-packing configuration used
     * throughout the computation pipeline for consistent memory layout and processing.
     * 
     * The sample shape determines how random samples are organized in memory and
     * processed by SYCL kernels. Proper configuration of these parameters is critical
     * for achieving optimal performance on different device architectures.
     * 
     * @tparam size_t_ Integer type for size and indexing
     * 
     * @note Total bitpacks = batch_size × bitpacks_per_batch
     * @note Batch size should be aligned with device work-group sizes for optimal performance
     * 
     * @example
     * @code
     * sample_shape<std::uint32_t> shape;
     * shape.batch_size = 1024;         // Process 1024 samples per batch
     * shape.bitpacks_per_batch = 16;   // Use 16 bitpacks per batch
     * 
     * auto total_bitpacks = shape.num_bitpacks();  // Returns 16384
     * @endcode
     */
    template<typename size_t_>
    struct sample_shape {
        /// @brief Number of samples processed in each batch
        size_t_ batch_size;
        
        /// @brief Number of bitpacks used per batch for bit-packed storage
        size_t_ bitpacks_per_batch;
        
        /**
         * @brief Calculates the total number of bitpacks needed
         * 
         * @details Computes the total number of bitpacks required for the configured
         * batch size and bitpacks per batch. This value is used for memory allocation
         * and kernel dispatch calculations.
         * 
         * @return Total number of bitpacks (batch_size × bitpacks_per_batch)
         * 
         * @note This is a convenience function for memory allocation calculations
         * 
         * @example
         * @code
         * sample_shape<std::uint32_t> shape{1024, 16};
         * auto total = shape.num_bitpacks();  // Returns 16384
         * auto buffer = sycl::malloc_device<std::uint64_t>(total, queue);
         * @endcode
         */
        size_t_ num_bitpacks() const { return batch_size * bitpacks_per_batch; }
    };

    /**
     * @brief Thin wrapper describing one contiguous allocation of Node objects.
     *
     * A node_block owns (or at least refers to) a *single* contiguous allocation
     * that stores <code>count</code> nodes of type <code>node_t</code>.  The core
     * motivation is to make the memory-layout property explicit so that callers
     * can reason about lifetime and de-allocation without having to keep track
     * of individual interior pointers.
     *
     * The wrapper itself is intentionally minimal – just a pointer and a size –
     * because different node types sometimes need extra bookkeeping (e.g. a
     * separate buffer block for `basic_event`).  Specialisations or derived
     * wrappers can extend it with such fields.
     */
    template<typename node_t>
    struct node_block {
        node_t       *data  = nullptr;   ///< first element of the contiguous allocation
        std::size_t   count = 0;         ///< number of valid nodes in <code>data</code>

        [[nodiscard]] node_t       &operator[](std::size_t i)       { return data[i]; }
        [[nodiscard]] const node_t &operator[](std::size_t i) const { return data[i]; }
    };

    /**
     * @brief Specialised node_block for <code>basic_event</code> that also owns the
     * contiguous device buffer block used for bit-packed samples.
     *
     * The <code>buffers</code> pointer is the *sole* allocation for all bit-packed
     * outputs of all events in this block.  Each individual event’s
     * <code>buffer</code> member points somewhere inside this big block at an
     * offset of <code>event_index * bitpacks_per_event</code>.
     */
    template<typename prob_t_, typename bitpack_t_, typename index_t_ = int32_t>
    struct basic_event_block : public node_block<basic_event<prob_t_, bitpack_t_, index_t_>> {
        bitpack_t_  *buffers            = nullptr;   ///< single contiguous device allocation
        std::size_t  bitpacks_per_event = 0;         ///< stride between successive event buffers
    };

    /**
     * @brief Factory that allocates and initialises a contiguous block of
     * <code>basic_event</code> objects *and* their shared output-buffer block.
     *
     * The layout mimics the one previously produced by <code>create_basic_events</code>
     * but returns a richer wrapper that makes ownership explicit and therefore
     * easier to free safely.
     */
    template<typename prob_t_, typename bitpack_t_, typename index_t_ = int32_t>
    [[nodiscard]]
    basic_event_block<prob_t_, bitpack_t_, index_t_>
    create_basic_event_block(const sycl::queue                                  &queue,
                             const std::vector<std::pair<index_t_, prob_t_>>    &indexed_probabilities,
                             const std::size_t                                  num_bitpacks) {

        const std::size_t num_events = indexed_probabilities.size();

        // 1. Allocate host-visible node array (USM shared memory).
        using event_t = basic_event<prob_t_, bitpack_t_, index_t_>;
        event_t *events = sycl::malloc_shared<event_t>(num_events, queue);

        // 2. Allocate device-side bit-packed buffer for *all* events.
        bitpack_t_ *buffer_block = sycl::malloc_device<bitpack_t_>(num_events * num_bitpacks, queue);

        // 3. Populate per-event fields.
        for (std::size_t i = 0; i < num_events; ++i) {
            //events[i].probability = indexed_probabilities[i].second;
            // NEW: pre-compute 32-bit Bernoulli threshold to avoid FP math in kernels
            events[i].probability_threshold = static_cast<std::uint32_t>(static_cast<std::double_t>(indexed_probabilities[i].second) * static_cast<std::double_t>(UINT32_MAX));
            events[i].index       = indexed_probabilities[i].first;
            events[i].buffer      = buffer_block + i * num_bitpacks;
        }

        // 4. Wrap everything in a basic_event_block and return.
        basic_event_block<prob_t_, bitpack_t_, index_t_> blk;
        blk.data               = events;
        blk.count              = num_events;
        blk.buffers            = buffer_block;
        blk.bitpacks_per_event = num_bitpacks;
        return blk;
    }

    /**
     * @brief Releases all device memory owned by a <code>basic_event_block</code>.
     *
     * After the call the block is zeroed out so that double-free / use-after-free
     * errors are easier to spot during debugging.
     */
    template<typename prob_t_, typename bitpack_t_, typename index_t_ = int32_t>
    void destroy_basic_event_block(const sycl::queue                                   &queue,
                                   basic_event_block<prob_t_, bitpack_t_, index_t_>      &blk) {
        if (blk.buffers) {
            sycl::free(blk.buffers, queue);
            blk.buffers = nullptr;
        }
        if (blk.data) {
            sycl::free(blk.data, queue);
            blk.data  = nullptr;
        }
        blk.count = 0;
        blk.bitpacks_per_event = 0;
    }

    // ---------------------------------------------------------------------
    //  Tally block wrappers (one contiguous allocation of tally nodes)
    // ---------------------------------------------------------------------

    /**
     * @brief Thin wrapper describing one contiguous allocation of `tally` nodes.
     *
     * Contrary to `basic_event_block`, a tally block owns only the array of
     * `tally` structs; the bit-packed sample buffers it references are owned by
     * the producer nodes (basic events or gates).  Therefore the wrapper holds
     * no extra fields beyond the base `node_block`.
     */
    template<typename bitpack_t_>
    struct tally_block : public node_block<tally<bitpack_t_>> {
        // no additional state – buffers are external
    };

    /**
     * @brief Allocates and initialises a contiguous array of `tally` objects.
     *
     * Each tally’s `buffer` member is set to the corresponding entry in
     * `source_buffers`, and `num_one_bits` is initialised to 0.
     */
    template<typename bitpack_t_>
    [[nodiscard]]
    tally_block<bitpack_t_>
    create_tally_block(const sycl::queue                    &queue,
                       const std::vector<bitpack_t_ *>      &source_buffers) {
        const std::size_t n = source_buffers.size();
        using tally_t = tally<bitpack_t_>;
        tally_t *tallies = sycl::malloc_shared<tally_t>(n, queue);

        for (std::size_t i = 0; i < n; ++i) {
            tallies[i].buffer       = source_buffers[i];
            tallies[i].num_one_bits = 0;
            tallies[i].mean         = 0.0;
            tallies[i].std_err      = 0.0;
            tallies[i].ci           = {0.0, 0.0, 0.0, 0.0};
        }

        tally_block<bitpack_t_> blk;
        blk.data  = tallies;
        blk.count = n;
        return blk;
    }

    /**
     * @brief Frees the USM allocation owned by a `tally_block`.
     *
     * Does **not** free the sample buffers the individual tally nodes point to;
     * those are owned by the nodes that produced the samples.
     */
    template<typename bitpack_t_>
    void destroy_tally_block(const sycl::queue          &queue,
                             tally_block<bitpack_t_>    &blk) {
        if (blk.data) {
            sycl::free(blk.data, queue);
            blk.data = nullptr;
        }
        blk.count = 0;
    }

    // ---------------------------------------------------------------------
    //  Gate block wrappers (one contiguous allocation of standard `gate`s)
    // ---------------------------------------------------------------------

    /**
     * @brief Thin wrapper describing one contiguous allocation of `gate` nodes.
     *
     * In addition to the base `node_block` data/size pair, a `gate_block` owns
     * a single device allocation that stores all bit-packed output buffers for
     * every gate in the block.  Each individual gate’s `buffer` member points
     * somewhere *inside* this big allocation at an offset of
     * `gate_index * bitpacks_per_gate`.
     */
    template<typename bitpack_t_, typename size_t_>
    struct gate_block : public node_block<gate<bitpack_t_, size_t_>> {
        /* contiguous device block that stores every gate’s computed output */
        bitpack_t_   *buffers           = nullptr;

        /* single USM-shared array containing *all* input buffer pointers
         * for *every* gate in this block.  Each gate’s `inputs` member is
         * simply a slice into this big array.                           */
        bitpack_t_  **all_inputs        = nullptr;
        std::size_t   total_inputs      = 0;        ///< length of all_inputs

        std::size_t   bitpacks_per_gate = 0;        ///< stride between successive gate buffers
    };

    /**
     * @brief Allocates and initialises a contiguous block of `gate` objects and
     * their shared output-buffer block.
     *
     * `inputs_per_gate` follows the same convention used previously in
     * `create_gates`: each entry is a pair whose first element is the *ordered*
     * vector of input buffers and whose second element is the number of *negated*
     * inputs at the tail of that vector.
     */
    template<typename bitpack_t_, typename size_t_>
    [[nodiscard]]
    gate_block<bitpack_t_, size_t_>
    create_gate_block(const sycl::queue                                                      &queue,
                      const std::vector<std::pair<std::vector<bitpack_t_ *>, size_t_>>       &inputs_per_gate,
                      const std::size_t                                                      num_bitpacks) {
        const std::size_t num_gates = inputs_per_gate.size();

        using gate_t = gate<bitpack_t_, size_t_>;

        /* First pass: count total unique pointer slots needed */
        std::size_t total_input_ptrs = 0;
        for (const auto &g : inputs_per_gate) {
            total_input_ptrs += g.first.size();
        }

        // 1) Allocate primary USM-shared structures.
        gate_t      *gates        = sycl::malloc_shared<gate_t>(num_gates, queue);
        bitpack_t_ **all_inputs   = sycl::malloc_shared<bitpack_t_ *>(total_input_ptrs, queue);

        // 2) Allocate single device block for all gate outputs.
        bitpack_t_  *buffer_block = sycl::malloc_device<bitpack_t_>(num_gates * num_bitpacks, queue);

        // 3) Populate gate structs and fill the all_inputs array.
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < num_gates; ++i) {
            const auto &gate_inputs   = inputs_per_gate[i].first;
            const std::size_t n_in    = gate_inputs.size();
            const std::size_t n_neg   = inputs_per_gate[i].second;

            assert(n_neg <= n_in);

            // Slice in all_inputs where this gate’s pointers will live.
            gates[i].inputs                = all_inputs + cursor;
            gates[i].num_inputs            = static_cast<size_t_>(n_in);
            gates[i].negated_inputs_offset = static_cast<size_t_>(n_in - n_neg);
            gates[i].buffer                = buffer_block + i * num_bitpacks;

            // Copy actual buffer pointers.
            std::copy(gate_inputs.begin(), gate_inputs.end(), all_inputs + cursor);
            cursor += n_in;
        }

        // 4) Wrap and return.
        gate_block<bitpack_t_, size_t_> blk;
        blk.data               = gates;
        blk.count              = num_gates;
        blk.buffers            = buffer_block;
        blk.all_inputs         = all_inputs;
        blk.total_inputs       = total_input_ptrs;
        blk.bitpacks_per_gate  = num_bitpacks;
        return blk;
    }

    /**
     * @brief Releases all device memory owned by a `gate_block`.
     *
     * Frees, in order:
     *   1. Each per-gate `inputs` array
     *   2. The shared output-buffer block
     *   3. The contiguous array of `gate` structures
     */
    template<typename bitpack_t_, typename size_t_>
    void destroy_gate_block(const sycl::queue                     &queue,
                            gate_block<bitpack_t_, size_t_>       &blk) {
        if (blk.all_inputs) {
            sycl::free(blk.all_inputs, queue);
            blk.all_inputs = nullptr;
        }

        if (blk.buffers) {
            sycl::free(blk.buffers, queue);
            blk.buffers = nullptr;
        }

        if (blk.data) {
            sycl::free(blk.data, queue);
            blk.data = nullptr;
        }

        blk.count = 0;
        blk.total_inputs = 0;
        blk.bitpacks_per_gate = 0;
    }

}// namespace scram::canopy::event
