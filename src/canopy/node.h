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

#include <sycl/sycl.hpp>

namespace scram::canopy {

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
        /// @brief Failure probability of this basic event (range: [0.0, 1.0])
        prob_t_ probability;
        
        /// @brief Unique identifier for this event within the computation graph
        index_t_ index;
    };

    /**
     * @struct tally_event
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
    struct tally_event : node<bitpack_t_> {
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
    tally_event<bitpack_t_> *create_tally_events(const sycl::queue &queue, const std::vector<bitpack_t_ *> &buffers, const std::vector<std::size_t> &initial_values) {
        const auto num_tallies = buffers.size();
        tally_event<bitpack_t_> *tallies = sycl::malloc_shared<tally_event<bitpack_t_>>(num_tallies, queue);
        for (auto i = 0; i < num_tallies; ++i) {
            tallies[i].buffer = buffers[i];
            tallies[i].num_one_bits = initial_values[i];
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
    void destroy_tally_event(const sycl::queue queue, tally_event<bitpack_t_> *event) {
        sycl::free(event, queue);
    }

    /**
     * @brief Factory function for creating device-allocated basic events
     * 
     * @details Creates an array of basic_event structures with optimized memory layout
     * for parallel processing. Allocates both the event structures and their associated
     * buffers in contiguous memory blocks for improved cache performance and reduced
     * memory fragmentation.
     * 
     * @tparam prob_t_ Floating-point type for probability values
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @param queue SYCL queue for memory allocation and device context
     * @param probabilities Vector of failure probabilities for each basic event
     * @param indices Vector of unique indices for each basic event
     * @param num_bitpacks Number of bitpacks to allocate per event buffer
     * 
     * @return Pointer to array of allocated basic events
     * 
     * @throws std::bad_alloc if device memory allocation fails
     * 
     * @note Buffers are allocated as a single contiguous block for efficiency
     * @note Caller is responsible for calling destroy_basic_events() for cleanup
     * 
     * @example
     * @code
     * std::vector<double> probs = {0.01, 0.05, 0.001};
     * std::vector<int32_t> indices = {1, 2, 3};
     * auto events = create_basic_events<double, std::uint64_t>(queue, probs, indices, 1024);
     * // Use events...
     * destroy_basic_events(queue, events, probs.size());
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_>
    basic_event<prob_t_, bitpack_t_> *create_basic_events(const sycl::queue &queue, const std::vector<prob_t_> &probabilities, const std::vector<int32_t> &indices, const std::size_t num_bitpacks) {
        const auto num_events = probabilities.size();
        // allocate the basic event objects in a contiguous block
        basic_event<prob_t_, bitpack_t_> *basic_events = sycl::malloc_shared<basic_event<prob_t_, bitpack_t_>>(num_events, queue);
        bitpack_t_* buffers = sycl::malloc_device<bitpack_t_>(num_events * num_bitpacks, queue);
        // allocate basic event buffers separately
        for (auto i = 0; i < num_events; ++i) {
            basic_events[i].probability = probabilities[i];
            basic_events[i].index = indices[i];
            //basic_events[i].buffer = sycl::malloc_device<bitpack_t_>(num_bitpacks, queue);
            basic_events[i].buffer = buffers + i * num_bitpacks;
            //LOG(DEBUG5) <<"building basic event "<<basic_events[i].index<<" with probability "<<basic_events[i].probability;
        }
        return basic_events;
    }

    /**
     * @brief Destroys a single basic event and frees associated memory
     * 
     * @details Properly releases device memory allocated for a basic event structure
     * and its associated buffer. This function handles both the event structure and
     * its data buffer.
     * 
     * @tparam prob_t_ Floating-point type for probability values
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @param queue SYCL queue for memory deallocation
     * @param event Pointer to basic event to destroy
     * 
     * @note This function frees both the event structure and its buffer
     * @note Should not be used with events created by create_basic_events() (use destroy_basic_events() instead)
     * 
     * @example
     * @code
     * // For individually allocated events
     * auto event = sycl::malloc_shared<basic_event<double, std::uint64_t>>(1, queue);
     * event->buffer = sycl::malloc_device<std::uint64_t>(1024, queue);
     * // Use event...
     * destroy_basic_event(queue, event);
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_>
    void destroy_basic_event(const sycl::queue &queue, basic_event<prob_t_, bitpack_t_> *event) {
        sycl::free(event->buffer, queue);
        sycl::free(event, queue);
    }

    /**
     * @brief Destroys an array of basic events created by create_basic_events()
     * 
     * @details Properly releases device memory allocated for an array of basic events
     * and their associated buffers. This function handles the contiguous memory layout
     * created by create_basic_events().
     * 
     * @tparam prob_t_ Floating-point type for probability values
     * @tparam bitpack_t_ Integer type for bit-packed data storage
     * 
     * @param queue SYCL queue for memory deallocation
     * @param events Pointer to array of basic events to destroy
     * @param count Number of events in the array
     * 
     * @note This function assumes events were created with create_basic_events()
     * @note Handles the contiguous buffer allocation optimization
     * 
     * @example
     * @code
     * auto events = create_basic_events<double, std::uint64_t>(queue, probs, indices, 1024);
     * // Use events...
     * destroy_basic_events(queue, events, probs.size());
     * @endcode
     */
    template<typename prob_t_, typename bitpack_t_>
    void destroy_basic_events(const sycl::queue &queue, basic_event<prob_t_, bitpack_t_> *events, const std::size_t count) {
        for (auto i = 0; i < count; ++i) {
            sycl::free(events[i]->buffer, queue);
        }
        sycl::free(events, queue);
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
        // allocate all the gate objects contiguously
        gate<bitpack_t_, size_t_> *gates = sycl::malloc_shared<gate<bitpack_t_, size_t_>>(num_gates, queue);
        bitpack_t_* buffers = sycl::malloc_device<bitpack_t_>(num_gates * num_bitpacks, queue);

        for (auto i = 0; i < num_gates; ++i) {
            const auto gate_input_buffers = inputs_per_gate[i].first;
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
}// namespace scram::canopy
