/**
 * @file layer_manager.h
 * @brief SYCL-based layer management for the compute graph
 * @author Arjun Earthperson
 * @date 2025
 * 
 * @details This file provides a templated layer manager that organizes graph
 * analysis computations into layers for efficient parallel execution on SYCL devices.
 * The layer manager performs topological sorting, kernel building, and execution
 * scheduling for probabilistic directed acyclic graphs (PDAGs).
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
#include "mc/queue/queueable.h"
#include "mc/queue/sample_shaper.h"
#include "mc/stats/tally_node_map.h"

#include "pdag.h"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
//  Forward declarations to break circular header dependencies.
// ---------------------------------------------------------------------------

// Forward-declare layer_manager itself so that subsequent declarations can
// reference it before its full definition appears later in this file.
namespace scram::mc::queue {
    template <typename bitpack_t_, typename prob_t_, typename size_t_>
    class layer_manager;
}

// Forward declaration of the CSV logging helper.
namespace scram::log::layers {
    template <typename bt_, typename pt_, typename st_>
    std::vector<std::pair<std::string, std::string>>
    csv_pairs(const scram::mc::queue::layer_manager<bt_, pt_, st_>&);
}

namespace scram::mc::queue {

/// AdaptiveCpp's default async handler prints the error list and std::terminate()s.
/// Do not throw from the queue handler: `wait_and_throw` / `~queue` both invoke
/// it, so a throw from `wait_and_throw` plus a second throw from `~queue` during
/// unwind calls std::terminate. Record the message and rethrow on `wait()`.
inline thread_local std::string g_sycl_async_error;

inline std::string &sycl_async_error_slot() {
    return g_sycl_async_error;
}

inline sycl::queue make_throwing_sycl_queue() {
    return sycl::queue{[](sycl::exception_list errors) {
        if (errors.empty()) {
            return;
        }
        std::string msg;
        for (std::exception_ptr const &err : errors) {
            try {
                if (err) {
                    std::rethrow_exception(err);
                }
            } catch (const std::exception &e) {
                if (!msg.empty()) {
                    msg += "; ";
                }
                msg += e.what();
            } catch (...) {
                if (!msg.empty()) {
                    msg += "; ";
                }
                msg += "unknown SYCL asynchronous error";
            }
        }
        auto &slot = sycl_async_error_slot();
        if (!slot.empty()) {
            slot += "; ";
        }
        slot += msg.empty() ? "SYCL asynchronous error" : msg;
    }};
}

inline void wait_sycl_queue(sycl::queue &queue) {
    queue.wait();
    queue.throw_asynchronous();
    auto &slot = sycl_async_error_slot();
    if (slot.empty()) {
        return;
    }
    std::string msg = std::move(slot);
    slot.clear();
    throw std::runtime_error(msg);
}

struct range3_description {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;

    [[nodiscard]] std::size_t product() const {
        return x * y * z;
    }
};

struct device_description {
    std::string type;
    std::string name;
    std::uint32_t vendor_id = 0;
    std::string vendor;
    std::string driver_version;
    std::string profile;
    std::string version;
    std::string opencl_c_version;
    std::vector<int> aspects;
    std::vector<std::string> extensions;
    std::uint32_t max_compute_units = 0;
    std::uint32_t max_clock_frequency = 0;
    std::uint32_t max_work_item_dimensions = 0;
    range3_description max_work_item_sizes_1d{};
    range3_description max_work_item_sizes_2d{};
    range3_description max_work_item_sizes_3d{};
    bool work_item_independent_forward_progress = false;
    std::size_t max_work_group_size = 0;
    std::uint32_t max_num_sub_groups = 0;
    bool sub_group_independent_forward_progress = false;
    std::vector<std::size_t> sub_group_sizes;
    std::uint32_t preferred_vector_width_char = 0;
    std::uint64_t max_mem_alloc_size = 0;
    std::uint64_t global_mem_size = 0;
    std::uint64_t global_mem_cache_size = 0;
    std::uint32_t global_mem_cache_line_size = 0;
    std::string global_mem_cache_type;
    std::string local_mem_type;
    std::uint64_t local_mem_size = 0;
};

struct kernel_dependency_edge {
    std::size_t source_partition = 0;
    std::size_t target_partition = 0;
    std::string reason = "buffer_dependency";
};

struct kernel_partition_description {
    std::size_t id = 0;
    std::size_t layer = 0;
    std::string kind;
    std::string connective;
    std::vector<std::int32_t> node_indices;
    std::vector<std::size_t> dependency_partition_ids;
    range3_description logical_range{};
    range3_description local_range{};
    range3_description global_range{};
    std::size_t node_count = 0;
    std::size_t work_group_count = 0;
    std::size_t work_item_count = 0;
    std::size_t logical_work_item_count = 0;
    std::size_t overprovisioned_work_items = 0;
    std::size_t bitpacks_per_node = 0;
    std::size_t bits_per_node = 0;
    std::size_t bytes_per_node = 0;
    std::size_t resident_bytes = 0;
    std::size_t min_fan_in = 0;
    std::size_t max_fan_in = 0;
    std::size_t total_fan_in = 0;
    std::size_t positive_input_count = 0;
    std::size_t negated_input_count = 0;
};

struct execution_node_description {
    std::int32_t index = 0;
    std::size_t layer = 0;
    std::size_t partition = 0;
    std::string kind;
    std::string connective;
    std::size_t fan_in = 0;
    std::size_t positive_input_count = 0;
    std::size_t negated_input_count = 0;
    std::size_t atleast_min = 0;
    bool module = false;
};

/**
 * @class layer_manager
 * @brief Manages layered execution of pdag computations on SYCL devices
 * 
 * @details The layer_manager class orchestrates the execution of pdag
 * computations by organizing nodes into layers based on topological ordering. It builds
 * and manages SYCL kernels for parallel execution of basic events and gates, handles
 * memory allocation, and provides tally computation for probabilistic analysis.
 * 
 * The class performs several key operations:
 * - Topological sorting of PDAG nodes into execution layers
 * - SYCL kernel generation for each layer of computations
 * - Memory management for device-side data structures
 * - Execution scheduling and synchronization
 * - Tally computation with confidence intervals
 * 
 * @tparam bitpack_t_ Integer type for bit packing operations (default: std::uint64_t)
 * @tparam prob_t_ Floating-point type for probability calculations (default: std::double_t)  
 * @tparam size_t_ Integer type for size and indexing (default: std::uint32_t)
 * 
 * @example Basic usage:
 * @code
 * // Create a layer manager for a pdag
 * scram::mc::queue::layer_manager<> manager(pdag, 1024, 16);
 * 
 * // Submit all computations and wait for completion
 * manager.submit_all().wait_and_throw();
 * 
 * // Get tally results for a specific event
 * auto tally_result = manager.tally(event_index, 1000);
 * std::cout << "Probability: " << tally_result.mean << " ± " << tally_result.std_err << std::endl;
 * @endcode
 * 
 * @note This class requires SYCL-compatible hardware and runtime
 * @warning The class manages GPU memory and should be used with proper exception handling
 * 
 * @since Version 1.0
 */
template <typename bitpack_t_, typename prob_t_, typename size_t_>
class layer_manager {

    /// @brief Index type for node identification
    using index_t_ = std::int32_t;

    /// @brief SYCL queue for device operations and kernel execution
    sycl::queue queue_{make_throwing_sycl_queue()};
    
    /// @brief Sample shape configuration defining batch size and bitpack dimensions
    event::sample_shape<size_t_> sample_shape_;

    sample_shaper<bitpack_t_> sample_shaper_{};

    /// @brief Vector containing all PDAG nodes in topological order
    std::vector<std::shared_ptr<core::Node>> pdag_nodes_;
    
    /// @brief Map from node index to node pointer for fast lookup
    std::unordered_map<index_t_, std::shared_ptr<core::Node>> pdag_nodes_by_index_;
    
    /// @brief Two-dimensional vector organizing nodes by execution layer
    std::vector<std::vector<std::shared_ptr<core::Node>>> pdag_nodes_by_layer_;

    /// @brief Map from node index to queueable computation object
    std::unordered_map<index_t_, std::shared_ptr<queueable_base>> queueables_by_index_;
    
    /// @brief Vector of all queueable objects in execution order
    std::vector<std::shared_ptr<queueable_base>> queueables_;

    std::unordered_map<const queueable_base *, std::size_t> partition_ids_by_queueable_;
    std::vector<kernel_partition_description> execution_partitions_;
    std::vector<kernel_dependency_edge> execution_dependency_edges_;

    /// @brief Map from node index to allocated device-side tally counters
    std::unordered_map<index_t_, event::tally_device<bitpack_t_> *> allocated_tally_events_by_index_;
    
    /// @brief Map from node index to allocated device-side basic event structures
    std::unordered_map<index_t_, event::basic_event<prob_t_, bitpack_t_> *> allocated_basic_events_by_index_;
    
    /// @brief Map from node index to allocated device-side gate structures
    std::unordered_map<index_t_, event::gate<bitpack_t_, size_t_> *> allocated_gates_by_index_;

    /// @brief Map from node index to allocated blocks
    std::vector<event::basic_event_block<prob_t_, bitpack_t_, index_t_>> device_basic_event_blocks_;

    /// @brief Map from node index to allocated blocks
    std::vector<event::gate_block<bitpack_t_, size_t_>> device_gate_blocks_;

    /// @brief Map from node index to allocated blocks
    std::vector<event::atleast_gate_block<bitpack_t_, size_t_>> device_atleast_gate_blocks_;

    /// @brief Map from node index to allocated blocks
    std::vector<event::tally_block<bitpack_t_>> device_tally_blocks_;

    /**
     * @brief Recursively gathers all nodes from a gate hierarchy
     * 
     * @details Performs a depth-first traversal of the gate hierarchy starting from
     * the given gate, collecting all reachable nodes (both gates and variables) while
     * avoiding duplicates through the visit mechanism.
     * 
     * @param gate Root gate to start traversal from
     * @param nodes [out] Vector to store collected nodes
     * @param nodes_by_index [out] Map from index to node for fast lookup
     * 
     * @throws std::runtime_error if duplicate indices are found
     * 
     * @note This is a static method that modifies the visit state of nodes
     * @warning Assumes nodes have been cleared of visit flags before calling
     * 
     * @example
     * @code
     * std::vector<std::shared_ptr<core::Node>> nodes;
     * std::unordered_map<std::int32_t, std::shared_ptr<core::Node>> nodes_by_index;
     * layer_manager::gather_all_nodes(root_gate, nodes, nodes_by_index);
     * @endcode
     */
    static void gather_all_nodes(const std::shared_ptr<core::Gate> &gate,
                                 std::vector<std::shared_ptr<core::Node>> &nodes,
                                 std::unordered_map<index_t_, std::shared_ptr<core::Node>> &nodes_by_index);

    /**
     * @brief Performs layered topological sorting of PDAG nodes
     * 
     * @details Organizes nodes into execution layers based on their topological order,
     * ensuring that all dependencies of a node are computed before the node itself.
     * Within each layer, variables are ordered before gates, and gates are ordered by type.
     * 
     * @param pdag Pointer to the probabilistic directed acyclic graph
     * @param nodes [out] Vector to store all nodes in topological order
     * @param nodes_by_index [out] Map from index to node pointer
     * @param nodes_by_layer [out] Two-dimensional vector organizing nodes by layer
     * 
     * @note This method clears visit flags and sorts nodes within each layer
     * @warning The PDAG must be acyclic for proper topological ordering
     * 
     * @example
     * @code
     * layer_manager::layered_toposort(pdag, nodes, nodes_by_index, nodes_by_layer);
     * // Now nodes_by_layer[0] contains all nodes in layer 0, etc.
     * @endcode
     */
    static void layered_toposort(core::Pdag *pdag, std::vector<std::shared_ptr<core::Node>> &nodes,
                                 std::unordered_map<index_t_, std::shared_ptr<core::Node>> &nodes_by_index,
                                 std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer);

    /**
     * @brief Partitions layer nodes into variables and gates by type
     * 
     * @details Separates nodes in a single layer into variables (basic events) and
     * gates grouped by their connective type (AND, OR, etc.). This organization
     * enables efficient kernel generation for homogeneous operations.
     * 
     * @param layer_nodes Input vector of nodes from a single layer
     * @param out_variables [out] Vector to store extracted variables
     * @param out_gates_by_type [out] Map from connective type to gates of that type
     * 
     * @note Variables represent basic events with probability distributions
     * @note Gates represent logical operations (AND, OR, K-out-of-N, etc.)
     * 
     * @example
     * @code
     * std::vector<std::shared_ptr<core::Variable>> variables;
     * std::unordered_map<core::Connective, std::vector<std::shared_ptr<core::Gate>>> gates_by_type;
     * layer_manager::gather_layer_nodes(layer_nodes, variables, gates_by_type);
     * @endcode
     */
    static void gather_layer_nodes(
        const std::vector<std::shared_ptr<core::Node>> &layer_nodes,
        std::vector<std::shared_ptr<core::Variable>> &out_variables,
        std::unordered_map<core::Connective, std::vector<std::shared_ptr<core::Gate>>> &out_gates_by_type);

    /**
     * @brief Builds SYCL kernels for all nodes in a single layer
     * 
     * @details Creates optimized SYCL kernels for parallel execution of computations
     * within a layer. Builds separate kernels for variables (basic events) and for
     * each type of gate operation, enabling efficient batched execution.
     * 
     * @param layer_nodes Vector of nodes to process in this layer
     * 
     * @note This method updates the queueables_ and queueables_by_index_ containers
     * @note Kernels are built but not yet submitted for execution
     * 
     * @example
     * @code
     * // Process layer 0 nodes
     * build_kernels_for_layer(nodes_by_layer[0]);
     * @endcode
     */
    void build_kernels_for_layer(const std::vector<std::shared_ptr<core::Node>> &layer_nodes, std::size_t layer_index);

    /**
     * @brief Maps nodes by layer and builds corresponding kernels
     * 
     * @details Iterates through all layers of nodes, building kernels for each layer
     * and setting up tally computations for the final layer. This method orchestrates
     * the complete kernel generation process for the entire computation graph.
     * 
     * @param nodes_by_layer Two-dimensional vector of nodes organized by layer
     * 
     * @note The final layer receives additional tally kernel generation
     * @note This method populates all kernel-related data structures
     * 
     * @example
     * @code
     * map_nodes_by_layer(pdag_nodes_by_layer_);
     * // All kernels are now built and ready for execution
     * @endcode
     */
    void map_nodes_by_layer(const std::vector<std::vector<std::shared_ptr<core::Node>>> &nodes_by_layer);

    event::tally<bitpack_t_> fetch_tally_for_event_with_index(index_t_ evt_idx);

  public:
    struct kernel_dag_node {
        index_t_ index = 0;
        std::size_t layer = 0;
        std::size_t partition = 0;
        std::string kind;
        std::string connective;
    };

    struct kernel_dag_partition {
        std::size_t id = 0;
        std::size_t layer = 0;
        std::string kind;
        std::string connective;
        std::vector<index_t_> node_indices;
    };

    struct kernel_dag_description {
        std::vector<kernel_dag_node> nodes;
        std::vector<kernel_dag_partition> partitions;
        std::size_t batch_size = 0;
        std::size_t bitpacks_per_batch = 0;
        std::size_t total_iterations = 0;
    };

    struct execution_plan_description {
        std::string schema = "mcscram.execution_plan.v1";
        std::string bitpack_type;
        std::size_t bits_per_bitpack = 0;
        std::size_t node_count = 0;
        device_description device;
        typename sample_shaper<bitpack_t_>::description sample_shaper;
        std::vector<kernel_partition_description> partitions;
        std::vector<kernel_dependency_edge> dependency_edges;
        std::vector<execution_node_description> nodes;
    };

    /**
     * @brief Constructs a layer manager for the given PDAG
     * 
     * @details Initializes the layer manager with the specified computation parameters,
     * performs topological sorting of the PDAG, and builds all necessary kernels for
     * parallel execution. The sample shape is automatically rounded to device-optimal
     * dimensions.
     * 
     * @throws std::runtime_error if PDAG processing fails
     * @throws std::runtime_error if kernel building fails
     * 
     * @note The constructor performs all setup operations synchronously
     * @note Sample shape is optimized for the target SYCL device
     *
     */
    layer_manager(core::Pdag *pdag, size_t_ num_trials, const stats::TallyNodeMap &to_tally, std::size_t requested_trials_per_pass);

    /**
     * @brief Submits all queued computations to the SYCL device
     * 
     * @details Submits all built kernels to the SYCL queue for execution. Returns
     * the SYCL queue object that can be used for synchronization and error checking.
     * This method does not block; use wait_and_throw() on the returned queue for
     * synchronous execution.
     * 
     * @return SYCL queue object for synchronization and error handling
     * 
     * @note This method is non-blocking; kernels execute asynchronously
     * @note Call wait_and_throw() on the returned queue to ensure completion
     * 
     * @example
     * @code
     * // Submit and wait for completion
     * manager.submit_all().wait_and_throw();
     * 
     * // Submit asynchronously
     * auto queue = manager.single_pass();
     * // Do other work...
     * queue.wait_and_throw();
     * @endcode
     */
    sycl::queue single_pass();


    sycl::queue pass(size_t count = 1);

    /**
     * @brief Computes tally statistics for a specific event
     * 
     * @details Performs Monte Carlo sampling to compute probability statistics for
     * the specified event, including mean probability, standard error, and confidence
     * intervals. The computation is repeated for the specified count to improve accuracy.
     * 
     * @param evt_idx Index of the event to compute tally for
     * @param count Number of tally computations to perform
     * 
     * @return tally_event structure containing computed statistics
     * @retval tally_event::mean Mean probability estimate
     * @retval tally_event::std_err Standard error of the estimate
     * @retval tally_event::ci[0] 5th percentile confidence bound
     * @retval tally_event::ci[1] 95th percentile confidence bound
     * @retval tally_event::num_one_bits Number of positive outcomes
     * 
     * @note This method blocks until all computations complete
     * @note Returns empty tally_event if event index is not found
     * 
     * @example
     * @code
     * // Compute tally for event 42
     * auto result = manager.tally(42);
     * 
     * if (result.mean > 0.0) {
     *     std::cout << "Event 42 probability: " << result.mean 
     *               << " ± " << result.std_err << std::endl;
     *     std::cout << "95% CI: [" << result.ci[0] << ", " << result.ci[1] << "]" << std::endl;
     * }
     * @endcode
     */
    event::tally<bitpack_t_> single_pass_and_tally(index_t_ evt_idx);

    stats::TallyNodeMap &pass_wait_collect(stats::TallyNodeMap &stats, std::size_t total_passes = 1, std::size_t passes_between_waits = 0);

    stats::TallyNodeMap &collect_tallies(stats::TallyNodeMap &stats);

    [[nodiscard]] std::size_t node_count() const;

    [[nodiscard]] kernel_dag_description describe_kernel_dag() const;

    [[nodiscard]] execution_plan_description describe_execution_plan() const;

    /**
     * @brief Retrieves the original MEF event (BasicEvent) corresponding to a PDAG node index.
     *
     * Given a PDAG node index, this function resolves the underlying MEF event that was
     * used to construct the node.  At the moment only Variable nodes map directly back
     * to MEF::BasicEvent instances.  If the index is unknown or the mapping cannot be
     * resolved an exception is thrown.
     */
    [[nodiscard]] const scram::mef::Event* get_mef_event(index_t_ event_id) const;

    /**
     * @brief Accessor for the internal sample shaper configuration
     *
     * @details Returns a const reference to the sample_shaper instance that
     * controls how the requested number of trials is split across device
     * iterations.  External convergence-management utilities can rely on
     * this accessor to obtain TOTAL_ITERATIONS as well as the chosen
     * SAMPLE_SHAPE.
     */
    [[nodiscard]] inline const sample_shaper<bitpack_t_> &shaper() const  { return sample_shaper_; }
    [[nodiscard]] sycl::queue &queue() { return queue_; }

    // ------------------------------------------------------------------
    //  Logging helpers – grant access to scram::log::layers::csv_pairs
    // ------------------------------------------------------------------
    template <typename bt_, typename pt_, typename st_>
    friend std::vector<std::pair<std::string, std::string>>
    scram::log::layers::csv_pairs(const scram::mc::queue::layer_manager<bt_, pt_, st_>&);
    /**
     * @brief Destructor that cleans up allocated device memory
     *
     * @details Properly releases all allocated device-side memory for basic events,
     * gates, and tally events. Ensures no memory leaks occur when the layer manager
     * is destroyed.
     *
     * @note All device memory is automatically freed
     * @note SYCL queue operations are completed before destruction
     *
     * @example
     * @code
     * {
     *     layer_manager<> manager(pdag, 1024, 16);
     *     // Use manager...
     * } // Destructor automatically cleans up all device memory
     * @endcode
     */
    ~layer_manager();
};
} // namespace scram::mc::queue

#include "layer_manager.tpp"
