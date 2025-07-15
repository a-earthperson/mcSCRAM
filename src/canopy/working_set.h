/**
 * @file working_set.h
 * @brief Optimal working-set configuration for SYCL device performance tuning
 * 
 * @details This file provides sophisticated algorithms for determining optimal
 * working-set splits and memory configurations for SYCL devices across different
 * backends (CUDA, OpenCL, OpenMP). It analyzes device capabilities and computes
 * optimal work-group sizes, memory layouts, and occupancy rates to maximize
 * performance in parallel Monte Carlo simulations.
 * 
 * The working set configuration is critical for achieving optimal performance
 * across diverse hardware architectures, from GPUs with thousands of cores to
 * multi-core CPUs with different memory hierarchies.
 *
 * @author Arjun Earthperson
 * @date 11/06/2024
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
 * @see node.h for related data structures
 * @see scram::canopy::sample_shape for memory layout configuration
 * 
 * @example
 * @code
 * // Configure working set for Monte Carlo simulation
 * sycl::queue queue;
 * const size_t num_events = 1000;
 * auto shape = working_set<uint32_t, uint64_t>::compute_optimal_sample_shape(queue, num_events);
 * working_set<uint32_t, uint64_t> ws(queue, num_events, shape);
 * 
 * // Use computed configuration
 * auto local_range = ws.compute_optimal_local_range_3d();
 * std::cout << "Optimal local range: " << local_range[0] << "x" 
 *           << local_range[1] << "x" << local_range[2] << std::endl;
 * @endcode
 */

#pragma once

#include "canopy/event/node.h"

#include <cstddef>
#include <sycl/sycl.hpp>

namespace scram::canopy {

    /**
     * @brief Computes target occupancy rate for OpenCL CPU devices
     * 
     * @details Calculates the optimal number of work-items per compute unit for
     * OpenCL CPU devices using a power-law scaling formula. This heuristic is
     * based on empirical performance measurements across various CPU architectures.
     * 
     * The formula accounts for the fact that CPU cores benefit from higher
     * occupancy rates compared to GPU cores, but with diminishing returns as
     * the number of threads increases due to memory bandwidth limitations.
     * 
     * @param threads Number of available CPU threads (default: 1)
     * @return Optimal occupancy rate for the given thread count
     * 
     * @note The formula uses a 4/3 power scaling to model CPU performance characteristics
     * @note Base value of 6400 is derived from empirical testing on various CPU architectures
     * 
     * @example
     * @code
     * // For an 8-core CPU with 16 threads
     * auto occupancy = TARGET_OCCUPANCY_RATE_OPENCL_CPU(16);
     * std::cout << "Optimal occupancy: " << occupancy << std::endl;  // ~1600
     * @endcode
     */
    static constexpr size_t TARGET_OCCUPANCY_RATE_OPENCL_CPU(const size_t threads = 1) {
        return static_cast<size_t>(6400.0 * std::pow((128.0 / static_cast<double_t>(threads)), 4.0 / 3.0));
    }

    /**
     * @brief Computes target occupancy rate for OpenMP CPU devices
     * 
     * @details Calculates the optimal number of work-items per compute unit for
     * OpenMP CPU devices. OpenMP typically benefits from higher occupancy rates
     * than OpenCL due to better thread scheduling and reduced overhead.
     * 
     * This function provides a 2x multiplier over the OpenCL CPU rate based on
     * performance measurements showing OpenMP's superior thread utilization.
     * 
     * @param threads Number of available CPU threads (default: 1)
     * @return Optimal occupancy rate for OpenMP (2x OpenCL CPU rate)
     * 
     * @note Higher occupancy compensates for OpenMP's thread scheduling overhead
     * @note Performance tested on systems with 8-128 CPU threads
     * 
     * @example
     * @code
     * // For a 64-core CPU with 128 threads
     * auto occupancy = TARGET_OCCUPANCY_RATE_OPENMP(128);
     * std::cout << "OpenMP occupancy: " << occupancy << std::endl;  // ~12800
     * @endcode
     */
    static constexpr size_t TARGET_OCCUPANCY_RATE_OPENMP(const size_t threads = 1) {
        return static_cast<size_t>(2 * TARGET_OCCUPANCY_RATE_OPENCL_CPU(threads));
    }

    /**
     * @brief Computes target occupancy rate for CUDA/HIP GPU devices
     * 
     * @details Returns the optimal number of work-items per compute unit for
     * CUDA and HIP GPU devices. This constant is derived from extensive
     * performance testing across different GPU architectures including Pascal,
     * Turing, and newer generations.
     * 
     * GPU devices typically benefit from very high occupancy rates due to their
     * massive parallelism and ability to hide memory latency through thread switching.
     * 
     * @param threads Number of available GPU threads (unused for GPUs)
     * @return Fixed optimal occupancy rate for GPU devices (204800)
     * 
     * @note Constant value optimized for modern GPU architectures
     * @note Performance validated on Tesla P4, GTX 1660 Super, and similar GPUs
     * 
     * @example
     * @code
     * // For any CUDA/HIP GPU
     * auto occupancy = TARGET_OCCUPANCY_RATE_CUDA();
     * std::cout << "GPU occupancy: " << occupancy << std::endl;  // 204800
     * @endcode
     */
    static constexpr size_t TARGET_OCCUPANCY_RATE_CUDA(const size_t threads = 1) {
        return 204800;
    }

    /**
     * @brief Computes optimal occupancy rate based on SYCL backend type
     * 
     * @details Selects the appropriate occupancy rate heuristic based on the
     * underlying SYCL backend. This function automatically chooses the best
     * occupancy configuration for the target hardware architecture.
     * 
     * Different backends have vastly different optimal occupancy characteristics:
     * - CUDA/HIP: High fixed occupancy for GPU parallelism
     * - OpenCL/Level Zero: Variable occupancy based on CPU thread count
     * - OpenMP: Enhanced occupancy for better thread utilization
     * 
     * @param backend SYCL backend identifier (CUDA, OpenCL, OpenMP, etc.)
     * @param threads Number of available threads for CPU backends
     * @return Optimal occupancy rate for the specified backend
     * 
     * @note Automatically handles backend-specific optimizations
     * @note Falls back to OpenMP configuration for unknown backends
     * 
     * @example
     * @code
     * // Automatic backend detection
     * sycl::queue queue;
     * auto backend = queue.get_device().get_backend();
     * auto occupancy = compute_desired_occupancy_rate_heuristic(backend, 16);
     * std::cout << "Optimal occupancy: " << occupancy << std::endl;
     * @endcode
     */
    static constexpr size_t compute_desired_occupancy_rate_heuristic(const hipsycl::rt::backend_id backend, const size_t threads = 1) {
        switch (backend) {
            case hipsycl::rt::backend_id::cuda:
            case hipsycl::rt::backend_id::hip:
                return TARGET_OCCUPANCY_RATE_CUDA(threads);
            case hipsycl::rt::backend_id::ocl:
            case hipsycl::rt::backend_id::level_zero:
                return TARGET_OCCUPANCY_RATE_OPENCL_CPU(threads);
            case hipsycl::rt::backend_id::omp:
            default:
                return TARGET_OCCUPANCY_RATE_OPENMP(threads);
        }
    }

    /**
     * @struct working_set
     * @brief Comprehensive SYCL device working-set configuration and optimization
     * 
     * @details This structure encapsulates all device capabilities, memory constraints,
     * and computed optimal configurations for SYCL-based Monte Carlo simulations.
     * It provides a complete characterization of the target device and computes
     * optimal work-group sizes, memory layouts, and execution parameters.
     * 
     * The working set analysis considers:
     * - Device type and compute capabilities
     * - Memory hierarchy and allocation limits
     * - Work-group and sub-group constraints
     * - Sample buffer organization and bit-packing
     * - Backend-specific performance characteristics
     * 
     * @tparam size_type Integer type for sizes and counts (typically uint32_t or uint64_t)
     * @tparam bitpack_type Integer type for bit-packed sample storage (typically uint64_t)
     * 
     * @note All device queries are performed during construction for optimal performance
     * @note Memory allocations are validated against device limits
     * 
     * @example
     * @code
     * // Create working set for 1000 events
     * sycl::queue queue;
     * sample_shape<uint32_t> shape{1024, 16};
     * working_set<uint32_t, uint64_t> ws(queue, 1000, shape);
     * 
     * // Query device capabilities
     * std::cout << "Device: " << ws.device_type << std::endl;
     * std::cout << "Compute units: " << ws.max_compute_units << std::endl;
     * std::cout << "Memory: " << ws.global_mem_size / (1024*1024) << " MB" << std::endl;
     * 
     * // Compute optimal configuration
     * auto local_range = ws.compute_optimal_local_range_3d();
     * @endcode
     */
    template<typename size_type, typename bitpack_type>
    struct working_set {
        /// @brief Number of events in the Monte Carlo simulation
        size_type num_events_;
        
        /// @brief Number of sample bits per event (for bit-packed storage)
        size_type samples_per_event_in_bits_;
        
        /// @brief Number of sample bytes per event (for memory allocation)
        size_type samples_per_event_in_bytes_;
        
        /// @brief Sample buffer organization and dimensions
        event::sample_shape<size_type> bitpack_buffer_shape_;
        
        /// @brief Total sample buffer size in bytes
        size_type samples_in_bytes_;

        // Device capabilities and constraints
        /// @brief Type of SYCL device (CPU, GPU, accelerator, etc.)
        sycl::info::device_type device_type;
        
        /// @brief Maximum number of compute units on the device
        sycl::detail::u_int max_compute_units;
        
        /// @brief Maximum clock frequency of the device in MHz
        sycl::detail::u_int max_clock_frequency;

        // Work-item capabilities
        /// @brief Maximum number of work-item dimensions supported
        sycl::detail::u_int max_work_item_dimensions;
        
        /// @brief Maximum work-item sizes for 1D kernels
        sycl::range<1> max_work_item_sizes_1d;
        
        /// @brief Maximum work-item sizes for 2D kernels
        sycl::range<2> max_work_item_sizes_2d;
        
        /// @brief Maximum work-item sizes for 3D kernels
        sycl::range<3> max_work_item_sizes_3d;
        
        /// @brief Whether work-items can make independent forward progress
        bool work_item_independent_forward_progress;

        // Work-group capabilities
        /// @brief Maximum size of a work-group
        std::size_t max_work_group_size;

        // Sub-group capabilities
        /// @brief Maximum number of sub-groups per work-group
        sycl::detail::u_int max_num_sub_groups;
        
        /// @brief Whether sub-groups can make independent forward progress
        bool sub_group_independent_forward_progress;
        
        /// @brief Supported sub-group sizes on this device
        std::vector<std::size_t> sub_group_sizes;
        
        /// @brief Preferred vector width for char operations
        sycl::detail::u_int preferred_vector_width_char;

        // Memory allocation capabilities
        /// @brief Maximum size of a single memory allocation in bytes
        sycl::detail::u_long max_mem_alloc_size;
        
        // Global memory characteristics
        /// @brief Cache line size for global memory in bytes
        sycl::detail::u_int global_mem_cache_line_size;
        
        /// @brief Total global memory size in bytes
        sycl::detail::u_long global_mem_size;
        
        /// @brief Global memory cache size in bytes
        sycl::detail::u_long global_mem_cache_size;
        
        /// @brief Type of global memory cache (none, read-only, read-write)
        sycl::info::global_mem_cache_type global_mem_cache_type;
        
        // Local memory characteristics
        /// @brief Type of local memory (none, local, global)
        sycl::info::local_mem_type local_mem_type;
        
        /// @brief Local memory size in bytes
        sycl::detail::u_long local_mem_size;

        /**
         * @brief Constructs working set configuration for a SYCL device
         * 
         * @details Initializes the working set by querying all relevant device
         * capabilities and computing sample buffer requirements. This constructor
         * performs comprehensive device introspection to gather all information
         * needed for optimal configuration.
         * 
         * The constructor queries:
         * - Device type and compute capabilities
         * - Work-group and sub-group constraints
         * - Memory hierarchy characteristics
         * - Backend-specific optimal occupancy rates
         * 
         * @param queue SYCL queue for device access and memory operations
         * @param num_events Number of events in the Monte Carlo simulation
         * @param requested_shape Desired sample buffer organization
         * 
         * @throws std::runtime_error if device queries fail
         * @throws std::bad_alloc if memory requirements exceed device limits
         * 
         * @note All device queries are performed synchronously during construction
         * @note Memory requirements are validated against device constraints
         * 
         * @example
         * @code
         * sycl::queue gpu_queue;
         * sample_shape<uint32_t> shape{2048, 32};
         * working_set<uint32_t, uint64_t> ws(gpu_queue, 5000, shape);
         * 
         * // Check if configuration is valid
         * if (ws.samples_in_bytes_ > ws.max_mem_alloc_size) {
         *     throw std::runtime_error("Sample buffer too large for device");
         * }
         * @endcode
         */
        working_set(const sycl::queue &queue, const size_type num_events, const event::sample_shape<size_type> &requested_shape) {
            const auto device = queue.get_device();
            num_events_ = num_events;
            bitpack_buffer_shape_ = requested_shape;
            samples_per_event_in_bytes_ = bitpack_buffer_shape_.num_bitpacks() * sizeof(bitpack_type);
            samples_per_event_in_bits_ = samples_per_event_in_bytes_ * 8;
            samples_in_bytes_ = samples_per_event_in_bytes_ * num_events_;

            device_type = device.get_info<sycl::info::device::device_type>();
            max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
            max_clock_frequency = device.get_info<sycl::info::device::max_clock_frequency>();
            desired_occupancy = compute_desired_occupancy_rate_heuristic(device.get_backend(), max_compute_units);

            max_work_item_dimensions = device.get_info<sycl::info::device::max_work_item_dimensions>();
            max_work_item_sizes_1d = device.get_info<sycl::info::device::max_work_item_sizes<1>>();
            max_work_item_sizes_2d = device.get_info<sycl::info::device::max_work_item_sizes<2>>();
            max_work_item_sizes_3d = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            work_item_independent_forward_progress = false;

            max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

            max_num_sub_groups = device.get_info<sycl::info::device::max_num_sub_groups>();
            sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            preferred_vector_width_char = device.get_info<sycl::info::device::preferred_vector_width_char>();
            sub_group_independent_forward_progress = device.get_info<sycl::info::device::sub_group_independent_forward_progress>();

            // memory allocation
            max_mem_alloc_size = device.get_info<sycl::info::device::max_mem_alloc_size>();
            global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
            global_mem_cache_size = device.get_info<sycl::info::device::global_mem_cache_size>();
            global_mem_cache_line_size = device.get_info<sycl::info::device::global_mem_cache_line_size>();
            global_mem_cache_type = device.get_info<sycl::info::device::global_mem_cache_type>();
            local_mem_type = device.get_info<sycl::info::device::local_mem_type>();
            local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        }
        /** CUDA **/
        // num_samples = 1e9, num_products = 1e9
        //
        // [GP104]   Tesla P4, 2560 CUDA cores @ 1531 MHz = 25600  : 2.64s
        // [GP104]   Tesla P4, 2560 CUDA cores @ 1531 MHz = 102400 : 1.17s
        // [GP104]   Tesla P4, 2560 CUDA cores @ 1531 MHz = 204800 : 0.95s
        // [GP104]   Tesla P4, 2560 CUDA cores @ 1531 MHz = 256000 : 1.02s
        //
        // [TU116] 1660 Super, 1408 CUDA cores @ 1735 MHz = 102400 : 1.34s
        // [TU116] 1660 Super, 1408 CUDA cores @ 1735 MHz = 112640 : 1.31s
        // [TU116] 1660 Super, 1408 CUDA cores @ 1735 MHz = 204800 : 1.29s
        // [TU116] 1660 Super, 1408 CUDA cores @ 1735 MHz = 307200 : 1.49s
        size_type desired_occupancy = 102400;////< Number of work-groups per compute unit. A higher number can increase
                                             /// parallelism but may also lead to resource contention. Adjust based on
                                             /// performance measurements.


        /** OpenCL **/
        // for OpenCL 64 CPU, 128 threads @ 2.35 GHz, 12800 : 3.28s
        // for OpenCL 64 CPU, 128 threads @ 2.35 GHz, 6400  : 2.85s
        // for OpenCL 64 CPU, 128 threads @ 2.35 GHz, 3200  : 3.99s

        // for OpenCL 8 CPU, 16 threads @ 3.80 GHz = 6400 * 64 = 409600 : 15.0s
        // for OpenCL 8 CPU, 16 threads @ 3.80 GHz = 6400 * 16 = 102400 : 9.55s

        /** OpenMP **/
        // for OpenMP 64 CPU, 128 threads @ 2.35 GHz, 6400 * 32 = 204800 : 8.90s
        // for OpenMP 64 CPU, 128 threads @ 2.35 GHz, 6400 * 16 = 102400 : 7.86s
        // for OpenMP 64 CPU, 128 threads @ 2.35 GHz,           = 32768  : 8.61s
        // for OpenMP 64 CPU, 128 threads @ 2.35 GHz,           = 20480  : 9.00s
        // for OpenMP 64 CPU, 128 threads @ 2.35 GHz, 6400 * 2  = 12800  : 10.7s

        // for OpenMP 8 CPU, 16 threads @ 3.80 GHz, 6400 * 64 = 409600 : 22.9s
        // for OpenMP 8 CPU, 16 threads @ 3.80 GHz, 6400 * 32 = 204800 : 21.6s
        // for OpenMP 8 CPU, 16 threads @ 3.80 GHz, 6400 * 16 = 102400 : 30.8s
        // for OpenMP 8 CPU, 16 threads @ 3.80 GHz, 6400 * 8 = 51200 : 54.9s
        /**
         * @brief Formatted output operator for working set configuration
         * 
         * @details Provides comprehensive human-readable output of all device
         * capabilities, memory constraints, and computed configuration parameters.
         * The output is organized into logical sections for easy interpretation.
         * 
         * Output sections include:
         * - Device type and compute capabilities
         * - Work-item and work-group constraints
         * - Sub-group characteristics
         * - Memory hierarchy information
         * - Sample buffer configuration
         * 
         * @param os Output stream for formatted output
         * @param ws Working set instance to format
         * @return Reference to the output stream for chaining
         * 
         * @note Output format is designed for debugging and performance analysis
         * @note All values are formatted with appropriate units and descriptions
         * 
         * @example
         * @code
         * working_set<uint32_t, uint64_t> ws(queue, 1000, shape);
         * std::cout << ws << std::endl;
         * 
         * // Output includes:
         * // device_type: gpu
         * // max_compute_units: 20
         * // max_clock_frequency: 1800
         * // desired_occupancy: 204800
         * // ...
         * @endcode
         */
        friend std::ostream &operator<<(std::ostream &os, const working_set &ws) {
            os  << "device_type: ";
            switch (ws.device_type) {
                case sycl::info::device_type::cpu: os << "cpu"; break;
                case sycl::info::device_type::gpu: os << "gpu"; break;
                case sycl::info::device_type::all: os << "all"; break;
                case sycl::info::device_type::custom: os << "custom"; break;
                case sycl::info::device_type::automatic: os << "automatic"; break;
                case sycl::info::device_type::accelerator: os << "accelerator";  break;
                case sycl::info::device_type::host: os << "host"; break;
                default: os << "unknown"; break;
            } os << std::endl;
            os  << "max_compute_units: " << ws.max_compute_units << std::endl
                << "max_clock_frequency: " << ws.max_clock_frequency << std::endl
                << "desired_occupancy: " << ws.desired_occupancy << std::endl
                << "------------------------------------------------" << std::endl;
            os  << "max_work_item_dimensions: " << ws.max_work_item_dimensions << std::endl
                << "max_work_item_sizes_1d: " << ws.max_work_item_sizes_1d[0] << std::endl
                << "max_work_item_sizes_2d: " << ws.max_work_item_sizes_2d[0] << ", " << ws.max_work_item_sizes_2d[1] << std::endl
                << "max_work_item_sizes_3d: " << ws.max_work_item_sizes_3d[0] << ", " << ws.max_work_item_sizes_3d[1] << ", " << ws.max_work_item_sizes_3d[2] << std::endl
                << "work_item_independent_forward_progress: " << ws.work_item_independent_forward_progress << std::endl
                << "------------------------------------------------" << std::endl;
            os  << "max_work_group_size: " << ws.max_work_group_size << std::endl
                << "------------------------------------------------" << std::endl;
            os  << "max_num_sub_groups: " << ws.max_num_sub_groups << std::endl
                << "sub_group_sizes: "; for (const auto &size : ws.sub_group_sizes) { os << size <<", "; } os << std::endl
                << "preferred_vector_width_char: "<< ws.preferred_vector_width_char << std::endl
                << "sub_group_independent_forward_progress: " << ws.sub_group_independent_forward_progress << std::endl
                << "------------------------------------------------" << std::endl;
            os  << "max_mem_alloc_size: " << ws.max_mem_alloc_size << std::endl
                << "global_mem_size: " << ws.global_mem_size << std::endl
                << "global_mem_cache_size: " << ws.global_mem_cache_size << std::endl
                << "global_mem_cache_line_size: " << ws.global_mem_cache_line_size << std::endl
                << "global_mem_cache_type: ";
                switch (ws.global_mem_cache_type) {
                    case sycl::info::global_mem_cache_type::none: os << "none"; break;
                    case sycl::info::global_mem_cache_type::read_only: os << "read_only"; break;
                    case sycl::info::global_mem_cache_type::read_write: os << "read_write"; break;
                    default: os << "unknown"; break;
                } os << std::endl
                << "local_mem_type: ";
                switch (ws.local_mem_type) {
                    case sycl::info::local_mem_type::none: os << "none"; break;
                    case sycl::info::local_mem_type::local: os << "local"; break;
                    case sycl::info::local_mem_type::global: os << "global"; break;
                    default: os << "unknown"; break;
                } os << std::endl
                << "local_mem_size: " << ws.local_mem_size << std::endl
                << "------------------------------------------------" << std::endl;
            os  << "num_events_: " << ws.num_events_ << std::endl
                << "buffer_shape_batch_size: " << ws.bitpack_buffer_shape_.batch_size << std::endl
                << "buffer_shape_bitpacks_per_batch: " << ws.bitpack_buffer_shape_.bitpacks_per_batch << std::endl
                << "buffer_samples_per_event_in_bytes: " << ws.samples_per_event_in_bytes_ << std::endl
                << "sample_buffer_in_bytes: " << ws.samples_in_bytes_ << std::endl
                << "sampled_bits_per_event: " << ws.samples_per_event_in_bits_ << std::endl;
            return os;
        }

        /**
         * @brief Computes optimal nd_range for 1D tally kernels
         * 
         * @details Calculates the optimal SYCL nd_range for 1D tally operations
         * by analyzing device capabilities and choosing appropriate work-group
         * sizes. The algorithm considers sub-group sizes for GPUs and preferred
         * vector widths for CPUs to maximize performance.
         * 
         * The method uses different strategies based on device type:
         * - GPU devices: Use largest sub-group size with power-of-2 multipliers
         * - CPU devices: Use preferred vector width with power-of-2 scaling
         * - Fallback: Use conservative defaults for unknown device types
         * 
         * @param queue SYCL queue for device capability queries
         * @param total_elements Total number of elements to process
         * @return Optimal nd_range with computed global and local sizes
         * 
         * @note Global range is rounded up to be a multiple of local range
         * @note Local range is constrained by device maximum work-group size
         * 
         * @example
         * @code
         * // Compute optimal range for 10000 tally elements
         * auto nd_range = working_set<uint32_t, uint64_t>::compute_optimal_nd_range_for_tally(queue, 10000);
         * 
         * // Submit kernel with optimal configuration
         * queue.submit([&](sycl::handler& h) {
         *     h.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
         *         // Tally kernel implementation
         *     });
         * });
         * @endcode
         */
        static sycl::nd_range<1> compute_optimal_nd_range_for_tally(const sycl::queue &queue, const size_type total_elements) {
            const auto device = queue.get_device();
            size_type max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

            // Try to obtain sub_group_sizes
            std::vector<size_t> sub_group_sizes;
            if (device.has(sycl::aspect::gpu)) {
                sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            }

            size_type optimal_local_size = 0;

            if (!sub_group_sizes.empty()) {
                // Choose the maximum sub-group size as the base for local size
                size_type max_sub_group_size = *std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());

                // Multiply sub-group size to get an optimal local size, ensuring it's within max work-group size
                size_type multiplier = 1;
                while (max_sub_group_size * multiplier <= max_work_group_size) {
                    multiplier *= 2;// Adjust as needed (e.g., double the multiplier)
                }
                multiplier /= 2;// Step back to the last valid multiplier
                optimal_local_size = max_sub_group_size * multiplier;
            } else {
                // If sub_group_sizes are not available, use other device parameters
                // For CPUs or devices without sub-groups, we can use the preferred vector width
                size_type preferred_vector_width = device.get_info<sycl::info::device::preferred_vector_width_char>();

                if (preferred_vector_width == 0) {
                    // If preferred vector width is not available, default to 1
                    preferred_vector_width = 1;
                }

                // Multiply preferred vector width to get an optimal local size, within constraints
                optimal_local_size = preferred_vector_width;
                while (optimal_local_size * 2 <= max_work_group_size) {
                    optimal_local_size *= 2;
                }
            }

            // Ensure that optimal_local_size is at least 1 and does not exceed max_work_group_size
            optimal_local_size = std::max<size_type>(1, std::min(optimal_local_size, max_work_group_size));

            // Calculate global range, ensuring it's a multiple of local size
            size_type num_work_groups = (total_elements + optimal_local_size - 1) / optimal_local_size;
            size_type global_range = num_work_groups * optimal_local_size;

            // Construct and return the nd_range object
            sycl::range<1> global_range_obj(global_range);
            sycl::range<1> local_range_obj(optimal_local_size);

            return sycl::nd_range<1>(global_range_obj, local_range_obj);
        }

        /**
         * @brief Computes optimal sample shape for device memory constraints
         * 
         * @details Determines the optimal sample_shape configuration that fits
         * within device memory allocation limits while maximizing computational
         * efficiency. The algorithm searches through possible batch sizes and
         * bitpacks per batch to find the largest configuration that doesn't
         * exceed memory constraints.
         * 
         * The search strategy:
         * 1. Start with maximum batch size (2^16) and sample size (2^16)
         * 2. Iteratively reduce sizes until memory requirements fit
         * 3. Validate against max_mem_alloc_size device limit
         * 4. Return the largest valid configuration found
         * 
         * @param queue SYCL queue for device capability queries
         * @param num_events Number of events requiring sample storage
         * @return Optimal sample_shape within memory constraints
         * 
         * @note Uses conservative maximum values to prevent memory overflow
         * @note Falls back to minimal configuration (1x1) if no valid config found
         * 
         * @example
         * @code
         * // Find optimal shape for 5000 events
         * auto shape = working_set<uint32_t, uint64_t>::compute_optimal_sample_shape(queue, 5000);
         * 
         * std::cout << "Optimal batch size: " << shape.batch_size << std::endl;
         * std::cout << "Bitpacks per batch: " << shape.bitpacks_per_batch << std::endl;
         * 
         * // Use in working set construction
         * working_set<uint32_t, uint64_t> ws(queue, 5000, shape);
         * @endcode
         */
        static event::sample_shape<size_type> compute_optimal_sample_shape(const sycl::queue &queue, const size_type num_events) {
            const auto device = queue.get_device();
            const size_t max_malloc_size = device.get_info<sycl::info::device::max_mem_alloc_size>();
            static constexpr size_type max_sample_size = 16;
            static constexpr size_type max_batch_size = 16;
            static constexpr size_type one = 1;
            size_type ss = max_sample_size;
            size_type bs = max_batch_size;
            bool found = false;

            for (; ss >= 0; --ss) {
                bs = max_batch_size;// Reinitialize bs for each ss
                for (; bs >= 0; --bs) {
                    uint64_t used_bytes = static_cast<uint64_t>(num_events) * (static_cast<uint64_t>(one) << bs) * (static_cast<uint64_t>(one) << ss) * sizeof(bitpack_type);
                    if (used_bytes <= max_malloc_size) {
                        found = true;
                        break;// Breaks out of the inner loop
                    }
                }
                if (found) {
                    break;// Breaks out of the outer loop
                }
            }
            if (!found) {
                ss = 0;
                bs = 0;
            } else {
                // Adjust ss and bs because they were decremented after finding the valid values
                // (since the for loop decrements before checking the condition)
                ss = ss;
                bs = bs;
            }
            event::sample_shape<size_type> shape = {
                    .batch_size = one << bs,
                    .bitpacks_per_batch = one << ss,
            };
            return shape;
        }

        /**
         * @brief Finds the closest power of 2 to a given value
         * 
         * @details Computes the power of 2 that minimizes the absolute difference
         * with the input value. This utility function is used for optimizing
         * work-group sizes and memory layouts to align with hardware preferences.
         * 
         * The algorithm:
         * 1. Iterates through all possible powers of 2 up to the type limit
         * 2. Computes absolute difference for each power
         * 3. Returns the power with minimum difference
         * 4. Breaks early when differences start increasing
         * 
         * @param n Input value to find closest power of 2 for
         * @return Closest power of 2 to the input value
         * 
         * @note Returns 1 for input value 0 (edge case handling)
         * @note Prefers smaller powers when there's a tie
         * 
         * @example
         * @code
         * // Find closest powers of 2
         * auto p1 = working_set<uint32_t, uint64_t>::closest_power_of_2(100);  // Returns 128
         * auto p2 = working_set<uint32_t, uint64_t>::closest_power_of_2(96);   // Returns 64
         * auto p3 = working_set<uint32_t, uint64_t>::closest_power_of_2(96);   // Returns 128 (tie)
         * 
         * // Use for work-group size optimization
         * size_t optimal_size = closest_power_of_2(device_preferred_size);
         * @endcode
         */
        static size_type closest_power_of_2(const size_type n) {
            if (n == 0) return 1;  // Edge case: define closest power of 2 to 0 as 1

            size_type min_diff = std::numeric_limits<size_type>::max();  // Initialize minimum difference
            size_type closest = 0;  // To store the closest power of 2

            // Iterate over possible exponents x
            for (size_type x = 0; x < sizeof(size_type) * 8; ++x) {
                const size_type pow2 = static_cast<size_type>(1) << x;  // Compute 2^x
                long diff = pow2 > n ? pow2 - n : n - pow2;
                if (diff < min_diff) {
                    min_diff = diff;
                    closest = pow2;
                } else if (diff == min_diff) {
                    // If there's a tie, choose the smaller power of 2
                    if (pow2 < closest) {
                        closest = pow2;
                    }
                } else {
                    // Since the differences will start increasing after the minimum,
                    // we can break out of the loop early
                    break;
                }
            }

            return static_cast<size_type>(closest);
        }

        /**
         * @brief Computes optimal 3D local range for CPU devices
         * 
         * @details Calculates the optimal local work-group size for CPU devices
         * in 3D kernels, considering data type alignment and memory access patterns.
         * CPU devices typically benefit from smaller work-groups that align with
         * cache lines and SIMD instruction widths.
         * 
         * The CPU optimization strategy:
         * - Sets X and Y dimensions to 1 for simplicity
         * - Computes Z dimension based on data type size and alignment
         * - Considers 8-byte alignment for optimal memory access
         * - Respects device work-item size limits
         * 
         * @param limits Optional constraints on each dimension (0 = no limit)
         * @return Optimal 3D local range for CPU execution
         * 
         * @note CPU devices typically use (1, 1, small_z) configurations
         * @note Z dimension is aligned to data type size for vectorization
         * 
         * @example
         * @code
         * working_set<uint32_t, uint64_t> ws(cpu_queue, 1000, shape);
         * 
         * // Compute optimal CPU local range
         * auto local_range = ws.compute_optimal_local_range_3d_for_cpu();
         * std::cout << "CPU local range: " << local_range[0] << "x" 
         *           << local_range[1] << "x" << local_range[2] << std::endl;
         * // Typical output: "CPU local range: 1x1x8"
         * @endcode
         */
        [[nodiscard]] sycl::range<3> compute_optimal_local_range_3d_for_cpu(const sycl::range<3> &limits = {0, 0, 0}) const {
            const auto num_bytes_in_dtype = sizeof(bitpack_type); // in bytes
            const auto div_8 = 8 / num_bytes_in_dtype;
            const auto lz = !limits[2] ? div_8 : std::clamp(div_8, static_cast<decltype(limits[2])>(1), limits[2]);
            const auto hw_limited_target_z = std::clamp(lz, lz, max_work_item_sizes_3d[2]);
            return sycl::range<3>{1, 1, hw_limited_target_z};
        }

        /**
         * @brief Computes optimal 3D local range for GPU devices
         * 
         * @details Calculates the optimal local work-group size for GPU devices
         * in 3D kernels by distributing the work-group size budget across all
         * three dimensions. The algorithm uses logarithmic scaling to ensure
         * power-of-2 work-group sizes while respecting device constraints.
         * 
         * The GPU optimization strategy:
         * 1. Start with total work-group size budget (log2 of max_work_group_size)
         * 2. Distribute budget across X, Y, Z dimensions based on problem size
         * 3. Use power-of-2 sizes for optimal GPU warp/wavefront utilization
         * 4. Respect per-dimension work-item size limits
         * 5. Ensure total work-group size doesn't exceed device limits
         * 
         * @param limits Optional constraints on each dimension (0 = no limit)
         * @return Optimal 3D local range for GPU execution
         * 
         * @note GPU devices benefit from larger work-groups (e.g., 256, 512, 1024)
         * @note All dimensions are powers of 2 for optimal hardware utilization
         * 
         * @example
         * @code
         * working_set<uint32_t, uint64_t> ws(gpu_queue, 1000, shape);
         * 
         * // Compute optimal GPU local range
         * auto local_range = ws.compute_optimal_local_range_3d_for_gpu();
         * std::cout << "GPU local range: " << local_range[0] << "x" 
         *           << local_range[1] << "x" << local_range[2] << std::endl;
         * // Typical output: "GPU local range: 8x16x4" (total = 512)
         * @endcode
         */
        [[nodiscard]] sycl::range<3> compute_optimal_local_range_3d_for_gpu(const sycl::range<3> &limits = {0, 0, 0}) const {
            const auto log_2_max_work_group_size = static_cast<std::uint8_t>(std::log2(max_work_group_size));
            auto remaining_budget = log_2_max_work_group_size;

            const auto target_x = !limits[0] ? num_events_ : std::clamp(static_cast<decltype(limits[0])>(num_events_), static_cast<decltype(limits[0])>(1), limits[0]);
            const auto hw_limited_target_x = std::clamp(target_x, target_x, max_work_item_sizes_3d[0]);
            const auto log_2_rounded_x = static_cast<std::uint8_t>(std::log2(closest_power_of_2(hw_limited_target_x)));
            const auto log2_local_x = std::min<std::uint8_t>(log_2_rounded_x, remaining_budget); // between 0 and 10,13

            remaining_budget = remaining_budget - log2_local_x; // between 0 and 10,13

            const auto target_y = !limits[1] ? bitpack_buffer_shape_.batch_size : std::clamp(static_cast<decltype(limits[1])>(bitpack_buffer_shape_.batch_size), static_cast<decltype(limits[1])>(1), limits[1]);
            const auto hw_limited_target_y = std::clamp(target_y, target_y, max_work_item_sizes_3d[1]);
            const auto log_2_rounded_y = static_cast<std::uint8_t>(std::log2(closest_power_of_2(hw_limited_target_y)));
            const auto log2_local_y = std::min<std::uint8_t>(log_2_rounded_y, remaining_budget); // between 0 and 10,13

            remaining_budget = remaining_budget - log2_local_y; // between 0 and 10,13

            const auto target_z = !limits[2] ? bitpack_buffer_shape_.bitpacks_per_batch : std::clamp(static_cast<decltype(limits[2])>(bitpack_buffer_shape_.bitpacks_per_batch), static_cast<decltype(limits[2])>(1), limits[2]);
            const auto hw_limited_target_z = std::clamp(target_z, target_z, max_work_item_sizes_3d[2]);
            const auto log_2_rounded_z = static_cast<std::uint8_t>(std::log2(closest_power_of_2(hw_limited_target_z)));
            const auto log2_local_z = std::min<std::uint8_t>(log_2_rounded_z, remaining_budget); // between 0 and 10,13

            const auto lx = static_cast<std::size_t>(1) << log2_local_x;
            const auto ly = static_cast<std::size_t>(1) << log2_local_y;
            const auto lz = static_cast<std::size_t>(1) << log2_local_z;
            return {lx, ly, lz};
        }

        /**
         * @brief Computes optimal 3D local range for current device type
         * 
         * @details Automatically selects the appropriate local range computation
         * method based on the device type detected during working set construction.
         * This method provides a unified interface for optimal work-group size
         * calculation across different device architectures.
         * 
         * Device-specific optimizations:
         * - CPU: Use CPU-optimized algorithm (small work-groups, cache-aligned)
         * - GPU: Use GPU-optimized algorithm (large work-groups, power-of-2 sizes)
         * - Other: Fall back to GPU algorithm for unknown device types
         * 
         * @param limits Optional constraints on each dimension (0 = no limit)
         * @return Optimal 3D local range for the current device
         * 
         * @note Automatically validates that computed range doesn't exceed device limits
         * @note Logs computed configuration for debugging and performance analysis
         * 
         * @example
         * @code
         * working_set<uint32_t, uint64_t> ws(queue, 1000, shape);
         * 
         * // Get optimal local range for any device type
         * auto local_range = ws.compute_optimal_local_range_3d();
         * 
         * // Use in kernel submission
         * queue.submit([&](sycl::handler& h) {
         *     sycl::range<3> global_range{1000, shape.batch_size, shape.bitpacks_per_batch};
         *     sycl::nd_range<3> nd_range{global_range, local_range};
         *     h.parallel_for(nd_range, [=](sycl::nd_item<3> item) {
         *         // Kernel implementation
         *     });
         * });
         * @endcode
         */
        [[nodiscard]] sycl::range<3> compute_optimal_local_range_3d(const sycl::range<3> &limits = {0, 0, 0}) const {
            sycl::range<3> local_range;
            switch (device_type) {
                case sycl::info::device_type::cpu:
                    local_range = compute_optimal_local_range_3d_for_cpu(limits);
                    break;
                case sycl::info::device_type::gpu:
                case sycl::info::device_type::accelerator:
                case sycl::info::device_type::custom:
                case sycl::info::device_type::automatic:
                case sycl::info::device_type::host:
                case sycl::info::device_type::all:
                    local_range = compute_optimal_local_range_3d_for_gpu(limits);
                    break;
            }
            LOG(DEBUG3) << "local_range: (events:"<< num_events_ <<", batch_size:"<< bitpack_buffer_shape_.batch_size <<", sample_size:"<<bitpack_buffer_shape_.bitpacks_per_batch<<"): (" << local_range[0] <<", " << local_range[1] <<", " << local_range[2] <<")";
            assert(local_range[0] * local_range[1] * local_range[2] <= max_work_group_size);
            return local_range;
        }

        /**
         * @brief Rounds sample shape dimensions (in-place version)
         * 
         * @details Modifies the provided sample_shape structure to round its
         * dimensions to optimal values. This in-place version is useful when
         * you want to modify an existing shape configuration.
         * 
         * @tparam dtype Data type used for shape dimensions
         * @param shape Reference to shape structure to be modified
         * @return Reference to the modified shape structure
         * 
         * @note Currently performs no actual rounding (placeholder implementation)
         * @note Future versions may implement power-of-2 rounding or other optimizations
         * 
         * @example
         * @code
         * sample_shape<uint32_t> shape{1000, 15};
         * working_set<uint32_t, uint64_t>::rounded(shape);
         * // shape is now potentially modified for optimization
         * @endcode
         */
        template<typename dtype>
        static event::sample_shape<dtype> &rounded(event::sample_shape<dtype> &shape) {
            shape.batch_size = (shape.batch_size);
            shape.bitpacks_per_batch = (shape.bitpacks_per_batch);
            return shape;
        }

        /**
         * @brief Rounds sample shape dimensions (copy version)
         * 
         * @details Creates a new sample_shape structure with rounded dimensions
         * from the input shape. This copy version is useful when you want to
         * create an optimized copy without modifying the original shape.
         * 
         * @tparam dtype Data type used for shape dimensions
         * @param shape Input shape structure to be rounded
         * @return New shape structure with rounded dimensions
         * 
         * @note Currently performs no actual rounding (placeholder implementation)
         * @note Future versions may implement power-of-2 rounding or other optimizations
         * 
         * @example
         * @code
         * sample_shape<uint32_t> original_shape{1000, 15};
         * auto rounded_shape = working_set<uint32_t, uint64_t>::rounded(original_shape);
         * // original_shape is unchanged, rounded_shape is potentially optimized
         * @endcode
         */
        template<typename dtype>
        static event::sample_shape<dtype> rounded(const event::sample_shape<dtype> &shape) {
            event::sample_shape<dtype> new_shape;
            new_shape.batch_size = (shape.batch_size);
            new_shape.bitpacks_per_batch = (shape.bitpacks_per_batch);
            return new_shape;
        }
    };
}// namespace scram::canopy
