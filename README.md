# mcSCRAM: Monte Carlo SCRAM

> **⚠️ ALPHA STAGE**  
> This is an experimental implementation with unstable APIs subject to frequent changes.  
> Interfaces may change without notice between versions.

**mcSCRAM** is a fork of [SCRAM](https://github.com/rakhimov/scram) that extends the original probabilistic risk assessment tool with multicore CPU, GPU-accelerated Monte Carlo simulation capabilities AdaptiveCpp's SYCL backend.

## Table of Contents

[[TOC]]

## Project Origin

This repository is forked from Olzhas Rakhimov's [SCRAM](https://github.com/rakhimov/scram) (System for Command-line Risk Analysis Multi-tool). The original SCRAM provides comprehensive fault tree and event tree analysis capabilities. This fork specifically focuses on enhancing Monte Carlo simulation performance through hardware acceleration.

## Objectives

The primary goals of this project include:

- **Parallel Monte Carlo Implementation**: Developing SYCL-based kernels for massively parallel sampling across GPU compute units
- **Statistical Precision**: Implementing advanced uncertainty quantification with confidence interval estimation
- **Hardware Optimization**: Exploring memory-efficient data structures and optimal kernel configurations for various accelerator architectures
- **Performance Characterization**: Benchmarking scalability and computational efficiency improvements over traditional CPU-based approaches

<details>
<summary>

## Technical Implementation

</summary>

### Monte Carlo Engine
The core contribution lies in the parallel Monte Carlo implementation featuring:
- **Philox PRNG**: Counter-based pseudorandom number generation enabling perfect parallelization without synchronization overhead
- **Bit-packed Sampling**: Memory-efficient boolean storage minimizing bandwidth requirements during GPU execution
- **Layered Graph Execution**: Topologically sorted fault tree evaluation with dependency-aware scheduling

### Hardware Acceleration
- **SYCL Backend**: Cross-platform acceleration via AdaptiveCpp supporting CUDA, ROCm, Intel oneAPI, and OpenCL
- **Work-group Optimization**: Dynamic kernel configuration adaptation for different hardware architectures
- **Memory Coalescing**: Optimized access patterns for GPU memory hierarchies

### Memory Management Architecture

mcSCRAM implements a **strict USM-only memory strategy** that eliminates SYCL buffers entirely, following AdaptiveCpp performance recommendations:

#### Device USM for High-Throughput Data
```cpp
// Large contiguous allocations for computational data
bitpack_t_* buffer_block = sycl::malloc_device<bitpack_t_>(
    num_events * num_bitpacks, queue);
```
- **Sample Data**: All Monte Carlo sample data resides in device memory
- **Contiguous Layout**: Single large allocations reduce fragmentation
- **Zero-Copy**: Computational kernels access data directly without transfers

#### Shared USM for Metadata and Control
```cpp
// Small metadata structures accessible from host
gate_t* gates = sycl::malloc_shared<gate_t>(num_gates, queue);
```
- **Graph Metadata**: Event and gate configurations in shared memory
- **Pointer Arrays**: Input/output buffer references for kernel dispatch
- **Host Access**: Configuration and results accessible without explicit copies

#### Performance Benefits
- **Eliminated Buffer Overhead**: No accessor creation or runtime dependency analysis
- **Predictable Memory Layout**: Static allocation patterns enable optimal caching
- **Reduced Host Latency**: Direct pointer access vs. buffer submission queues

### Bit-Packing Optimization

Monte Carlo simulations are memory bandwidth-bound. mcSCRAM addresses this through aggressive bit-packing:

```cpp
template<typename bitpack_t_>  // typically uint64_t
static bitpack_t_ generate_samples(const sampler_args &args) {
    constexpr uint8_t bits_in_bitpack = sizeof(bitpack_t_) * 8;  // 64 bits
    constexpr uint8_t samples_per_pack = bits_in_bitpack / bernoulli_bits_per_generation;
    // Pack 64 boolean samples into single 64-bit integer
}
```

#### Memory Bandwidth Optimization
- **64:1 Compression**: 64 boolean samples packed into single `uint64_t`
- **Coalesced Access**: Contiguous memory layout maximizes GPU memory throughput
- **Cache Efficiency**: Reduced memory footprint improves L1/L2 cache utilization

#### Configurable Dimensions
- **Batch Size**: Number of simulation trials processed simultaneously
- **Sample Size**: Bit-packs per batch (configurable: 16, 32, 64 typical)
- **Dynamic Sizing**: Runtime optimization based on device memory and compute capabilities

</details>

## Build and Installation

### Container-based Development (Recommended)

The project provides multi-stage Docker builds for different phases:

```bash
# Development environment with full toolchain
docker build --target devimage -t mc-scram:dev .
docker run -it --rm --gpus all -v $(pwd):/workspace mc-scram:dev

# Production runtime (minimal dependencies)
docker build --target scramruntime -t mc-scram:runtime .
```

Build arguments for configurations:
- `CMAKE_BUILD_TYPE`: Debug, Release, RelWithDebInfo, MinSizeRel (default: Release)
- `APP_MALLOC_TYPE`: tcmalloc, jemalloc, malloc (default: tcmalloc)

### Native Build

#### Requirements:
- CMake ≥ 3.18.4
- C++23 compiler:
  - Clang ≥ 18.0 ✅
  - GCC ≥ 7.1 ⚠️ Untested
  - AppleClang ≥ 9.0 ⚠️ Untested
  - Intel ≥ 18.0.1 ⚠️ Untested
- AdaptiveCpp ≥ 25.2.0
- Memory allocator:
  - tcmalloc (default)
  - jemalloc
  - malloc
- Drivers for your CUDA/ROCm/OpenCL/ZE/OpenMP runtimes

#### Auto-Fetched
- LibXML2 (with LZMA, ZLIB, and ICONV support for compressed `.xml.gz` files)
- Boost 1.88.0 libraries (automatically fetched via FetchContent):
  - program_options, filesystem, system, random, range
  - exception, multi_index, accumulators, multiprecision
  - icl, math, dll, regex, unit_test_framework


```bash
git clone --recursive https://github.com/your-username/mc-scram.git
cd mc-scram
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMALLOC_TYPE=tcmalloc \
  -DBUILD_TESTS=ON \
  -DOPTIMIZE_FOR_NATIVE=ON
make -j$(nproc)
```

### CMake Build Options

| Option | Description | Default | Values |
|--------|-------------|---------|--------|
| `CMAKE_BUILD_TYPE` | Build configuration | Release | Debug, Release, RelWithDebInfo, MinSizeRel |
| `MALLOC_TYPE` | Memory allocator | tcmalloc | tcmalloc, jemalloc, malloc |
| `BUILD_TESTS` | Build test suite | ON | ON, OFF |
| `WITH_COVERAGE` | Enable coverage instrumentation | OFF | ON, OFF |
| `WITH_PROFILE` | Enable profiling instrumentation | OFF | ON, OFF |
| `OPTIMIZE_FOR_NATIVE` | Build with -march=native | ON | ON, OFF |
| `BUILD_SHARED_LIBS` | Build shared libraries | OFF | ON, OFF |

## Usage

```bash
# Container execution
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo --num-trials 1000000 /input/model.xml

# Native binary
./scram --monte-carlo --num-trials 1000000 \
        --confidence-intervals input/model.xml
```

### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-trials` | Monte Carlo iterations | 1,000,000 |
| `--batch-size` | Samples per kernel launch | 1,024 |
| `--sample-size` | Bit-packs per batch | 16 |
| `--confidence-intervals` | Statistical bounds (95%, 99%) | disabled |

## Runtime Environment Variables

AdaptiveCpp environment variables control hardware acceleration behavior, debugging output, and performance tuning. For detailed performance optimization guidance, see the [AdaptiveCpp Performance Tuning Guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md).

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `ACPP_VISIBILITY_MASK` | Controls which backends are available for execution | `cuda`, `rocm`, `opencl`, `lz`, `omp`, combinations (e.g., `cuda,opencl`), or `all` | `all` |
| `ACPP_DEBUG_LEVEL` | Controls runtime debug output verbosity | `0` (silent), `1` (fatal), `2` (errors/warnings), `3` (info) | `0` |
| `ACPP_ADAPTIVITY_LEVEL` | Controls JIT kernel optimization and runtime adaptivity | `0` (static), `1` (basic), `2` (standard) | `2` |
| `ACPP_ALLOCATION_TRACKING` | Enables memory allocation tracking for debugging | `0` (disabled), `1` (enabled) | `0` |

### Usage Examples

```bash
# Production: CUDA backend with minimal output
export ACPP_VISIBILITY_MASK=cuda ACPP_DEBUG_LEVEL=0
./scram --monte-carlo input/model.xml

# Development: Multiple backends with error reporting and memory tracking
export ACPP_VISIBILITY_MASK=cuda,opencl ACPP_DEBUG_LEVEL=2 ACPP_ALLOCATION_TRACKING=1
./scram --monte-carlo input/model.xml

# Container usage with environment variables
docker run --rm --gpus all \
  -e ACPP_VISIBILITY_MASK=cuda \
  -e ACPP_DEBUG_LEVEL=1 \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo /input/model.xml
```

## Performance Considerations

### JIT Optimization and Warm-up
mcSCRAM uses AdaptiveCpp's `generic` compilation target, which performs **runtime JIT optimization**:

- **First Run**: Kernels compile and optimize for your specific hardware
- **Subsequent Runs**: Optimized kernels load from cache (`~/.acpp/apps/`)
- **Recommendation**: Run **3-4 iterations** to reach peak performance
- **Adaptivity Level ≥ 2**: Enables aggressive optimizations including constant propagation for invariant kernel arguments

```bash
# First run - includes JIT compilation time
time ./scram --monte-carlo --num-trials 1000000 input/model.xml

# Subsequent runs - optimized kernel execution
time ./scram --monte-carlo --num-trials 1000000 input/model.xml
```

### Cache Management
When upgrading AdaptiveCpp or GPU drivers, clear the kernel cache to benefit from improvements:
```bash
# Clear JIT kernel cache
rm -rf ~/.acpp/apps/*
```

### Memory Layout Optimization
For large models with memory constraints:
- **Monitor VRAM**: Use `nvidia-smi`, `nvtop` or similar tools to track memory usage

### Backend-Specific Tuning
- **CUDA/HIP**: Optimal for discrete GPUs with high memory bandwidth
- **OpenCL**: Cross-platform compatibility, may require driver-specific tuning
- **Level Zero**: Optimized for Intel discrete GPUs, experimental for integrated GPUs

## Contributing

Please see `CONTRIBUTING.md` for development guidelines and `ICLA.md` for contributor license requirements.

## Licensing

This program is free software distributed under the **GNU Affero General Public License v3.0** (AGPL v3).

**Important Note:** The original SCRAM code (from Olzhas Rakhimov) remains under GPL v3, while mcSCRAM enhancements and new code are licensed under AGPL v3. When combined, the entire project is governed by AGPL v3 terms.

**Key implications of AGPL v3:**
- ✅ **Freedom to use** for any purpose, including research and commercial applications
- ✅ **Freedom to study and modify** the source code
- ✅ **Freedom to distribute** copies and modifications
- ⚠️ **Copyleft requirement**: Derivative works must also be licensed under AGPL v3
- ⚠️ **Source disclosure**: Distributed binaries must include or provide access to source code
- ⚠️ **Network provision**: If you run AGPL code on a server accessible over a network, you must provide source code to users

**For users and researchers:**
- Publication of results does not require AGPL compliance
- Modifications for personal research do not require public release
- If you provide the software as a network service, users must be able to access the source code

**For developers and redistributors:**
- Must preserve copyright notices and license terms
- Must provide source code when distributing binaries
- Must provide source code when offering the software as a network service
- Cannot incorporate into proprietary software without AGPL compliance

**For commercial users:**
- Must comply with AGPL if distributing the software or offering it as a service
- Network-accessible deployments require source code provision to users

**Educational resources on AGPL v3:**
- [Official AGPL v3 Text](https://www.gnu.org/licenses/agpl-3.0.html)
- [AGPL v3 Quick Guide](https://www.gnu.org/licenses/quick-guide-gplv3.html)
- [AGPL v3 FAQ](https://www.gnu.org/licenses/gpl-faq.html)
- [Understanding Copyleft](https://copyleft.org/guide/)
- [Why AGPL?](https://www.gnu.org/licenses/why-affero-gpl.html)

## Acknowledgments

- **Original SCRAM**: Copyright (C) 2014-2018 Olzhas Rakhimov  
  Repository: https://github.com/rakhimov/scram
- **mcSCRAM**: Copyright (C) 2025 Arjun Earthperson
- **Synthetic Models**: OpenPRA Initiative contributors
- **Testing Infrastructure**: Fault tree benchmarks from various PRA/PSA research groups
