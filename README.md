# mcSCRAM: Monte Carlo SCRAM

**mcSCRAM** is a fork of @rakhimov Olzhas Rakhimov's [SCRAM](https://github.com/rakhimov/scram) that extends the original probabilistic risk assessment tool with multicore CPU, GPU-accelerated Monte Carlo simulation capabilities using AdaptiveCpp's SYCL backend.

> [!CAUTION]
> **⚠️ ALPHA ** This project is under active development. The APIs are unstable, interfaces may change without notice
> until the first release.

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
- **Layered Graph Execution**: Topologically sorted PRA model evaluation with dependency-aware scheduling

### Hardware Acceleration
- **SYCL Backend**: Cross-platform acceleration via AdaptiveCpp supporting CUDA, ROCm, Intel oneAPI, and OpenCL
- **Work-group Optimization**: Dynamic kernel configuration adaptation for different hardware architectures
- **Memory Coalescing**: Optimized access patterns for GPU memory hierarchies

### Memory Management Architecture

mcSCRAM implements a **strict USM-only memory strategy** that eliminates SYCL buffers entirely, following AdaptiveCpp performance recommendations:

#### Device USM for High-Throughput Data
```cpp
// Large contiguous allocations for computational data
bitpack_t* buffer_block = sycl::malloc_device<bitpack_t>(
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
template<typename bitpack_t>  // typically uint64_t
static bitpack_t generate_samples(const sampler_args &args) {
    constexpr uint8_t bits_in_bitpack = sizeof(bitpack_t) * 8;  // 64 bits
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

### Optimized ATLEAST Gate Implementation

mcSCRAM implements a **direct bit-counting algorithm** for ATLEAST gates (k-out-of-n logic) that significantly outperforms traditional AND/OR expansion methods.
For example, a 3-out-of-5 ATLEAST gate, if expanded, requires 10 intermediate AND/OR gates. 
mcSCRAM's implementation uses a single kernel with 5 input reads and parallel bit accumulation.

```cpp
// Per-bit accumulation instead of combinatorial expansion
sycl::marray<bitpack_t, NUM_BITS> accumulated_counts(0);
for (auto i = 0; i < num_inputs; ++i) {
    const bitpack_t val = inputs[i][index];
    #pragma unroll
    for (auto bit_idx = 0; bit_idx < NUM_BITS; ++bit_idx) {
        accumulated_counts[bit_idx] += ((val >> bit_idx) & 1);
    }
}
```
- **No Combinatorial Explosion**: Traditional ATLEAST implementations expand k-out-of-n into complex trees of AND/OR gates (C(n,k) combinations)
- **Parallel Bit Processing**: All 64 bits processed simultaneously vs. sequential popcount operations  
- **Memory Efficiency**: Single pass through inputs vs. multiple intermediate gate evaluations
- **Optimal GPU Utilization**: Vectorized accumulation operations match GPU SIMD architecture

**Example**: 
</details>

## Build and Installation

### Container-based Development (Recommended)

The project provides multi-stage Docker builds for different phases. 

- For CUDA runtime support, install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).
- For AMD/ROCm runtime support, install the [AMD Container Toolkit](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/docker-compose.html).
 -- alternatively, you can try mapping devices `/dev/dri`, `/dev/kfd`, but ymmw.
- For Intel GPUs (discrete or embedded), you just need to map devices `/dev/dri`.
```bash
# Development environment with full toolchain
docker build --target devimage -t mc-scram:dev .
docker run -it --rm --gpus all -v $(pwd):/workspace mc-scram:dev

## for intel, amd 
docker run -it --rm --device=/dev/dri -v $(pwd):/workspace mc-scram:dev

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
git clone --recursive https://github.com/a-earthperson/mcSCRAM.git
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

| Option                | Description                      | Default  | Values                                     |
|-----------------------|----------------------------------|----------|--------------------------------------------|
| `CMAKE_BUILD_TYPE`    | Build configuration              | Release  | Debug, Release, RelWithDebInfo, MinSizeRel |
| `MALLOC_TYPE`         | Memory allocator                 | tcmalloc | tcmalloc, jemalloc, malloc                 |
| `BUILD_TESTS`         | Build test suite                 | ON       | ON, OFF                                    |
| `WITH_COVERAGE`       | Enable coverage instrumentation  | OFF      | ON, OFF                                    |
| `WITH_PROFILE`        | Enable profiling instrumentation | OFF      | ON, OFF                                    |
| `OPTIMIZE_FOR_NATIVE` | Build with -march=native         | ON       | ON, OFF                                    |
| `BUILD_SHARED_LIBS`   | Build shared libraries           | OFF      | ON, OFF                                    |

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
```bash
Usage:    mcscram [options] input-files...

Monte Carlo Options:
  --monte-carlo                         enable monte carlo sampling
  -N [ --num-trials ] double (=0)       bernoulli trials [N ∈ ℕ, 0=auto]
  --early-stop                          stop on convergence (implied if N=0)
  --seed int (=372)                     philox-4x32-10 seed
  -d [ --delta ] double (=0.001)        compute as ε=δ·p̂ [δ > 0]
  -b [ --burn-in ] double (=1048576)    trials before convergence check [0=off]
  -a [ --confidence ] double            two-sided conf. lvl [α ∈ (0,1)] (0.99)

Graph Compilation Options:
  --no-kn                               expand k/n to and/or [off]
  --no-xor                              expand xor to and/or [off]
  --nnf                                 compile to negation normal form [off]
  -c [ --compilation-passes ] int (=2)  0=off 1=null-only 2=optimize 
                                        3+=multipass

Debug Options:
  -w [ --watch ]                        enable watch mode [off]
  -h [ --help ]                         display this help message
  --no-report                           dont generate analysis report
  -p [ --oracle ] double (=-1)          true µ [µ ∈ [0,∞), -1=off]
  --preprocessor                        stop analysis after preprocessing
  --print                               print analysis results to terminal
  --serialize                           serialize the input model and exit
  -V [ --verbosity ] int                set log verbosity
  -v [ --version ]                      display version information

Legacy Options:
  --project path                        project analysis config file
  --allow-extern                        **UNSAFE** allow external libraries
  --validate                            validate input files without analysis
  --pdag                                perform qualitative analysis with PDAG
  --bdd                                 perform qualitative analysis with BDD
  --zbdd                                perform qualitative analysis with ZBDD
  --mocus                               perform qualitative analysis with MOCUS
  --prime-implicants                    calculate prime implicants
  --probability                         perform probability analysis
  --importance                          perform importance analysis
  --uncertainty                         perform uncertainty analysis
  --ccf                                 compute common-cause failures
  --sil                                 compute safety-integrity-level metrics
  --rare-event                          use the rare event approximation
  --mcub                                use the MCUB approximation
  -l [ --limit-order ] int              upper limit for the product order
  --cut-off double                      cut-off probability for products
  --mission-time double                 system mission time in hours
  --time-step double                    timestep in hours
  --num-quantiles int                   number of quantiles for distributions
  --num-bins int                        number of bins for histograms
  -o [ --output ] path                  output file for reports
  --no-indent                           omit indented whitespace in output XML
```

### Example Run
```bash
ACPP_VISIBILITY_MASK=cuda \
ACPP_ADAPTIVITY_LEVEL=2 \
ACPP_ALLOCATION_TRACKING=1 \
ACPP_DEBUG_LEVEL=0 \
ACPP_PERSISTENT_RUNTIME=1 \
ACPP_USE_ACCELERATED_CPU=on \
mcscram \
--pdag \
--monte-carlo \
--probability \
--oracle 0.000713018 \
--compilation-passes 5 \
--watch \
../../../input/Aralia/baobab2.xml

[burn-in]     ::      (ε)= 3.026e-06 |      (ε₀)= 7.129e-07 :: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% [00m:01s<00m:00s] [1/1]                                                                        
[convergence] ::      (ε)= 7.135e-07 |      (ε₀)= 7.135e-07 :: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% [00m:02s<00m:00s] [18/18]                                                                      
[log10-conv]  :: log10(ε)= 9.212e-04 | log10(ε₀)= 1.000e-03 :: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% [00m:00s<00m:00s] [4/4]                                                                        
[estimate]    :: p01= 7.12748e-04  |  p05= 7.12919e-04  |  mu = 7.13462e-04  |  p95= 7.14005e-04  |  p99= 7.14175e-04  |                                                                            
[diagnostics] :: z=  1.602e+00 | p_val=  1.092e-01 | CI95=T | CI99=T | n_req=9287175163 | n_rat=  1.001e+00                                                                                         
[accuracy]    :: true(p)= 7.130e-04 | Δ=  4.436e-07 | δ=  6.222e-04 | b=  4.436e-07 | MSE=  1.968e-13 | log10(Δ)= -6.353e+00 | |log10|=  2.701e-04                                                  
[throughput]  :: 5.34 it/s | 42.79 Gbit/it | 228.57 Gbit/s | 492.36 Mbit/node/it | 2.57 Gbit/node/s                                                                                                 
[info-gain]   :: 0.235169 bit/s | 0.043719 bit/iter | Σ 19.346960 bit                                                                                                                               
```
## Runtime Environment Variables

AdaptiveCpp environment variables control hardware acceleration behavior, debugging output, and performance tuning. For detailed performance optimization guidance, see the [AdaptiveCpp Performance Tuning Guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/performance.md).

| Variable                   | Description                                             | Values                                                                              | Default |
|----------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------|---------|
| `ACPP_VISIBILITY_MASK`     | Controls which backends are available for execution     | `cuda`, `rocm`, `opencl`, `lz`, `omp`, combinations (e.g., `cuda,opencl`), or `all` | `all`   |
| `ACPP_DEBUG_LEVEL`         | Controls runtime debug output verbosity                 | `0` (silent), `1` (fatal), `2` (errors/warnings), `3` (info), `4` extra             | `0`     |
| `ACPP_ADAPTIVITY_LEVEL`    | Controls JIT kernel optimization and runtime adaptivity | `0` (static), `1` (basic), `2` (standard)                                           | `2`     |
| `ACPP_ALLOCATION_TRACKING` | Enables memory allocation tracking for debugging        | `0` (disabled), `1` (enabled)                                                       | `0`     |
| `ACPP_PERSISTENT_RUNTIME`  | Keeps the runtime running between successive calls      | `0` (disabled), `1` (enabled)                                                       | `1`     |

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
