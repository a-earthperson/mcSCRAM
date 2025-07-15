# mcSCRAM: Hardware-Accelerated Monte Carlo for Probabilistic Risk Assessment

**mcSCRAM** is a high-performance, hardware-accelerated implementation of SCRAM (System for Command-line Risk Analysis Multi-tool) featuring parallel Monte Carlo simulation using AdaptiveCpp-based SYCL backends.

## Overview

This project enhances the original SCRAM probabilistic risk analysis tool with GPU-accelerated Monte Carlo simulation capabilities. It performs fault tree analysis, event tree analysis, and uncertainty quantification with massive parallelization across modern computing hardware.

### Key Features

- **Hardware Acceleration**: GPU/CPU parallel execution using SYCL with AdaptiveCpp backend
- **High-Performance Monte Carlo**: Parallel sampling with Philox PRNG for cryptographic-quality randomness
- **Statistical Analysis**: Confidence intervals, standard error computation, and uncertainty quantification  
- **Combined Event Tree/Fault Tree Analysis**: Static analysis with logical gate operations (NOT, AND, OR, XOR, ATLEAST, etc.)
- **Memory Efficient**: Bit-packed data structures for optimal memory bandwidth utilization
- **Scalable**: Linear scaling across available compute units with minimal synchronization overhead

### Performance Characteristics

- **Parallel Execution**: Billions of concurrent Monte Carlo trials
- **Memory Optimization**: Bit-packed sampling with configurable batch sizes
- **Statistical Precision**: Confidence intervals computed using Central Limit Theorem
- **Device Optimization**: Automatic work-group sizing for different hardware architectures

## Quick Start

### Docker (Recommended)

The Dockerfile uses multi-stage builds to provide optimized images for different use cases - from minimal production runtime (~3GB) to full development environments with all hardware backends. This approach allows you to choose the right balance of features vs. image size.

#### Docker Build Stages

| Stage | Purpose | Key Components | Use Case |
|-------|---------|----------------|----------|
| `scramruntime` | Production runtime | Minimal deps, SCRAM only | Production analysis |
| `devimage` | Development | Full toolchain, dev tools | Interactive development |
| `ssh-devimage` | SSH development | Dev environment + SSH | Remote development |
| `generic_backend` | Complete backend | All hardware acceleration | Custom builds |
| `adaptivecpp-amd-lz-oneapi-clang` | SYCL backend | AdaptiveCpp with all backends | SYCL development |

#### Production Usage (Minimal Runtime)

```bash
# Build minimal runtime image (~2-3GB)
docker build --target scramruntime -t mc-scram:runtime .

# Run Monte Carlo analysis
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo --num-trials 1000000 /input/example.xml
```

#### Development Environment

```bash
# Build full development environment
docker build --target devimage -t mc-scram:dev .

# Interactive development
docker run -it --rm --gpus all \
  -v $(pwd):/workspace \
  --user $(id -u):$(id -g) \
  mc-scram:dev

# Development with SSH access
docker build --target ssh-devimage -t mc-scram:ssh-dev .
docker run -d --gpus all \
  -p 2222:22 \
  -v $(pwd):/workspace \
  --name mc-scram-dev \
  mc-scram:ssh-dev
```

#### Debug and Profiling

```bash
# Debug build with all tools
docker build --target devimage \
  --build-arg CMAKE_BUILD_TYPE=Debug \
  --build-arg APP_MALLOC_TYPE=jemalloc \
  -t mc-scram:debug .

# Performance profiling build
docker build --target devimage \
  --build-arg CMAKE_BUILD_TYPE=RelWithDebInfo \
  --build-arg APP_MALLOC_TYPE=tcmalloc \
  -t mc-scram:profile .
```

#### Hardware Verification

```bash
# Verify all hardware backends
docker run --rm --gpus all mc-scram:runtime \
  bash -c "acpp-info && clinfo"

# Test specific backend
docker run --rm --gpus all \
  -e ACPP_TARGETS=cuda \
  mc-scram:runtime --monte-carlo /input/test.xml
```

#### Build Arguments

| Argument | Default | Purpose | Options |
|----------|---------|---------|---------|
| `CMAKE_BUILD_TYPE` | Release | Build configuration | Debug, Release, RelWithDebInfo |
| `APP_MALLOC_TYPE` | tcmalloc | Memory allocator | tcmalloc, jemalloc, malloc |
| `USER` | coder | Development user name | Any valid username |
| `UID` | 1000 | Development user ID | User's UID for permissions |
| `GID` | 1000 | Development group ID | User's GID for permissions |

### Manual Build

#### Prerequisites

- **CMake** ≥ 3.18.4
- **C++23** compatible compiler (GCC ≥ 7.1, Clang ≥ 5.0)
- **AdaptiveCpp** (for SYCL support): supports CUDA, ROCm, Intel ZE, OpenCL, OpenMP backends.

#### Build Instructions

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/mc-scram.git
cd mc-scram

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Basic Analysis

```bash
# Run Monte Carlo fault tree analysis (using Docker runtime)
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo --num-trials 1000000 /input/example.xml

# Native binary (after manual build)
./scram --monte-carlo --num-trials 1000000 input/example.xml

# Configure sampling parameters
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo --batch-size 1024 --sample-size 16 /input/fault_tree.xml
```

### Advanced Configuration

```bash
# High-precision analysis with GPU optimization (Docker)
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo \
    --num-trials 10000000 \
    --confidence-intervals \
    /input/complex_system.xml

# Native binary
./scram --monte-carlo \
        --num-trials 10000000 \
        --confidence-intervals \
        input/complex_system.xml
```

### Input File Format

MC-SCRAM uses the Open-PSA Model Exchange Format (MEF) for input files:

```xml
<?xml version="1.0"?>
<opsa-mef>
  <define-fault-tree name="system">
    <define-gate name="top">
      <or>
        <basic-event name="pump_failure"/>
        <basic-event name="valve_failure"/>
      </or>
    </define-gate>
  </define-fault-tree>
</opsa-mef>
```

## Algorithm Details

### Monte Carlo Implementation

- **SYCL Kernels**: Parallel execution across GPU compute units
- **Philox PRNG**: Counter-based random number generation for perfect parallelization
- **Layered Computation**: Topologically sorted graph execution with dependency management
- **Statistical Estimation**: Bernoulli parameter estimation with normal approximation

### Performance Optimization

- **Bit-Packing**: Memory-efficient boolean sample storage
- **Work-Group Optimization**: Device-specific kernel launch configurations  
- **Atomic Reduction**: Minimal synchronization through group-level operations
- **Memory Coalescing**: Optimal GPU memory access patterns

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-trials` | Number of Monte Carlo iterations | 1,000,000 |
| `--batch-size` | Samples per kernel invocation | 1,024 |
| `--sample-size` | Bit-packs per batch | 16 |
| `--confidence-intervals` | Compute 95% and 99% CI | false |

## License

- Copyright (C) 2014 Olzhas Rakhimov [SCRAM]
- Copyright (C) 2025 Arjun Earthperson [MC-SCRAM] (this-repo)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
 