# mcSCRAM: Monte Carlo SCRAM

> **⚠️ RESEARCH TOOL - ALPHA STAGE**  
> This is an experimental implementation with unstable APIs subject to frequent changes.  
> Interfaces may change without notice between versions.

**mcSCRAM** is a fork of [SCRAM](https://github.com/rakhimov/scram) that extends the original probabilistic risk assessment tool with multicore CPU, GPU-accelerated Monte Carlo simulation capabilities AdaptiveCpp's SYCL backend.

## Project Origin

This repository is forked from Olzhas Rakhimov's [SCRAM](https://github.com/rakhimov/scram) (System for Command-line Risk Analysis Multi-tool). The original SCRAM provides comprehensive fault tree and event tree analysis capabilities. This fork specifically focuses on enhancing Monte Carlo simulation performance through hardware acceleration.

## Objectives

The primary goals of this project include:

- **Parallel Monte Carlo Implementation**: Developing SYCL-based kernels for massively parallel sampling across GPU compute units
- **Statistical Precision Enhancement**: Implementing advanced uncertainty quantification with confidence interval estimation
- **Hardware Optimization**: Exploring memory-efficient data structures and optimal kernel configurations for various accelerator architectures
- **Performance Characterization**: Benchmarking scalability and computational efficiency improvements over traditional CPU-based approaches

## Technical Implementation

### Monte Carlo Engine
The core contribution lies in the parallel Monte Carlo implementation featuring:
- **Philox PRNG**: Counter-based pseudorandom number generation enabling perfect parallelization without synchronization overhead
- **Bit-packed Sampling**: Memory-efficient boolean storage minimizing bandwidth requirements during GPU execution
- **Layered Graph Execution**: Topologically sorted fault tree evaluation with dependency-aware scheduling

### Hardware Acceleration
- **SYCL Backend**: Cross-platform acceleration via AdaptiveCpp supporting CUDA, ROCm, Intel oneAPI, and OpenCL
- **Work-group Optimization**: Dynamic kernel configuration adaptation for different hardware architectures
- **Memory Coalescing**: Optimized access patterns for GPU memory hierarchies

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
- `CMAKE_BUILD_TYPE`: Debug, Release, RelWithDebInfo
- `APP_MALLOC_TYPE`: tcmalloc, jemalloc, malloc

### Native Build

Requirements:
- CMake ≥ 3.18.4
- C++23 compiler (GCC ≥ 7.1, Clang ≥ 5.0)
- AdaptiveCpp for SYCL support
- Boost libraries (automatically fetched)

```bash
git clone --recursive https://github.com/your-username/mc-scram.git
cd mc-scram
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMALLOC_TYPE=tcmalloc
make -j$(nproc)
```

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
