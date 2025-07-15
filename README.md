# mc-SCRAM: Monte Carlo Enhancement for SCRAM Probabilistic Risk Assessment

> **⚠️ RESEARCH TOOL - ALPHA STAGE**  
> This is an experimental research implementation with unstable APIs subject to frequent changes.  
> Not recommended for production use. Interfaces may change without notice between versions.

**mc-SCRAM** is a research fork of [SCRAM](https://github.com/rakhimov/scram) that extends the original probabilistic risk assessment tool with GPU-accelerated Monte Carlo simulation capabilities using SYCL and AdaptiveCpp.

## Project Origin

This repository is forked from Olzhas Rakhimov's [SCRAM](https://github.com/rakhimov/scram) (System for Command-line Risk Analysis Multi-tool). The original SCRAM provides comprehensive fault tree and event tree analysis capabilities. This fork specifically focuses on enhancing Monte Carlo simulation performance through hardware acceleration.

## Research Objectives

The primary research goals of this project include:

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

The project provides multi-stage Docker builds for different research phases:

```bash
# Development environment with full toolchain
docker build --target devimage -t mc-scram:dev .
docker run -it --rm --gpus all -v $(pwd):/workspace mc-scram:dev

# Production runtime (minimal dependencies)
docker build --target scramruntime -t mc-scram:runtime .
```

Build arguments for research configurations:
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

### Basic Monte Carlo Analysis
```bash
# Container execution
docker run --rm --gpus all \
  -v $(pwd)/input:/input \
  mc-scram:runtime --monte-carlo --num-trials 1000000 /input/model.xml

# Native binary
./scram --monte-carlo --num-trials 1000000 \
        --confidence-intervals input/model.xml
```

### Research Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-trials` | Monte Carlo iterations | 1,000,000 |
| `--batch-size` | Samples per kernel launch | 1,024 |
| `--sample-size` | Bit-packs per batch | 16 |
| `--confidence-intervals` | Statistical bounds (95%, 99%) | disabled |

### Input Format
The tool accepts Open-PSA Model Exchange Format (MEF) files:

```xml
<?xml version="1.0"?>
<opsa-mef>
  <define-fault-tree name="system">
    <define-gate name="top_event">
      <or>
        <basic-event name="component_a_failure"/>
        <basic-event name="component_b_failure"/>
      </or>
    </define-gate>
  </define-fault-tree>
</opsa-mef>
```

## Research Applications

This implementation has been used in several research studies:

- Performance analysis of parallel PRA quantification engines
- Benchmark comparisons with existing fault tree analysis tools  
- Investigation of GPU acceleration effects on uncertainty quantification
- Development of scalable Monte Carlo methods for large-scale reliability models

Test cases and synthetic models are available in the `input/synthetic-models/` directory, including fault trees ranging from hundreds to tens of thousands of basic events.

## Contributing to Research

We welcome contributions from the probabilistic risk assessment and high-performance computing communities. Areas of particular research interest include:

- Novel parallel algorithms for fault tree analysis
- Advanced statistical methods for uncertainty quantification
- Optimization techniques for different hardware architectures
- Validation studies comparing results with established tools

Please see `CONTRIBUTING.md` for development guidelines and `ICLA.md` for contributor license requirements.

## Licensing

This program is free software distributed under the **GNU General Public License v3.0** (GPL v3).

**Key implications of GPL v3:**
- ✅ **Freedom to use** for any purpose, including research and commercial applications
- ✅ **Freedom to study and modify** the source code
- ✅ **Freedom to distribute** copies and modifications
- ⚠️ **Copyleft requirement**: Derivative works must also be licensed under GPL v3
- ⚠️ **Source disclosure**: Distributed binaries must include or provide access to source code

**For users and researchers:**
- No restrictions on using the software for research or analysis
- Publication of results does not require GPL compliance
- Modifications for personal research do not require public release

**For developers and redistributors:**
- Must preserve copyright notices and license terms
- Must provide source code when distributing binaries
- Cannot incorporate into proprietary software without GPL compliance

**For commercial users:**
- Free to use for internal business operations
- Must comply with GPL if distributing the software
- Consider consulting legal counsel for complex integration scenarios

**Educational resources on GPL v3:**
- [Official GPL v3 Text](https://www.gnu.org/licenses/gpl-3.0.html)
- [GPL v3 Quick Guide](https://www.gnu.org/licenses/quick-guide-gplv3.html)
- [GPL v3 FAQ](https://www.gnu.org/licenses/gpl-faq.html)
- [Understanding Copyleft](https://copyleft.org/guide/)

## Acknowledgments

- **Original SCRAM**: Copyright (C) 2014-2018 Olzhas Rakhimov  
  Repository: https://github.com/rakhimov/scram
- **mc-SCRAM Enhancements**: Copyright (C) 2025 Arjun Earthperson
- **Synthetic Models**: OpenPRA Initiative contributors
- **Testing Infrastructure**: Fault tree benchmarks from various PRA research groups

For questions about this research or potential collaborations, please open an issue or discussion in this repository.
 