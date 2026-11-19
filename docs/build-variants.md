# Build variants

Supported compile-time variants of fuzzybits (formerly mcSCRAM). Toolchain
versions live in [`toolchain.env`](../toolchain.env); CMake entry points are
the presets in [`CMakePresets.json`](../CMakePresets.json).

```bash
cmake --preset <variant> && cmake --build --preset <variant>
# optional, currently a no-op until #31 re-enables the suite:
ctest --preset <variant>
```

Presets require CMake ≥ 3.25 (schema version 6). `cmake_minimum_required` in
`CMakeLists.txt` stays 3.18.4 for non-preset configure.

| variant | platform | AdaptiveCpp flow | runtime backends | notes |
|---|---|---|---|---|
| **`generic`** (default) | Linux x86_64 (+ aarch64 later) | `generic` (SSCP JIT) | CUDA, OpenMP CPU | covers sm_86 / sm_120 via runtime JIT |
| **`cuda-sm70`** | Linux x86_64 | `cuda:sm_70` (SMCP, clang CUDA), SASS + PTX | CUDA | Volta-specialised AOT |
| **`metal`** | macOS 26, Apple Silicon | `generic` + Metal backend (experimental) | Metal, OpenMP CPU | hosted `macos-metal` + `macos-wheel` in [macos.yml](../.github/workflows/macos.yml). Device-code gaps: [#30](https://github.com/fuzzybit-org/fuzzybits/issues/30) |

No specialisation for sm_86 / sm_120. Use `generic` there (see
[backend-per-architecture](#backend-per-architecture)).

There is also a `debug` preset: generic SSCP, `CMAKE_BUILD_TYPE=Debug`,
`-march=native`, `ACPP_DEBUG_LEVEL=4`.

## CMake presets

Hidden base `base` is inherited by every configure preset. It is not
invoked directly.

| preset | `ACPP_TARGETS` | `MALLOC_TYPE` | `CMAKE_BUILD_TYPE` | `BUILD_PYTHON_BINDINGS` | other cache |
|---|---|---|---|---|---|
| `generic` | `generic` | `malloc` | Release | ON | `OPTIMIZE_FOR_NATIVE=OFF` |
| `cuda-sm70` | `cuda:sm_70` | `malloc` | Release | ON | `ACPP_USE_ACCELERATED_CPU=off`, `ACPP_STDPAR=off`, `ACPP_EXTRA_COMPILE_OPTIONS=--cuda-include-ptx=sm_70`; Linux-only |
| `metal` | `generic` | `malloc` | Release | ON | Darwin-only (Apple Silicon required at run time) |
| `metal-cpu` | (test preset) | | | | `ctest --preset metal-cpu`: same metal build, `ACPP_VISIBILITY_MASK=omp` |
| `debug` | `generic` | `malloc` | Debug | ON | `OPTIMIZE_FOR_NATIVE=ON`, `ACPP_DEBUG_LEVEL=4` |

Option names (from `CMakeLists.txt` / `cmake/Find_adaptivecpp.cmake`):

| CMake option | role |
|---|---|
| `ACPP_TARGETS` | AdaptiveCpp compilation target (`generic`, `cuda:sm_70`, …) |
| `MALLOC_TYPE` | `malloc` (default), `tcmalloc`, `jemalloc` |
| `BUILD_PYTHON_BINDINGS` | native `mcscram_native` extension |
| `BUILD_TESTS` | test suite (ON in CMakeLists; the `add_subdirectory(tests)` block is commented out until [#31](https://github.com/fuzzybit-org/fuzzybits/issues/31)) |
| `WITH_PROFILE` | `-pg` instrumentation |
| `WITH_COVERAGE` | coverage instrumentation |
| `OPTIMIZE_FOR_NATIVE` | `-march=native` (ON in CMakeLists, **OFF** in portable presets) |

Test presets set `CTEST_NO_TESTS_ACTION=ignore` so `ctest --preset <variant>`
exits 0 while the global suite is disabled. `#31` re-enables tests without
changing preset names.

## Pinned toolchains

[`toolchain.env`](../toolchain.env) is the single source of truth.

| pin | value | consumer |
|---|---|---|
| `ACPP_REF` / `ACPP_COMMIT` | `v25.10.0` / `9f842c701a599107cc6d117d3539f971036363a1` | AdaptiveCpp clone; Docker `ARG ACPP_REF` |
| `CLANG_VERSION` / `LLVM_VERSION` | `20` | `llvm.sh`, `clang-20`, Docker `ARG CLANG_VERSION` |
| `CUDA_DEVEL_IMAGE` / `CUDA_RUNTIME_IMAGE` | `nvidia/cuda:12.8.1-{devel,runtime}-ubuntu24.04` | Docker `ARG`s |
| `BOOST_VERSION` | `1.89.0` (≥ `BOOST_MIN_VERSION=1.89`) | `cmake/FindBoost.cmake` FetchContent URL |
| `METAL_CPP_VERSION` / `ACPP_REF_METAL` | `release/metal-cpp_macOS26_iOS26` / `96c7d17a9d029a5660ac06d7519b3763e9823ab6` | [#30](https://github.com/fuzzybit-org/fuzzybits/issues/30) — AdaptiveCpp Metal (`WITH_METAL_BACKEND`) landed after v25.10.0 (upstream `94e1749`); no tagged AdaptiveCpp release yet. metal-cpp tag is AdaptiveCpp's FetchContent default |

CMake loads the file via `cmake/ToolchainPins.cmake` (`FUZZYBITS_PIN_<KEY>`).
Shell: `set -a && . ./toolchain.env && set +a`. Docker/CI: pass matching keys
as `--build-arg` (see [#26](https://github.com/fuzzybit-org/fuzzybits/issues/26)).

## `MALLOC_TYPE=malloc`

`CMakeLists.txt` defaults `MALLOC_TYPE` to `malloc`. That is required for
library and Python-module builds: tcmalloc (and typical jemalloc
`LD_PRELOAD` setups) must be loaded **before the process starts**. A
pip-installed extension is `dlopen`'d by an already-running `python`, so
`LD_PRELOAD` cannot apply.

CLI container images may still use tcmalloc by configuring
`-DMALLOC_TYPE=tcmalloc` **and** setting `LD_PRELOAD` on the image
`ENTRYPOINT` (current `Dockerfile.cuda12.8` `scramruntime` does this). Do
not ship that combination in wheels.

CMake warns if `BUILD_PYTHON_BINDINGS=ON` and `MALLOC_TYPE=tcmalloc`.

## AdaptiveCpp configure recipe per variant

These flags are for **building AdaptiveCpp**, not fuzzybits. fuzzybits then
uses `cmake --preset <variant>` against that install.

### `generic`

SSCP JIT needs the full compiler feature profile (minimal disables SSCP):

```bash
cmake -S AdaptiveCpp -B AdaptiveCpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCLANG_EXECUTABLE_PATH="$(command -v clang++-20)" \
  -DLLVM_DIR=/usr/lib/llvm-20/cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DWITH_CUDA_BACKEND=ON \
  -DWITH_OPENCL_BACKEND=OFF \
  -DWITH_ROCM_BACKEND=OFF \
  -DWITH_LEVEL_ZERO_BACKEND=OFF \
  -DACPP_COMPILER_FEATURE_PROFILE=full
```

OpenMP (`WITH_CPU_BACKEND`) is always on; it is the required CPU fallback.
`ACPP_TARGETS=generic` at fuzzybits configure time.

### `cuda-sm70`

clang CUDA SMCP (integrated-multipass), Volta AOT. SSCP is optional; a
slimmer AdaptiveCpp uses `minimal`:

```bash
cmake -S AdaptiveCpp -B AdaptiveCpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCLANG_EXECUTABLE_PATH="$(command -v clang++-20)" \
  -DLLVM_DIR=/usr/lib/llvm-20/cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DWITH_CUDA_BACKEND=ON \
  -DWITH_OPENCL_BACKEND=OFF \
  -DWITH_ROCM_BACKEND=OFF \
  -DWITH_LEVEL_ZERO_BACKEND=OFF \
  -DACPP_COMPILER_FEATURE_PROFILE=minimal
```

fuzzybits: `ACPP_TARGETS=cuda:sm_70`. The preset also forwards
`--cuda-include-ptx=sm_70` so the fatbin carries SASS + PTX ISA 8.7 (clang 20
never emits PTX 8.8). A `full`-profile AdaptiveCpp can still compile this
preset; `minimal` cannot compile `generic`.

### `metal`

macOS 26, Apple Silicon only. Homebrew `llvm@20` (AArch64 target) is enough
for the GitHub-hosted attempt; AdaptiveCpp docs still recommend a 2-stage LLVM
build for a fully working generic SSCP compiler. Pins:
`ACPP_REF_METAL=96c7d17a9d029a5660ac06d7519b3763e9823ab6` (develop SHA after
upstream `94e1749`; no Metal-capable tagged release yet),
`METAL_CPP_VERSION=release/metal-cpp_macOS26_iOS26` (AdaptiveCpp's metal-cpp
FetchContent `GIT_TAG` when `Metal/Metal.hpp` is absent). Hosted CI lives in
[`.github/workflows/macos.yml`](../.github/workflows/macos.yml) (`macos-26`
arm64): `macos-metal` (native ctest) and `macos-wheel` (cp312 `macosx_*_arm64`
wheel, artifact-only). Device-code gaps (no `double`, emulated 64-bit atomics)
stay on [#30](https://github.com/fuzzybit-org/fuzzybits/issues/30). The wheel
job is **not** a required check; publishing is [#36](https://github.com/fuzzybit-org/fuzzybits/issues/36).

```bash
# llvm@20 / lld@20 / libomp from Homebrew. AppleClang is the AdaptiveCpp
# *host* compiler; LLVM 20 is CLANG_EXECUTABLE_PATH (matches upstream
# macos-metal.yml).
LLVM_PREFIX="$(brew --prefix llvm@20)"
LLD_PREFIX="$(brew --prefix lld@20)"
OMP_PREFIX="$(brew --prefix libomp)"

cmake -S AdaptiveCpp -B AdaptiveCpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCLANG_EXECUTABLE_PATH="${LLVM_PREFIX}/bin/clang++" \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)" \
  -DCMAKE_INSTALL_PREFIX=/your/install/path \
  -DACPP_COMPILER_FEATURE_PROFILE=full \
  -DWITH_METAL_BACKEND=ON \
  -DWITH_OPENCL_BACKEND=OFF \
  -DWITH_CUDA_BACKEND=OFF \
  -DWITH_ROCM_BACKEND=OFF \
  -DWITH_LEVEL_ZERO_BACKEND=OFF \
  -DACPP_LLD_PATH="${LLD_PREFIX}/bin/ld64.lld" \
  -DMETAL_INCLUDE_DIR=/path/to/metal-cpp \
  -DSEQUENTIAL_LINK_LINE="${OMP_PREFIX}/lib/libomp.dylib" \
  "-DOMP_LINK_LINE=-fopenmp ${OMP_PREFIX}/lib/libomp.dylib -L${OMP_PREFIX}/lib/" \
  "-DOMP_CXX_FLAGS=-fopenmp -D_ENABLE_EXTENDED_ALIGNED_STORAGE -I${OMP_PREFIX}/include"
```

If `METAL_INCLUDE_DIR` is omitted, AdaptiveCpp FetchContent-clones
`apple/metal-cpp` at `METAL_CPP_VERSION`. fuzzybits: `ACPP_TARGETS=generic`
(Metal is an SSCP runtime backend, not an AOT `--acpp-targets` value).

CPU tests on a metal build: `ctest --preset metal-cpu` (`ACPP_VISIBILITY_MASK=omp`).
Metal device tests: `ctest --preset metal` (or `ctest -L mc` with
`ACPP_VISIBILITY_MASK=metal`).

### macOS arm64 wheel

`macos-wheel` in [macos.yml](../.github/workflows/macos.yml) builds a
`cp312-macosx_*_arm64` wheel with scikit-build-core against the same cached
AdaptiveCpp prefix as `macos-metal`. `cmake/bundle_acpp.py` copies
`libacpp-rt.dylib` / `libacpp-common.dylib`, `hipSYCL/librt-backend-{metal,omp}.dylib`,
`llvm-to-{backend,host,metal}` plus the SSCP redistributable
(`hipSYCL/ext/llvm/bin/{opt,llc,ld.lld,ld64.lld}`, `libLLVM.dylib`, Homebrew
`libomp.dylib`). `relocate_wheel_rpath.py` rewrites install names to
`@loader_path`; `delocate-wheel --lib-sdir lib` copies leftover non-system
deps next to `libacpp-rt` without flattening `hipSYCL/`. There is no
`libmc.dylib`. Artifact name: `fuzzybits-cp312-macosx_arm64`. Do not publish
from this job ([#36](https://github.com/fuzzybit-org/fuzzybits/issues/36)).

Smoke (clean venv on the hosted runner): `import fuzzybits`;
`list_devices()` shows the paravirtual Metal GPU and the OpenMP host;
`fuzzybits.run(..., backend="omp")` on `input/Aralia/chinese.xml` at
`trials=2**20` is within 4σ of `1.17058e-3`; `backend="metal"` raises a
Python exception until #17/#30 (JIT: "Double type is not supported on Metal
GPU"). CPU pytest: `pytest tests/python -m "not gpu"`.

## `ACPP_VISIBILITY_MASK` / device selection

`omp` is always loaded (AdaptiveCpp requires a CPU backend). An empty mask
exposes every compiled-in backend.

| variant | default mask | device selection | notes |
|---|---|---|---|
| `generic` | `cuda` | `CUDA_VISIBLE_DEVICES` for the NVIDIA device; OpenMP host is always present | without a GPU, `acpp-info -l` lists only the OpenMP host device — that is expected |
| `cuda-sm70` | `cuda` | `CUDA_VISIBLE_DEVICES`; binary only contains sm_70 SASS (+ embedded PTX) | PTX driver-JIT can run the image on newer archs; sm_70 SASS will not |
| `metal` | `metal` (device) / `omp` (`metal-cpu` preset) | AdaptiveCpp auto-detects the Apple GPU; `omp` remains the CPU fallback | `metal` is accepted by AdaptiveCpp's backend loader; it is not yet listed in upstream `env_variables.md`. GitHub-hosted `macos-26` exposes a paravirtual GPU (`Mac2`/`Common3`), not `Apple7`. Unset the mask to let auto-detect pick Metal |

Other runtime defaults used by the test presets: `ACPP_DEBUG_LEVEL=0`,
`ACPP_PERSISTENT_RUNTIME=1`. Production CLI containers may also set
`ACPP_ADAPTIVITY_LEVEL=2` and `ACPP_ALLOCATION_TRACKING=1`.

## Backend-per-architecture

Measurements from [#26](https://github.com/fuzzybit-org/fuzzybits/issues/26)
(das9201, **ms per 1e9 trials**, `basic_event` kernel; nsys). Lower is better.
**Bold** is the fastest flow on that GPU.

| GPU | SMCP SASS (AOT) | PTX via driver JIT | generic SSCP |
|---|---|---|---|
| V100S sm_70 | **306** | 306 | 369 |
| RTX 3070 sm_86 | 514 | 553 | **460** |
| RTX 5060 Ti sm_120 | 441 | **394** | (new SSCP: kernels not measured) |

So:

- **sm_70 (Volta):** SMCP AOT wins (306 vs 369 SSCP). This is the only
  architecture with a specialised preset (`cuda-sm70`).
- **sm_86 (Ampere):** generic SSCP is fastest (460 vs 514 SMCP SASS). Stay on
  `generic`; do not add a `cuda-sm86` preset.
- **sm_120 (Blackwell):** driver-JIT PTX beats AOT SASS (394 vs 441). Stay on
  `generic` (SSCP JIT at runtime). A PTX-only SMCP fatbin is a Docker
  `CUDA_EMBED_PTX=ptx-only` concern, not a CMake preset.

Gate-kernel times and register counts are in #26; they do not change the
preset split. Docker images, Nsight, and `mcscram-smoke`:
[docs/cuda-builds.md](cuda-builds.md).
