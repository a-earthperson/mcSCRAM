# CUDA image builds

Architecture-specific CUDA images live next to the generic production
`Dockerfile`. Use `Dockerfile.cuda12.8` for anything that has to run on
Volta (`sm_70`) or that needs a slim runtime / Nsight stage.
`Dockerfile.cuda` (CUDA 13.5) is **blocked**; see below.

ARG defaults must match [`toolchain.env`](../toolchain.env) (issue #29 / #59):
AdaptiveCpp `v25.10.0` (`9f842c7`), clang 20, CUDA 12.8.1. That file is the
source of truth; pass it through at build time rather than forking versions:

```bash
set -a && . ./toolchain.env && set +a
docker build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_REF --build-arg ACPP_REPO \
  --build-arg CLANG_VERSION \
  --build-arg CUDA_DEVEL_IMAGE --build-arg CUDA_RUNTIME_IMAGE \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE=minimal \
  --build-arg ACPP_TARGETS=cuda:sm_70 \
  --target scramruntime -t mcscram:cuda-sm70 .
```

`cuda-sm70` uses `ACPP_COMPILER_FEATURE_PROFILE_CUDA_SM70=minimal`; the generic
image uses `ACPP_COMPILER_FEATURE_PROFILE=full`.

## Files

| file | role |
|---|---|
| `Dockerfile` | generic production image (SSCP, AdaptiveCpp `v25.10.0`) |
| `Dockerfile.cuda12.8` | CUDA-only SMCP (and optional generic SSCP), slim runtime + profiler |
| `Dockerfile.cuda` | CUDA 13.5 experiment; fails unless `--build-arg EXPERIMENTAL=1` |
| `docker/acpp-cuda-config.py` | resolves `ACPP_TARGETS` + `CUDA_EMBED_PTX` into clang flags |
| `docker/smoke.sh` | installed as `/usr/local/bin/mcscram-smoke` |
| `docker/inputs/` | `chinese.xml`, `baobab1.xml`, `das9201.xml` |

## `Dockerfile.cuda12.8` stages

`base` → `minimal` → `llvm-clang` → `adaptivecpp-clang` → `builder` →
`devimage` / `ssh-devimage`, then slim `scramruntime` (from
`nvidia/cuda:12.8.1-runtime`) and `profiler` / `scramruntime-prof`.

```bash
# SMCP AOT for a single arch (production on Volta: cuda:sm_70)
docker build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_TARGETS=cuda:sm_70 \
  --target scramruntime -t mcscram:cuda12.8-sm70 .

docker build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_TARGETS=cuda:sm_70 \
  --target scramruntime-prof -t mcscram:cuda12.8-sm70-prof .

# generic SSCP (needs the full AdaptiveCpp compiler feature profile)
docker build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_TARGETS=generic \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE=full \
  --target scramruntime -t mcscram:cuda12.8-generic .
```

Comma-separated `ACPP_TARGETS` are accepted and rewritten to AdaptiveCpp's
semicolon form (Docker ARG cannot easily pass `;`):

```bash
--build-arg ACPP_TARGETS=cuda:sm_70,cuda:sm_86
```

`generic` combined with SMCP (`generic,cuda:sm_70`) also requires
`ACPP_COMPILER_FEATURE_PROFILE=full`.

## Build args

| ARG | default (`Dockerfile.cuda12.8`) | notes |
|---|---|---|
| `ACPP_TARGETS` | `cuda:sm_86` | `cuda:sm_XX`, comma lists for fat binaries, or `generic` |
| `CUDA_EMBED_PTX` | `highest` | `none` \| `highest` \| `all` \| `ptx-only` |
| `ACPP_REF` | `v25.10.0` | AdaptiveCpp git ref; same pin as the generic `Dockerfile` |
| `ACPP_COMPILER_FEATURE_PROFILE` | `minimal` | `full` required for `generic` SSCP |
| `CLANG_VERSION` | `20` | CUDA 12.8 is the last toolkit clang 20 officially supports |
| `CUDA_DEVEL_IMAGE` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | |
| `CUDA_RUNTIME_IMAGE` | `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | |
| `NSIGHT_COMPUTE_PKG` | `nsight-compute-2025.2.1` | last Nsight Compute with Volta |
| `NSIGHT_SYSTEMS_PKG` | `nsight-systems-2025.2.1` | not in the CUDA 12.8 Ubuntu 24.04 apt repo (see below) |
| `NSIGHT_SYSTEMS_DEB_URL` | NVIDIA 2025.2.1 host+CLI `.deb` | used when the apt package is absent |
| `APP_MALLOC_TYPE` | `tcmalloc` | `tcmalloc` \| `jemalloc` \| `malloc` |
| `EXPERIMENTAL` | `0` (`Dockerfile.cuda` only) | CUDA 13.5 builds refuse unless `1` |

### `CUDA_EMBED_PTX`

AdaptiveCpp CUDA SMCP does not expose `--cuda-include-ptx`. Flags are
forwarded to clang via `ACPP_EXTRA_COMPILE_OPTIONS`.

| value | fatbin contents |
|---|---|
| `none` | SASS only (`--no-cuda-include-ptx=all`) |
| `highest` | SASS for each `sm_XX` plus PTX of the highest arch |
| `all` | SASS + PTX for every arch |
| `ptx-only` | rewrite `cuda:` → `cuda.explicit-multipass:` (clang `-S` PTX; driver JIT). Clang will not emit `compute_XX` virtual-arch fatbins. |

clang 20 never emits PTX ISA 8.8, even for `sm_120`. Expect PTX ISA 8.7.

## Backend per architecture

Kernel times below are das9201, milliseconds per 1e9 trials, basic_event /
gates, from nsys (issue #26). Lower is better.

| GPU | SMCP SASS | PTX via driver JIT | generic SSCP | production choice |
|---|---|---|---|---|
| Tesla V100S `sm_70` | **306** / 61 (28 regs) | 306 / 62 | 369–370 / 52–64 (64 regs) | **SMCP AOT** (`cuda:sm_70`) |
| RTX 3070 `sm_86` | 514 / 111 | 553 / 96 | **460** / 101–114 | generic SSCP |
| RTX 5060 Ti `sm_120` | 441 / 98 | **394** / 101 | (kernels not measured) | embedded PTX (driver JIT) |

Issue #29 therefore ships two Linux CUDA variants: **`generic`** (default
SSCP, covers `sm_86` / `sm_120` via runtime JIT) and **`cuda-sm70`** (SMCP
AOT for Volta). There is no specialised production image for `sm_86` or
`sm_120` yet.

On Volta the upgraded generic SSCP is **host-bound** (~37% GPU busy
fraction with a warm JIT cache; MC-phase wall ~3.46 s vs 1.15 s for SMCP
SASS on das9201). That fails issue #24's "≥ 0.9 GPU busy fraction"
criterion, so **`cuda-sm70` is the production variant on Volta**. That is
the "#24 or SMCP is used on that architecture" path; see issue #29.

`CUDA_FORCE_PTX_JIT=1` forces the driver to ignore embedded SASS and JIT
the embedded PTX. It is the right way to measure the PTX column. Forward
compat: an `sm_86` image with embedded PTX runs on `sm_120` via driver JIT.

## Smoke tests

Every runtime/profiler image installs `/usr/local/bin/mcscram-smoke`.
Each model is run under `timeout 60` with a fixed trial count, and the
script exits non-zero if `|z| > 4` versus the Aralia BDD reference.

| model | BDD reference |
|---|---|
| `chinese` | `1.17058e-3` |
| `baobab1` | `1.01708e-4` |
| `das9201` | `1.34237e-2` |
| `das9701` (optional) | `0.07447` |

```bash
docker run --rm --gpus device=0 --entrypoint bash mcscram:cuda12.8-sm70 \
  -lc 'timeout 60 /usr/local/bin/mcscram-smoke'

# equivalent, one model
docker run --rm --gpus device=0 --entrypoint bash IMG \
  -c 'timeout 60 scram-cli --monte-carlo --probability --pdag -c 2 --no-kn -N 1e8 -P wald /app/input/das9201.xml'
```

`SMOKE_N`, `SMOKE_TIMEOUT`, `SMOKE_Z_MAX`, `SMOKE_INPUT_DIR` override the
defaults. Run this on **both** `cuda-sm70` and the generic image on a
V100S; that is the #24 regression (the June 2026 generic-SSCP image
returned exactly 0 on several models).

## Nsight / driver compatibility

Host driver on the validation machines is **580** (CUDA 13.0). Profiling
needs root, `--cap-add SYS_ADMIN`, and an empty `LD_LIBRARY_PATH` so the
CUDA 12.8 compat `libcuda` does not shadow the host driver:

```bash
docker run --rm --gpus device=0 --user root --cap-add SYS_ADMIN \
  -e LD_LIBRARY_PATH= \
  --entrypoint bash mcscram:cuda12.8-sm70-prof -lc \
  'nsys profile -t cuda --force-overwrite=true -o /tmp/rep --stats=true \
     /usr/local/bin/scram-cli --monte-carlo --probability --pdag -c 2 --no-kn -N 1e8 -P wald /app/input/das9201.xml \
   && test -f /tmp/rep.nsys-rep'
```

The profiler stage sets `QT_QPA_PLATFORM=offscreen` and installs the
QdstrmImporter system libraries (GLX, xcb, NSS, …). Without those, `nsys
profile` on the slim runtime writes only a `.qdstrm` and never produces an
importable `.nsys-rep`.

| tool | version | driver | notes |
|---|---|---|---|
| Nsight Systems | **2025.2.x** (default) | 580 (CUDA 13.0) | records CUDA kernel rows on Volta |
| Nsight Systems | 2025.3.2 | 580 | no CUDA kernel rows on V100S |
| Nsight Systems | ≥ 2026.x | **≥ 595 (CUDA 13.2)** | `CUPTI_ERROR_NOT_INITIALIZED` on driver 580 |
| Nsight Compute | **2025.2.1** (default) | 580 | last release with Volta (`sm_70`) |
| Nsight Compute | ≥ 2025.3 | — | dropped Volta |

The CUDA 12.8 Ubuntu 24.04 apt repo has `nsight-systems-2025.1.3` and
`nsight-systems-2025.3.2` but not 2025.2.x. The profiler stage therefore
fetches NVIDIA's 2025.2.1 `.deb` when the apt package is absent. Override
with `NSIGHT_SYSTEMS_PKG` / `NSIGHT_SYSTEMS_DEB_URL`.

Inspect embedded SASS/PTX with `cuobjdump` from the profiler image:

```bash
docker run --rm --entrypoint bash mcscram:cuda12.8-sm70-prof -lc \
  'cuobjdump --list-elf --list-ptx /usr/local/lib/libmc.so'
```

## `Dockerfile.cuda` (CUDA 13.5) — blocked

Do not use this file. It is kept so the CUDA 13 experiment is not lost.

- clang 21 cannot parse CUDA 13.5 headers.
- clang 22 needs AdaptiveCpp LLVM 22 support (AdaptiveCpp PR #1986);
  `v25.10.0` errors on LLVM > 20 unless `ACPP_EXPERIMENTAL_LLVM=ON`, and
  even then does not compile against LLVM 22.
- CUDA 13 removed offline compilation for `sm_50`–`sm_72` (no Volta).

```bash
# fails immediately with the message above
docker build -f Dockerfile.cuda --target experimental-gate .

# opt-in (still rejects sm_70)
docker build -f Dockerfile.cuda --build-arg EXPERIMENTAL=1 --target scramruntime .
```
