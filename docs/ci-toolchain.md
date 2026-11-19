# CI toolchain images

Prebuilt compiler images for hosted-runner CI (#33). Building LLVM + AdaptiveCpp
from scratch is tens of minutes; CI of fuzzybits itself should start from these
images and finish the C++ build in under 10 minutes on a hosted-runner-sized
machine (`--cpus=4 -m 16g`).

Image: **`ghcr.io/fuzzybit-org/fuzzybits-toolchain`** (private, linked to this
repo via `org.opencontainers.image.source`). Packages stay private so only
fuzzybits workflows (and org members) can pull them with `GITHUB_TOKEN`.

Metal tarball cache is deferred; see [#30](https://github.com/fuzzybit-org/fuzzybits/issues/30).

## What is in the image

One image serves **both** Linux presets. AdaptiveCpp is built with
`ACPP_COMPILER_FEATURE_PROFILE=full`, which is SSCP (generic JIT) **and** clang
CUDA SMCP (`cuda:sm_70`). A `minimal` AdaptiveCpp cannot configure `generic`;
`full` can configure both, so the image is published under both tags rather
than building twice.

| layer | contents |
|---|---|
| base | `nvidia/cuda:12.8.1-devel-ubuntu24.04` (from `toolchain.env`) |
| build deps | cmake, ninja, git, python3-dev, libxml2-dev, libomp, tcmalloc/jemalloc headers |
| LLVM | clang/llvm/lld/libomp **20** from apt.llvm.org (not built from source) |
| AdaptiveCpp | `v25.10.0` (`ACPP_COMMIT`), CUDA backend + OpenMP, **no** OpenCL / ROCm / Level Zero / oneAPI |
| Boost | 1.89.0 static+PIC, CMake package in `/usr/local` (same URL/SHA as `toolchain.env`) |

`cmake --preset generic` and `cmake --preset cuda-sm70` both work:

```bash
docker run --rm --cpus=4 -m 16g \
  ghcr.io/fuzzybit-org/fuzzybits-toolchain:v25.10.0-generic \
  bash -c 'cmake --preset generic -DBUILD_TESTING=OFF && cmake --build --preset generic'
```

`include(CTest)` defaults `BUILD_TESTING=ON` (the #63 suite). The product C++
build (scram-cli, libmc, Python module, fzb) meets the < 10 min budget with
`-DBUILD_TESTING=OFF`. #34 should use that flag on the compiler job and run
the suite in a separate job.

## Verification (xerxes, 2026-11-18)

`docker --context xerxes` (48-CPU daemon). C++ builds used `--cpus=4 -m 16g`
as the hosted-runner proxy from #33. Image
`ghcr.io/fuzzybit-org/fuzzybits-toolchain` is **12.5 GB**. One image ID is
tagged as both `-generic` and `-cuda12.8`. `acpp-info -l` reports OpenMP +
CUDA only; `plugin-with-sscp-compiler=true`; no oneAPI / Level Zero / ROCm
packages.

| step | wall |
|---|---|
| toolchain image, LLVM+AdaptiveCpp cache hit, Boost 1.89 compile | 70–109 s |
| LABEL-only rebuild (`FUZZYBITS_REVISION` / `TOOLCHAIN_SHA` change) | 4 s |
| `cmake --preset generic -DBUILD_TESTING=OFF && cmake --build` | **452 s (7.5 min)** |
| `cmake --preset cuda-sm70 -DBUILD_TESTING=OFF && cmake --build` | **486 s (8.1 min)** |
| same presets, default `BUILD_TESTING=ON` (#63 tests + Catch2) | 850 s / 833 s |

LLVM is installed from apt.llvm.org (not built from source), so a cold
GitHub-hosted `ubuntu-24.04` image job is tens of minutes (AdaptiveCpp + Boost
on 2 cores), not the 6 h limit. If a future pin requires compiling LLVM from
source, keep this workflow on GitHub-hosted only if it still fits; otherwise
propose a `push`/`workflow_dispatch` self-hosted builder (never `pull_request`).

Local helper: `docker/verify-toolchain.sh` (does not push).

CI copies the fuzzybits source into a thin overlay (`docker/toolchain-verify.Dockerfile`)
or bind-mounts it. The toolchain image itself does **not** contain fuzzybits.

Inspect backends (expect OpenMP + CUDA only; `plugin-with-sscp-compiler: true`):

```bash
docker run --rm --entrypoint bash IMG -lc \
  'acpp-info -l; cat /usr/local/etc/AdaptiveCpp/acpp-core.json'
```

`acpp-core.json` may mention `default-rocm-path: /opt/rocm`; that path is
absent and no ROCm/oneAPI/Level Zero packages are installed.

## Tags

Pins come from [`toolchain.env`](../toolchain.env). Let `ACPP_REF` be the
AdaptiveCpp ref (currently `v25.10.0`) and `TOOLCHAIN_SHA` the first 12 hex
chars of `sha256sum toolchain.env`.

| tag | mutability | meaning |
|---|---|---|
| `<ACPP_REF>-generic` | moves when the image is rebuilt | SSCP + CUDA SMCP toolchain |
| `<ACPP_REF>-cuda12.8` | moves | **same image** as `-generic` (clang CUDA SMCP for `cuda-sm70`) |
| `<ACPP_REF>-generic-<sha12>` | immutable | content-addressed by `toolchain.env` |
| `<ACPP_REF>-cuda12.8-<sha12>` | immutable | same digest as the generic hash tag |
| `buildcache-generic` | cache | BuildKit registry cache (`mode=max`) |

Labels (also set by the workflow):

| label | value |
|---|---|
| `org.opencontainers.image.source` | `https://github.com/fuzzybit-org/fuzzybits` |
| `org.opencontainers.image.revision` | fuzzybits git commit that built the image |
| `org.opencontainers.image.version` | `ACPP_REF` |
| `dev.fuzzybits.acpp.commit` | `ACPP_COMMIT` from `toolchain.env` |
| `dev.fuzzybits.toolchain.sha` | `TOOLCHAIN_SHA` |
| `dev.fuzzybits.acpp.profile` | `full` |

## How images are built

Workflow: [`.github/workflows/toolchain.yml`](../.github/workflows/toolchain.yml).

Triggers (GitHub-hosted `ubuntu-24.04` only; never `pull_request`, never
self-hosted):

- `push` to `develop` when `toolchain.env`, `docker/toolchain.Dockerfile`,
  `Dockerfile.cuda12.8`, or the workflow file change
- `workflow_dispatch`

The job reads `toolchain.env` into `--build-arg`s (single source of truth),
logs into GHCR with `GITHUB_TOKEN`, and builds

```text
docker/toolchain.Dockerfile --target toolchain
```

with BuildKit registry cache
`ghcr.io/fuzzybit-org/fuzzybits-toolchain:buildcache-generic`.

Top-level `permissions: contents: read`. The publish job adds `packages: write`.
`actions/checkout` uses `persist-credentials: false`. Third-party actions are
pinned to full commit SHAs.

Disk: the job deletes unused hosted-image trees before building (dotnet,
Android SDK, GHC, CodeQL, extra hostedtoolcache languages, JVMs, Miniconda,
PowerShell, Swift). LLVM is **not** built from source (apt.llvm.org), so the
job stays well under the 6 h limit; AdaptiveCpp + Boost compile on 2 cores is
tens of minutes.

Local (xerxes):

```bash
set -a && . ./toolchain.env && set +a
TOOLCHAIN_SHA=$(sha256sum toolchain.env | cut -c1-12)
docker --context xerxes build -f docker/toolchain.Dockerfile --target toolchain \
  --build-arg ACPP_REF --build-arg ACPP_REPO --build-arg ACPP_COMMIT \
  --build-arg CLANG_VERSION --build-arg CUDA_DEVEL_IMAGE \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE \
  --build-arg BOOST_URL --build-arg BOOST_SHA256 --build-arg BOOST_VERSION \
  --build-arg FUZZYBITS_REVISION="$(git rev-parse HEAD)" \
  --build-arg TOOLCHAIN_SHA \
  -t ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic \
  -t ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-cuda12.8 \
  .
```

Do not `docker push` from a laptop; publishing is the workflow after merge.

## How CI consumes them

A later build/test workflow (#34) should:

1. `docker login ghcr.io` with `GITHUB_TOKEN` (`packages: read` is enough to
   pull a private package linked to this repo).
2. Pull
   `ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic`
   (or the immutable `-${TOOLCHAIN_SHA}` tag).
3. Copy or mount the fuzzybits source and run
   `cmake --preset <variant> && cmake --build --preset <variant>`.

`cmake/FindBoost.cmake` uses a CMake-package Boost ≥ `BOOST_MIN_VERSION` when
present (the toolchain image) and FetchContent 1.89.0 otherwise.

## How to bump pins

1. Edit [`toolchain.env`](../toolchain.env) only. Do not fork versions into
   Dockerfiles or the workflow YAML.
2. `ACPP_REF` / `ACPP_COMMIT` together. After changing the AdaptiveCpp tag,
   set `ACPP_COMMIT` to `git rev-parse` of that tag.
3. `CLANG_VERSION` must stay on a release AdaptiveCpp supports and that the
   CUDA devel image's headers accept (clang 20 ↔ CUDA 12.8).
4. `CUDA_DEVEL_IMAGE` / `CUDA_RUNTIME_IMAGE` are the `FROM` lines.
5. `BOOST_URL` / `BOOST_SHA256` / `BOOST_VERSION` must stay ≥ `BOOST_MIN_VERSION`
   (1.87/1.88 break AdaptiveCpp CUDA SMCP).
6. Merge to `develop`. The toolchain workflow rebuilds and retags
   `<ACPP_REF>-generic` / `-cuda12.8`. Previous immutable
   `<ACPP_REF>-*-<oldsha>` tags remain.
7. If you add a Boost library to `CMakeLists.txt` `BOOST_INCLUDE_LIBRARIES`,
   add it to the `BOOST_INCLUDE_LIBRARIES` ARG in `docker/toolchain.Dockerfile`
   as well (or CI will CONFIG-find Boost and then fail linking the new
   component).
