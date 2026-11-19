# Runtime images

Slim runtime images published to GHCR by [`.github/workflows/images.yml`](../.github/workflows/images.yml) (#35). They replace the manually built `ghcr.io/a-earthperson/mcscram:latest` (14.4 GB, oneAPI/L0/ROCm).

Image: **`ghcr.io/fuzzybit-org/fuzzybits-runtime`** (public). Packages are linked to this repo via `org.opencontainers.image.source`.

Compiler toolchain (private, not this image): [`docs/ci-toolchain.md`](ci-toolchain.md).

## Why FROM the toolchain image

Hosted `ubuntu-24.04` builds `docker/runtime.Dockerfile` **on top of**
`ghcr.io/fuzzybit-org/fuzzybits-toolchain:<ACPP_REF>-generic` (pulled with
`GITHUB_TOKEN` on the publish job). The toolchain already has LLVM 20,
AdaptiveCpp `full` (SSCP + clang CUDA SMCP), CUDA 12.8.1 devel, and Boost 1.89.

Re-running the LLVM/AdaptiveCpp stages on a 4-core hosted runner is tens of
minutes (AdaptiveCpp + Boost); a LABEL-only toolchain rebuild is seconds. The
product C++ build is ~8 min per preset with `BUILD_TESTING=OFF` (see the
toolchain doc). This workflow still compiles twice per variant — once for
`scram-cli`/`fzb` (non-SKBUILD; SKBUILD skips `fzb`) and once for the
scikit-build-core wheel — so a **cold** publish is roughly toolchain pull +
~16 min compile + slim assemble; a **warm** registry cache (`buildcache-<variant>`,
`mode=max`) should drop to the C++ rebuild plus a few minutes of COPY/pip.

The final image does **not** contain the toolchain. `runtime` is `ubuntu:24.04`
plus:

| contents | notes |
|---|---|
| `scram-cli`, `fzb` | `/usr/local/bin`; entrypoint wrapper: `docker run IMG` → scram-cli, `docker run IMG fzb …` → fzb |
| AdaptiveCpp runtime | `libacpp-rt`, `libacpp-common`, CUDA/OMP backends, `acpp-info` |
| SSCP JIT (`generic` only) | `libLLVM.so.20.1`, `libedit2`, `libdevice.10.bc`, `opt`/`llc`, `llvm-to-backend` |
| `fuzzybits` Python | wheel built in the same Dockerfile, `pip install` into the image; `import mcscram_native` shim; no `PYTHONPATH` / `LD_PRELOAD` |
| `mcscram-smoke` | chinese, baobab1, das9201 vs Aralia BDD refs; `timeout 60` each |
| CUDA | `libcudart` from the toolchain (CUDA 12.8.1). Driver (`libcuda.so.1`) comes from `--gpus`. Not `FROM nvidia/cuda:*-runtime` (that image is ~3 GB of unused cublas/cufft/…). |

Runs as non-root `coder` (uid 1000) with writable JIT cache
`ACPP_APPDB_DIR=/home/coder/.cache/fuzzybits/acpp`. `MALLOC_TYPE=malloc`.

Size budget: **generic ≤ ~4 GB**, **cuda-sm70 ≤ ~3.5 GB** (docker uncompressed).

## Tags

Let `SHA12` be the first 12 hex chars of the fuzzybits git SHA. Variant is
`generic` (default, SSCP) or `cuda12.8-sm70` (`ACPP_TARGETS=cuda:sm_70`).

| tag | when | mutability |
|---|---|---|
| `ci-<sha12>-<variant>` | push to `ci/**` | immutable (that SHA) |
| `sha-<sha12>-<variant>` | `develop`, schedule, `workflow_dispatch` on develop | immutable |
| `nightly-<variant>` | same as `sha-` | moves |
| `vX.Y.Z-<variant>` | git tag `vX.Y.Z` | immutable |
| `vX.Y.Z` | git tag, **generic only** | immutable |
| `latest` | git tag, **generic only** | moves |
| `buildcache-<variant>` | BuildKit registry cache (`mode=max`) | cache |

Pre-merge smoke on `ci/images` pulls `ci-<sha12>-<variant>` so it never races a moving `nightly-` tag.

## How to pull

```bash
# default generic (after a develop publish)
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic

# Volta AOT
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-cuda12.8-sm70

# pin a SHA
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:sha-<sha12>-generic
```

The package is public; no `docker login` is required. Self-hosted smoke jobs
pull this way (no `GITHUB_TOKEN`, no `packages: write`).

If a freshly created GHCR package is still private, flip
`ghcr.io/fuzzybit-org/fuzzybits-runtime` to public in the org package settings.

## How to run

GPU device mapping (xerxes Docker; same as gpu.yml / #34):

| host GPU | `--gpus` | image variant |
|---|---|---|
| Tesla V100S (`sm_70`) | `device=0` | `cuda12.8-sm70` **and** `generic` |
| RTX 3070 (`sm_86`) | `device=1` | `generic` |
| RTX 5060 Ti (`sm_120`, shelly-1) | `device=0` | `generic` |

```bash
IMG=ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic

# scram-cli
docker run --rm --gpus device=0 "$IMG" --help

# fzb
docker run --rm --gpus device=0 "$IMG" fzb compile /app/input/chinese.xml -o /tmp/chinese.pcfb
docker run --rm --gpus device=0 "$IMG" fzb run /tmp/chinese.pcfb --trials 1e8 --backend cuda

# in-image smoke (mcscram-smoke + fzb compile/run + Python import)
docker run --rm --gpus device=0 --entrypoint fuzzybits-image-smoke "$IMG"

# Python (no PYTHONPATH / LD_PRELOAD)
docker run --rm --gpus device=0 --entrypoint python3 "$IMG" -c \
  'import fuzzybits, mcscram_native; fuzzybits.init(); print(fuzzybits.list_devices())'

# AdaptiveCpp backends
docker run --rm --entrypoint acpp-info "$IMG" -l
```

`docker run IMG mcscram-smoke` does **not** work: the entrypoint forwards unknown
args to `scram-cli`. Use `--entrypoint mcscram-smoke` or `fuzzybits-image-smoke`.

Every Monte Carlo run inside the image is wrapped in `timeout 60`.

## Workflow

Triggers (never `pull_request` / `pull_request_target` / `workflow_run`):

- `push` to `develop` (path-filtered to engine/runtime sources, Dockerfiles,
  `docker/`, `toolchain.env`, CMake, this workflow)
- `push` to `ci/**` (always; tags are `ci-<sha12>-<variant>` only)
- git tags `v*`
- nightly `schedule` (`47 7 * * *` UTC)
- `workflow_dispatch`

Jobs:

| job | runner | permissions |
|---|---|---|
| `gate` | `ubuntu-24.04` | `contents: read` |
| `publish-<variant>` | `ubuntu-24.04` | `contents: read`, `packages: write`, `id-token: write`, `attestations: write` |
| `smoke-*` | self-hosted `[self-hosted, nvidia, sm_70\|sm_86\|sm_120]` | `contents: read` only; no login, no secrets used |

Publish uses `docker/build-push-action` with `--sbom=true --provenance=mode=max`
and registry cache `ghcr.io/fuzzybit-org/fuzzybits-runtime:buildcache-<variant>`.
Pins come from [`toolchain.env`](../toolchain.env) as `--build-arg`s.

Smoke concurrency groups (shared with gpu.yml, one physical GPU each):
`gpu-xerxes-v100s`, `gpu-xerxes-3070`, `gpu-shelly-5060ti`.

`actions/checkout` uses `persist-credentials: false`. Third-party actions are
pinned to full commit SHAs. Jobs have `timeout-minutes`. Smoke uses
`docker run --rm` only and does not install packages on the host. shelly-1 is a
production host — smoke does not `docker ps` / `docker system prune` / touch
other containers.

## Follow-ups

- **`-prof` variants** (Nsight Compute/Systems on top of the slim runtime).
  Nsight versions must stay compatible with the host driver (see
  [`docs/cuda-builds.md`](cuda-builds.md) and #26). Not in this workflow.
- **Release-tag flow** is implemented in the tagging logic (`vX.Y.Z-<variant>`,
  plus `vX.Y.Z` / `latest` for generic) but has not been exercised until a
  `v*` tag is pushed.
- **arm64** generic runtime: later.
- fuzzybits-bench should switch `stack.yml` / `scripts/publish-docker-images.sh`
  to these images (fuzzybit-org/fuzzybits-bench#1, #2). Until then the
  user-owned `ghcr.io/a-earthperson/mcscram*` packages stay.
