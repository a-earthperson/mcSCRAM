#!/usr/bin/env bash
# Pull the pinned GHCR toolchain image and run a command inside it.
# Source is bind-mounted at /src. Caller must already `docker login ghcr.io`
# when the package is private (hosted: packages:read; self-hosted: same,
# documented exception — no other secrets).
#
# Usage:
#   scripts/ci/docker-toolchain-run.sh [--gpus SPEC] [--name NAME] -- cmd...
set -euo pipefail

gpus=""
name="fuzzybits-ci-${GITHUB_RUN_ID:-local}-${GITHUB_JOB:-job}-${GITHUB_RUN_ATTEMPT:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpus="${2:?}"
      shift 2
      ;;
    --name)
      name="${2:?}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown flag $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--gpus SPEC] -- command..." >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${root}"
set -a
# shellcheck disable=SC1091
. ./toolchain.env
set +a
sha="$(sha256sum toolchain.env | cut -c1-12)"
image="${TOOLCHAIN_IMAGE:-ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic-${sha}}"

echo "toolchain image=${image}"
ensure_image() {
  local ref="$1"
  # Prefer a local copy: later pulls in the same job can 401 even after
  # docker/login-action (token lifetime / rate), and GHCR is private.
  if docker image inspect "${ref}" >/dev/null 2>&1; then
    echo "using local ${ref}"
    image="${ref}"
    return 0
  fi
  if docker pull "${ref}"; then
    image="${ref}"
    return 0
  fi
  return 1
}
if ! ensure_image "${image}"; then
  fallback="ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic"
  echo "immutable tag missing; falling back to ${fallback}"
  if ! ensure_image "${fallback}"; then
    echo "error: cannot pull or find ${image} or ${fallback}" >&2
    exit 1
  fi
fi

run_args=(
  --rm
  --name "${name}"
  -v "${root}:/src"
  -w /src
  -e CCACHE_DIR=/src/.ccache
  -e CCACHE_MAXSIZE=2G
  -e CCACHE_COMPILERCHECK=content
  -e CCACHE_COMPILERTYPE=clang
  -e CCACHE_BASEDIR=/src
  -e CCACHE_NOHASHDIR=1
  -e ACPP_APPDB_DIR=/src/.cache/acpp
  -e CUDA_CACHE_PATH=/src/.cache/cuda
  -e BUILD_TESTING="${BUILD_TESTING:-ON}"
  -e PRESET="${PRESET:-generic}"
  -e ACPP_VISIBILITY_MASK="${ACPP_VISIBILITY_MASK:-}"
  -e ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}"
  -e ACPP_PERSISTENT_RUNTIME="${ACPP_PERSISTENT_RUNTIME:-1}"
  -e SPINE_BACKEND="${SPINE_BACKEND:-}"
  -e SPINE_TRIALS="${SPINE_TRIALS:-}"
  -e SPINE_TIMEOUT="${SPINE_TIMEOUT:-}"
  -e SPINE_Z_MAX="${SPINE_Z_MAX:-}"
  -e SPINE_INPUT_DIR="${SPINE_INPUT_DIR:-}"
  --entrypoint bash
)
if [[ -n "${gpus}" ]]; then
  echo "docker --gpus ${gpus}"
  run_args+=(--gpus "${gpus}")
fi

# --rm so a job timeout does not leave a GPU-resident container. Inner MC
# runs still use `timeout 60` (CTest TIMEOUT / smoke.sh / spine.sh) inside
# the container.
docker run "${run_args[@]}" "${image}" "$@"
