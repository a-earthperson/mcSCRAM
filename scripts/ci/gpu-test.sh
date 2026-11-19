#!/usr/bin/env bash
# One GPU preset: configure+product, tests, ctest -L mc, smoke, ccache stats.
# Requires DOCKER_GPUS, FUZZYBITS_CTR, PRESET, and a GHCR login.
set -euo pipefail
preset="${1:?preset}"
export PRESET="${preset}"
export BUILD_TESTING=ON
gpus="${DOCKER_GPUS:?DOCKER_GPUS}"
name="${FUZZYBITS_CTR:?FUZZYBITS_CTR}"
run() {
  scripts/ci/docker-toolchain-run.sh --gpus "${gpus}" --name "${name}" -- scripts/ci/inner-build.sh "${preset}" "$1"
}
run configure-product
run tests
run ctest-mc
run smoke
run ccache-stats
