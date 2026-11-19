#!/usr/bin/env bash
# Per-runner GPU mapping diagnostic for fuzzybits gpu.yml.
# Prints host visibility, then whether docker --gpus works and which
# device index the NVIDIA toolkit maps into the container.
#
# Safe on shelly: only `docker run --rm` of this job's own containers.
# Public image: no GHCR token.
set -euo pipefail

IMAGE="${GPU_DIAG_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu24.04}"
NAME="fuzzybits-gpu-diag-${GITHUB_RUN_ID:-local}-${GITHUB_JOB:-manual}"

cleanup() {
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

banner() {
  printf '\n======== %s ========\n' "$*"
}

banner "runner"
echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "whoami=$(whoami)"
echo "pwd=$(pwd)"
echo "GITHUB_RUNNER_NAME=${GITHUB_RUNNER_NAME:-}"
echo "GITHUB_JOB=${GITHUB_JOB:-}"
echo "RUNNER_NAME=${RUNNER_NAME:-}"
echo "RUNNER_TRACKING_ID=${RUNNER_TRACKING_ID:-}"

banner "env"
for v in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES \
         NVIDIA_REQUIRE_CUDA CUDA_DEVICE_ORDER; do
  eval "val=\${$v-UNSET}"
  echo "$v=${val}"
done

banner "host nvidia-smi -L"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || echo "nvidia-smi -L failed rc=$?"
  echo "--- nvidia-smi (summary) ---"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total \
    --format=csv || true
else
  echo "nvidia-smi not on PATH"
fi

banner "host /dev/nvidia*"
ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia*"

banner "docker"
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' || docker version
docker info --format 'runtime={{json .Runtimes}} default={{.DefaultRuntime}} nvidiagpus={{.NCPU}}' || true
# Do not `docker ps`: shelly hosts the production swarm.

pull_ok=0
banner "docker pull ${IMAGE}"
if docker pull "${IMAGE}"; then
  pull_ok=1
else
  echo "pull failed; trying without pull if the image is already local"
  docker image inspect "${IMAGE}" >/dev/null 2>&1 && pull_ok=1 || true
fi

run_probe() {
  local spec="$1"
  banner "docker run --rm --gpus ${spec}"
  local rc=0
  docker run --rm --name "${NAME}" --gpus "${spec}" --entrypoint bash "${IMAGE}" -lc '
    set -e
    echo "container hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
    echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-UNSET}"
    echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-UNSET}"
    echo "--- /dev/nvidia* ---"
    ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia*"
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L
    echo "--- nvidia-smi query ---"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total --format=csv
  ' || rc=$?
  echo "rc=${rc} spec=${spec}"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  return 0
}

if [[ "${pull_ok}" -eq 1 ]]; then
  run_probe all
  run_probe device=0
  run_probe device=1
else
  echo "skipping docker --gpus probes: image unavailable"
fi

banner "done"
echo "gpu-diagnose complete"
