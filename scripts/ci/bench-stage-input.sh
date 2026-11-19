#!/usr/bin/env bash
# Copy dataset models onto the docker host path the harness mounts.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="${FUZZYBITS_BENCH_INPUT:-/tmp/fuzzybits-bench/input}"
work="${FUZZYBITS_BENCH_WORK:-/tmp/fuzzybits-bench/work}"
mkdir -p "${dest}" "${work}"
if [[ ! -d "${root}/input/Aralia" ]]; then
  echo "error: missing ${root}/input/Aralia" >&2
  exit 1
fi
rm -rf "${dest}/Aralia"
cp -a "${root}/input/Aralia" "${dest}/Aralia"
if [[ -d "${root}/input/generic-pwr-model" ]]; then
  rm -rf "${dest}/generic-pwr-model"
  cp -a "${root}/input/generic-pwr-model" "${dest}/generic-pwr-model"
fi
echo "staged ${dest} (Aralia + optional generic-pwr-model)"
