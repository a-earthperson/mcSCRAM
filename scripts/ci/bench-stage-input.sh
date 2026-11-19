#!/usr/bin/env bash
# Copy Aralia models onto the docker host path the harness mounts.
# Per-job dest (FUZZYBITS_BENCH_INPUT) so V100S and RTX 3070 on the same
# xerxes host do not race on /tmp/fuzzybits-bench/input.
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
echo "staged ${dest}/Aralia"
