#!/usr/bin/env bash
# Run the harness on the runner host (local docker, local GPU locks).
# Required env: TARGET, MODE (accuracy|perf), OUT
# A/B also needs BASE_IMAGE, HEAD_IMAGE. Accuracy needs HEAD_IMAGE.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${root}"
job="${GITHUB_JOB:-local}"
export FUZZYBITS_BENCH_INPUT="${FUZZYBITS_BENCH_INPUT:-/tmp/fuzzybits-bench/${job}/input}"
export FUZZYBITS_BENCH_WORK="${FUZZYBITS_BENCH_WORK:-/tmp/fuzzybits-bench/${job}/work}"
bash scripts/ci/bench-stage-input.sh

out="${OUT}"
md="${MD:-${out%.json}.md}"
mkdir -p "$(dirname "${out}")"

ctx=(
  --docker-context default
  --skip-validate
  --models-dir "${FUZZYBITS_BENCH_INPUT}"
  --work-dir "${FUZZYBITS_BENCH_WORK}"
  --guard-retries "${GUARD_RETRIES:-20}"
)
if [[ "${MODE}" == "accuracy" ]]; then
  python3 -m benchmarks run \
    --target cpu \
    --mode accuracy \
    --image "${HEAD_IMAGE}" \
    --out "${out}" \
    --md "${md}" \
    "${ctx[@]}" \
    --commit "${HEAD_SHA:-${GITHUB_SHA:-}}"
else
  extra=()
  if [[ -n "${INJECT:-}" ]]; then
    extra+=(--inject-regression "${INJECT}")
  fi
  # GPU targets default to the workflow's GPU_PERF_IDS; CPU keeps the manifest perf set.
  ids_list="${IDS:-}"
  if [[ -z "${ids_list}" && "${TARGET}" != "cpu" ]]; then
    ids_list="${GPU_PERF_IDS:-}"
  fi
  if [[ -n "${ids_list}" ]]; then
    read -r -a ids <<< "${ids_list}"
    extra+=(--ids "${ids[@]}")
  fi
  [[ -n "${METRIC:-}" ]] && extra+=(--metric "${METRIC}")
  [[ -n "${SECONDS_PER_RUN:-}" ]] && extra+=(--seconds "${SECONDS_PER_RUN}")
  [[ -n "${BUDGET_S:-}" ]] && extra+=(--budget-s "${BUDGET_S}")
  [[ -n "${ATTEMPTS:-}" ]] && extra+=(--attempts "${ATTEMPTS}")
  python3 -m benchmarks ab \
    --target "${TARGET}" \
    --mode perf \
    --base-image "${BASE_IMAGE}" \
    --head-image "${HEAD_IMAGE}" \
    --base "${BASE_SHA:-}" \
    --head "${HEAD_SHA:-}" \
    --out "${out}" \
    --md "${md}" \
    --rounds "${ROUNDS:-6}" \
    "${ctx[@]}" \
    "${extra[@]}"
fi
