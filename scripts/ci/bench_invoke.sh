#!/usr/bin/env bash
# Run the harness on the runner host (local docker, local GPU locks).
# Required env: TARGET, MODE (accuracy|perf), OUT
# A/B also needs BASE_IMAGE, HEAD_IMAGE. Accuracy needs HEAD_IMAGE.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${root}"
bash scripts/ci/bench-stage-input.sh

out="${OUT}"
md="${MD:-${out%.json}.md}"
mkdir -p "$(dirname "${out}")"

ctx=(--docker-context default --skip-validate)
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
