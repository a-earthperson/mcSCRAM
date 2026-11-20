#!/usr/bin/env bash
# Single-build accuracy + perf for one target, then merge. bench-history.yml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

TARGET="${TARGET:?}"
IMAGE="${IMAGE:?}"
VERSION="${VERSION:?}"
OUT_DIR="${OUT_DIR:?}"
DOCKER_CONTEXT="${DOCKER_CONTEXT:-default}"
COMMIT="${COMMIT:-}"
DRIVER="${DRIVER:-}"
RUNNER_NAME="${RUNNER_NAME:-}"
MODELS_SRC="${MODELS_SRC:-${ROOT}/input}"
MODELS_DIR="${MODELS_DIR:-/tmp/fuzzybits-bench/input}"
WORK_DIR="${WORK_DIR:-/tmp/fuzzybits-bench/work}"

mkdir -p "${OUT_DIR}" "${MODELS_DIR}" "${WORK_DIR}"
if [[ -d "${MODELS_SRC}/Aralia" ]]; then
  rm -rf "${MODELS_DIR}/Aralia"
  cp -a "${MODELS_SRC}/Aralia" "${MODELS_DIR}/Aralia"
fi

PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --no-project --python 3.12 python)
fi

run_mode() {
  local mode="$1"
  local out="${OUT_DIR}/${mode}.json"
  local md="${OUT_DIR}/${mode}.md"
  local rc=0
  "${PYTHON[@]}" -m benchmarks run \
    --skip-validate \
    --target "${TARGET}" \
    --mode "${mode}" \
    --image "${IMAGE}" \
    --docker-context "${DOCKER_CONTEXT}" \
    --out "${out}" \
    --md "${md}" \
    --commit "${COMMIT}" \
    --no-lock \
    || rc=$?
  echo "${mode}_rc=${rc}" >> "${OUT_DIR}/status.txt"
  return 0
}

run_mode accuracy
run_mode perf

merge_args=(--version "${VERSION}" --out "${OUT_DIR}/merged.json")
if [[ -s "${OUT_DIR}/accuracy.json" ]]; then
  merge_args+=(--accuracy "${OUT_DIR}/accuracy.json")
fi
if [[ -s "${OUT_DIR}/perf.json" ]]; then
  merge_args+=(--perf "${OUT_DIR}/perf.json")
fi
if [[ ${#merge_args[@]} -le 4 ]]; then
  echo "no harness JSON produced" >&2
  exit 1
fi
if [[ -n "${DRIVER}" ]]; then
  merge_args+=(--driver "${DRIVER}")
fi
if [[ -n "${RUNNER_NAME}" ]]; then
  merge_args+=(--runner-name "${RUNNER_NAME}")
fi
"${PYTHON[@]}" -m benchmarks.history merge "${merge_args[@]}"
test -s "${OUT_DIR}/merged.json"
