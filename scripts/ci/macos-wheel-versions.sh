#!/usr/bin/env bash
# Build and/or smoke fuzzybits wheels for PYTHON_VERSIONS (default 3.12).
# Usage: macos-wheel-versions.sh [build|smoke|all]
# Requires uv, AdaptiveCpp prefix env (ACPP_PREFIX, LLVM_PREFIX, OMP_PREFIX).
set -euo pipefail
export COPYFILE_DISABLE=1

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${root}"

cmd="${1:-all}"
read -r -a pythons <<< "${PYTHON_VERSIONS:-3.12}"
if ((${#pythons[@]} == 0)); then
  echo "error: PYTHON_VERSIONS is empty" >&2
  exit 2
fi

: "${GITHUB_WORKSPACE:=${root}}"
: "${RUNNER_TEMP:=/tmp}"
mkdir -p dist
timing_file="${RUNNER_TEMP}/fuzzybits-wheel-timings.txt"

record() {
  echo "$1" | tee -a "${timing_file}"
}

case "${cmd}" in
  build|all)
    : >"${timing_file}"
    uv python install "${pythons[@]}"
    for py in "${pythons[@]}"; do
      echo "=== CPython ${py} wheel ($(uname -m)) ==="
      start=$(date +%s)
      venv="${RUNNER_TEMP}/fb-build-${py}"
      rm -rf "${venv}"
      uv venv --seed --python "${py}" "${venv}"
      # shellcheck disable=SC1091
      . "${venv}/bin/activate"
      python -m pip install -U pip
      python -m pip install 'scikit-build-core>=0.10' 'pybind11>=3.1' ninja cmake build 'delocate>=0.13'
      tag="cp${py//./}"
      wheelhouse="${RUNNER_TEMP}/wheelhouse-${tag}"
      raw="${RUNNER_TEMP}/dist-raw-${tag}"
      rm -rf "${wheelhouse}" "${raw}"
      mkdir -p "${wheelhouse}" "${raw}"
      scripts/ci/macos-wheel-build.sh "${raw}" "${wheelhouse}"
      shopt -s nullglob
      built=("${raw}"/fuzzybits-*.whl)
      if ((${#built[@]} != 1)); then
        echo "error: expected one wheel in ${raw}, got ${#built[@]}" >&2
        ls -la "${raw}" >&2
        exit 1
      fi
      cp "${built[0]}" dist/
      deactivate
      record "wheel_${tag}_seconds=$(( $(date +%s) - start ))"
    done
    ls -lah dist
    ;;
  smoke)
    ;;
  *)
    echo "usage: $0 [build|smoke|all]" >&2
    exit 2
    ;;
esac

if [[ "${cmd}" == "smoke" || "${cmd}" == "all" ]]; then
  uv python install "${pythons[@]}"
  for py in "${pythons[@]}"; do
    echo "=== CPython ${py} smoke ($(uname -m)) ==="
    start=$(date +%s)
    venv="${RUNNER_TEMP}/fb-venv-${py}"
    rm -rf "${venv}"
    uv venv --seed --python "${py}" "${venv}"
    tag="cp${py//./}"
    shopt -s nullglob
    whls=(dist/fuzzybits-*-"${tag}"-*.whl)
    if ((${#whls[@]} != 1)); then
      echo "error: expected one ${tag} wheel in dist/, got ${#whls[@]}" >&2
      ls -la dist >&2
      exit 1
    fi
    "${venv}/bin/python" -m pip install -U pip
    "${venv}/bin/python" -m pip install "${whls[0]}" pytest
    "${venv}/bin/python" scripts/ci/macos-smoke.py \
      --wheel "${whls[0]}" \
      --repo "${GITHUB_WORKSPACE}"
    record "smoke_${tag}_seconds=$(( $(date +%s) - start ))"
  done
fi

if [[ -n "${GITHUB_ENV:-}" && -f "${timing_file}" ]]; then
  cat "${timing_file}" >> "${GITHUB_ENV}"
fi

