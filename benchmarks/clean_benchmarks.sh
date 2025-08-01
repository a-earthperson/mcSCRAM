#!/usr/bin/env bash
# Remove benchmark artefacts produced by bench_scram and other benchmarking utilities.
#
# 1. Deletes the top-level CSV summaries created by post-processing: convergence.csv and compiler.csv
# 2. For every `prepend_path` listed in benchmarks/input.csv it deletes *all* .csv files inside that
#    directory (relative to the benchmarks directory). This conveniently cleans the per-model
#    convergence CSV files generated by SCRAM.
#
# Usage: run from anywhere inside the repository
#    bash benchmarks/clean_benchmarks.sh
#
# The script is intentionally idempotent and will not error if files are already missing.

set -euo pipefail

BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INPUT_CSV="${BENCH_DIR}/input.csv"

# Local artefacts in benchmarks/
for f in "${BENCH_DIR}/convergence.csv" "${BENCH_DIR}/compiler.csv"; do
  if [[ -f "${f}" ]]; then
    echo "Removing ${f}"
    rm -f "${f}"
  else
    echo "Not found (skipped): ${f}"
  fi
done

# Verify input.csv exists
if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "input.csv not found: ${INPUT_CSV}" >&2
  exit 1
fi

# Extract unique prepend_path values (skip header), trim potential whitespace
mapfile -t PREPEND_PATHS < <(tail -n +2 "${INPUT_CSV}" | cut -d',' -f1 | sed 's/[[:space:]]//g' | sort -u)

for path in "${PREPEND_PATHS[@]}"; do
  # Ignore empty fields (defensive in case of malformed CSV rows)
  [[ -z "${path}" ]] && continue

  TARGET_DIR="${BENCH_DIR}/${path}"
  # Ensure path ends with a slash when originally present; users often include it in the CSV.
  TARGET_DIR="${TARGET_DIR%/}"

  if [[ -d "${TARGET_DIR}" ]]; then
    shopt -s nullglob
    CSV_FILES=("${TARGET_DIR}"/*.csv)
    shopt -u nullglob

    if (( ${#CSV_FILES[@]} > 0 )); then
      for csv in "${CSV_FILES[@]}"; do
        echo "Removing ${csv}"
        rm -f "${csv}"
      done
    else
      echo "No .csv files found in ${TARGET_DIR}"
    fi
  else
    echo "Directory listed in input.csv does not exist: ${TARGET_DIR} (skipped)"
  fi
done

echo "Cleanup complete."
