#!/usr/bin/env bash
# Move distribution files larger than MAX_MB out of SRC_DIR into HOLD_DIR so the
# PyPI upload does not fail on the per-file size limit (100 MB by default).
# Held-back files are still attached to the GitHub Release. After PyPI raises
# the project's limit, set the repository variable PYPI_MAX_FILE_MB.
#
# Prints a table of every file (name, bytes, MB, keep/hold) and the total size
# of the release set. PyPI's per-project cap is 10 GB (#101).
set -euo pipefail
src="${1:?src dir}"
hold="${2:?hold dir}"
max_mb="${3:?max MB}"
limit=$((max_mb * 1000 * 1000))

mkdir -p "${hold}"
shopt -s nullglob
kept=0
held=0
total=0
printf '%-80s %12s %8s %s\n' "file" "bytes" "MB" "status"
for f in "${src}"/*; do
  [[ -f "${f}" ]] || continue
  size=$(stat -c %s "${f}")
  total=$((total + size))
  mb=$((size / 1000 / 1000))
  name=$(basename "${f}")
  if ((size > limit)); then
    echo "::warning::holding back ${name} (${mb} MB > ${max_mb} MB PyPI limit)"
    mv "${f}" "${hold}/"
    printf '%-80s %12d %8d HOLD\n' "${name}" "${size}" "${mb}"
    held=$((held + 1))
  else
    printf '%-80s %12d %8d keep\n' "${name}" "${size}" "${mb}"
    kept=$((kept + 1))
  fi
done
total_mb=$(awk -v t="${total}" 'BEGIN { printf "%.1f", t / 1000000 }')
echo "kept=${kept} held=${held} total_bytes=${total} total_mb=${total_mb} pypi_project_limit_gb=10"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## wheel size gate (limit ${max_mb} MB)"
    echo "- kept: ${kept}"
    echo "- held: ${held}"
    echo "- total: ${total_mb} MB (${total} bytes)"
    echo "- PyPI project limit: 10 GB (#101)"
  } >> "${GITHUB_STEP_SUMMARY}"
fi
if ((kept == 0)); then
  echo "::error::no distribution files left to upload"
  exit 1
fi
