#!/usr/bin/env bash
# Move distribution files larger than MAX_MB out of SRC_DIR into HOLD_DIR so the
# PyPI upload does not fail on the per-file size limit (100 MB by default).
# Held-back files are still attached to the GitHub Release. After PyPI raises
# the project's limit, set the repository variable PYPI_MAX_FILE_MB.
set -euo pipefail
src="${1:?src dir}"
hold="${2:?hold dir}"
max_mb="${3:?max MB}"
limit=$((max_mb * 1000 * 1000))

mkdir -p "${hold}"
shopt -s nullglob
kept=0
for f in "${src}"/*; do
  size=$(stat -c %s "${f}")
  if ((size > limit)); then
    echo "::warning::holding back $(basename "${f}") ($((size / 1000 / 1000)) MB > ${max_mb} MB PyPI limit)"
    mv "${f}" "${hold}/"
  else
    echo "upload $(basename "${f}") ($((size / 1000 / 1000)) MB)"
    kept=$((kept + 1))
  fi
done
if ((kept == 0)); then
  echo "::error::no distribution files left to upload"
  exit 1
fi
