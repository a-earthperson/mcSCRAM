#!/usr/bin/env bash
# Fail if any tracked non-LFS file is larger than 5 MiB (deferred from #39).
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
limit=$((5 * 1024 * 1024))
fail=0
count=0

while IFS= read -r -d '' f; do
  [[ -f "${f}" ]] || continue
  count=$((count + 1))
  if head -c 200 "${f}" | grep -q 'git-lfs.github.com/spec/v1'; then
    continue
  fi
  if [[ -d "${f}" ]]; then
    continue
  fi
  sz=$(wc -c < "${f}" | tr -d ' ')
  if (( sz > limit )); then
    printf 'FAIL %10s  %s\n' "${sz}" "${f}"
    fail=1
  fi
done < <(git ls-files -z)

if (( fail )); then
  echo "tracked non-LFS files exceed 5 MiB" >&2
  exit 1
fi
echo "ok: ${count} tracked files, none over 5 MiB (non-LFS)"
