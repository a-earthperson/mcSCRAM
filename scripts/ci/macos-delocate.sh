#!/usr/bin/env bash
# delocate repair keeping AdaptiveCpp's hipSYCL/ tree next to libacpp-rt.
set -euo pipefail

src_dir="${1:?wheelhouse}"
dest_dir="${2:?dest}"
mkdir -p "${dest_dir}"

# --lib-sdir lib: extra deps (libxml2, Homebrew libc++, …) land next to
# libacpp-rt so @loader_path/lib resolves both. Excludes keep AdaptiveCpp's
# plugin/JIT tree from being flattened out of hipSYCL/.
excludes=(
  -e hipSYCL
  -e libacpp-rt
  -e libacpp-common
  -e libLLVM
  -e liblld
  -e libomp.dylib
  -e libscram.dylib
)

shopt -s nullglob
wheels=("${src_dir}"/*.whl)
if ((${#wheels[@]} == 0)); then
  echo "no wheels in ${src_dir}" >&2
  exit 1
fi

for whl in "${wheels[@]}"; do
  echo "delocate-wheel --lib-sdir lib --require-archs arm64 ${whl}"
  delocate-wheel -v \
    --lib-sdir lib \
    --require-archs arm64 \
    --no-sanitize-rpaths \
    --wheel-dir "${dest_dir}" \
    "${excludes[@]}" \
    "${whl}"
  shopt -s nullglob
  repaired=("${dest_dir}"/*.whl)
  if ((${#repaired[@]})); then
    echo "delocate-listdeps ${repaired[0]}"
    delocate-listdeps --all "${repaired[0]}" || true
  fi
done
