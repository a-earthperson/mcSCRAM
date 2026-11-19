#!/usr/bin/env bash
# auditwheel repair with AdaptiveCpp layout preserved under fuzzybits/lib/.
set -euo pipefail

src_dir="${1:?wheelhouse}"
dest_dir="${2:?dest}"
mkdir -p "${dest_dir}"

# --lib-sdir lib: extra deps (libxml2, …) land next to libacpp-rt so the
# extension's $ORIGIN/lib RPATH finds both. Excludes keep AdaptiveCpp's
# plugin/JIT tree from being rewritten into a parallel .libs directory.
excludes=(
  --exclude libcuda.so.1
  --exclude libcuda.so
  --exclude libnvidia-ml.so.1
  --exclude libnvidia-ptxjitcompiler.so.1
  --exclude libacpp-rt.so
  --exclude libacpp-common.so
  --exclude librt-backend-cuda.so
  --exclude librt-backend-omp.so
  --exclude libllvm-to-ptx.so
  --exclude libllvm-to-backend.so
  --exclude libllvm-to-host.so
  --exclude libLLVM.so.20.1
  --exclude libLLVM.so
  --exclude libcudart.so.12
  --exclude libcudart.so
  --exclude libomp.so
  --exclude libomp.so.5
  --exclude libscram.so
  --exclude libmc.so
  --exclude libxml2.so.16
  --exclude libxml2.so
  --exclude libatomic.so.1
  --exclude libatomic.so
)

shopt -s nullglob
wheels=("${src_dir}"/*.whl)
if ((${#wheels[@]} == 0)); then
  echo "no wheels in ${src_dir}" >&2
  exit 1
fi

for whl in "${wheels[@]}"; do
  echo "auditwheel repair --lib-sdir lib ${whl}"
  auditwheel repair --lib-sdir lib "${excludes[@]}" -w "${dest_dir}" "${whl}"
  echo "auditwheel show repaired"
  shopt -s nullglob
  repaired=("${dest_dir}"/*.whl)
  if ((${#repaired[@]})); then
    auditwheel show "${repaired[0]}" || true
  fi
done
