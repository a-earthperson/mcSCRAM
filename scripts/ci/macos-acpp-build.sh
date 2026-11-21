#!/usr/bin/env bash
# Build AdaptiveCpp into ACPP_PREFIX. WITH_METAL_BACKEND=ON|OFF (default ON).
# Usage: macos-acpp-build.sh [--check]
#   --check  exit 0 if ACPP_PREFIX works and matches the pins, 1 otherwise.
set -euo pipefail

test -n "${ACPP_PREFIX:-}"
test -n "${ACPP_REPO:-}"
test -n "${LLVM_PREFIX:-}"
test -n "${OMP_PREFIX:-}"
test -n "${ACPP_LLD_PATH:-}"
: "${RUNNER_TEMP:=/tmp}"
metal="${WITH_METAL_BACKEND:-ON}"
ref="${ACPP_REF_TO_BUILD:-${ACPP_REF_METAL:?}}"

metal_cpp=""
if [[ "${metal}" == "ON" ]]; then
  metal_cpp="${METAL_CPP_VERSION:?}"
fi
llvm_version="$("${LLVM_PREFIX}/bin/llvm-config" --version)"
stamp_file="${ACPP_PREFIX}/.fuzzybits-pin"
expected_stamp="ref=${ref}
metal=${metal}
metal_cpp=${metal_cpp}
llvm=${llvm_version}
sdk=$(xcrun --show-sdk-version)
profile=full"

acpp_ok=0
if [[ -x "${ACPP_PREFIX}/bin/acpp-info" ]]; then
  if DYLD_LIBRARY_PATH="${ACPP_PREFIX}/lib:${OMP_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}" \
       "${ACPP_PREFIX}/bin/acpp-info" >/dev/null 2>&1; then
    acpp_ok=1
  fi
fi
if [[ "${acpp_ok}" -eq 1 ]]; then
  if [[ -f "${stamp_file}" ]]; then
    [[ "$(cat "${stamp_file}")" == "${expected_stamp}" ]] || acpp_ok=0
  elif [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
    # Unstamped prefixes predate pin tracking. The Actions cache key already
    # encodes the pins; anywhere else the prefix may be stale.
    acpp_ok=0
  fi
fi
if [[ "${1:-}" == "--check" ]]; then
  [[ "${acpp_ok}" -eq 1 ]]
  exit
fi

start=$(date +%s)
if [[ "${acpp_ok}" -eq 1 ]]; then
  echo "AdaptiveCpp cache hit at ${ACPP_PREFIX}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "cache=hit" >> "${GITHUB_OUTPUT}"
  fi
  echo "acpp_build_seconds=0"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "acpp_build_seconds=0" >> "${GITHUB_ENV}"
  fi
  exit 0
fi

echo "AdaptiveCpp cache miss; building ref=${ref} WITH_METAL_BACKEND=${metal}"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "cache=miss" >> "${GITHUB_OUTPUT}"
fi
rm -rf "${ACPP_PREFIX}" "${RUNNER_TEMP}/acpp"
mkdir -p "${ACPP_PREFIX}" "${RUNNER_TEMP}/acpp"
git clone --filter=blob:none "${ACPP_REPO}" "${RUNNER_TEMP}/acpp/src"
git -C "${RUNNER_TEMP}/acpp/src" fetch --depth 1 origin "${ref}"
git -C "${RUNNER_TEMP}/acpp/src" checkout --detach FETCH_HEAD
got=$(git -C "${RUNNER_TEMP}/acpp/src" rev-parse HEAD)
if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
  test "${got}" = "${ref}"
fi

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=/usr/bin/clang
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++
  -DCMAKE_C_COMPILER_LAUNCHER=ccache
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCLANG_EXECUTABLE_PATH="${LLVM_PREFIX}/bin/clang++"
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
  -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"
  -DCMAKE_INSTALL_PREFIX="${ACPP_PREFIX}"
  -DWITH_OPENCL_BACKEND=OFF
  -DWITH_CUDA_BACKEND=OFF
  -DWITH_ROCM_BACKEND=OFF
  -DWITH_LEVEL_ZERO_BACKEND=OFF
  -DWITH_METAL_BACKEND="${metal}"
  -DACPP_COMPILER_FEATURE_PROFILE=full
  -DACPP_LLD_PATH="${ACPP_LLD_PATH}"
  -DSEQUENTIAL_LINK_LINE="${OMP_PREFIX}/lib/libomp.dylib"
  "-DOMP_LINK_LINE=-fopenmp ${OMP_PREFIX}/lib/libomp.dylib -L${OMP_PREFIX}/lib/"
  "-DOMP_CXX_FLAGS=-fopenmp -D_ENABLE_EXTENDED_ALIGNED_STORAGE -I${OMP_PREFIX}/include"
)
if [[ "${metal}" == "ON" ]]; then
  test -n "${METAL_CPP_DIR:-}"
  test -f "${METAL_CPP_DIR}/Metal/Metal.hpp"
  cmake_args+=(-DMETAL_INCLUDE_DIR="${METAL_CPP_DIR}")
fi

cmake -S "${RUNNER_TEMP}/acpp/src" -B "${RUNNER_TEMP}/acpp/build" -G Ninja "${cmake_args[@]}"
cmake --build "${RUNNER_TEMP}/acpp/build"
cmake --install "${RUNNER_TEMP}/acpp/build"
rm -rf "${RUNNER_TEMP}/acpp"
echo "${expected_stamp}" > "${stamp_file}"
elapsed=$(( $(date +%s) - start ))
echo "acpp_build_seconds=${elapsed}"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "acpp_build_seconds=${elapsed}" >> "${GITHUB_ENV}"
fi
df -h /
