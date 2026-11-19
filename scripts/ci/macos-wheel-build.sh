#!/usr/bin/env bash
# Build a macosx_*_{arm64,x86_64} wheel against the AdaptiveCpp prefix.
set -euo pipefail
export COPYFILE_DISABLE=1

dest="${1:-dist}"
wheelhouse="${2:-wheelhouse}"
mkdir -p "${dest}" "${wheelhouse}"

arch="$(uname -m)"
test -n "${ACPP_PREFIX:-}"
test -x "${ACPP_PREFIX}/bin/acpp-info"

export PATH="${ACPP_PREFIX}/bin:${LLVM_PREFIX:-}/bin:${PATH}"
export DYLD_LIBRARY_PATH="${ACPP_PREFIX}/lib:${OMP_PREFIX:-}/lib:${DYLD_LIBRARY_PATH:-}"
export AdaptiveCpp_DIR="${ACPP_PREFIX}/lib/cmake/AdaptiveCpp"
export CMAKE_PREFIX_PATH="${ACPP_PREFIX}${BOOST_PREFIX:+;${BOOST_PREFIX}}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
if [[ "${arch}" == "arm64" ]]; then
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"
else
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
fi
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"

cfg=(
  --config-settings=cmake.define.MALLOC_TYPE=malloc
  --config-settings=cmake.define.FUZZYBITS_WITH_MC=ON
  --config-settings=cmake.define.BUILD_TESTING=OFF
  --config-settings=cmake.define.OPTIMIZE_FOR_NATIVE=OFF
  --config-settings=cmake.define.WITH_PROFILE=OFF
  --config-settings=cmake.define.WITH_COVERAGE=OFF
  --config-settings=cmake.define.ACPP_TARGETS=generic
  --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/clang
  --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++
  --config-settings=cmake.define.CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
  --config-settings=cmake.define.AdaptiveCpp_DIR="${AdaptiveCpp_DIR}"
  --config-settings=cmake.define.CMAKE_OSX_SYSROOT="${SDKROOT}"
  --config-settings=cmake.define.CMAKE_OSX_ARCHITECTURES="${arch}"
  --config-settings=cmake.define.CMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}"
)

if command -v ccache >/dev/null 2>&1; then
  cfg+=(
    --config-settings=cmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache
    --config-settings=cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

echo "pip wheel macos generic arch=${arch} macosx=${MACOSX_DEPLOYMENT_TARGET} python=$(python -c 'import sys; print(sys.version.split()[0])')"
python -m pip wheel . -w "${wheelhouse}" --no-deps "${cfg[@]}"

shopt -s nullglob
raw=("${wheelhouse}"/fuzzybits-*.whl)
if ((${#raw[@]} == 0)); then
  echo "error: no wheel in ${wheelhouse}" >&2
  exit 1
fi

scripts/ci/macos-delocate.sh "${wheelhouse}" "${dest}" "${arch}"
ls -lah "${dest}"
