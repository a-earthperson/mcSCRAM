#!/usr/bin/env bash
# Commands run *inside* the toolchain container (workdir /src).
# Usage: inner-build.sh <preset> <configure|product|tests|ctest-omp|ctest-mc|cuobjdump-sm70|ccache-stats|smoke|wheel>
set -euo pipefail
cd /src
git config --global --add safe.directory /src || true

preset="${1:?preset}"
cmd="${2:?command}"

ensure_ccache() {
  export DEBIAN_FRONTEND=noninteractive
  if ! command -v ccache >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ccache >/dev/null
  fi
  mkdir -p "${CCACHE_DIR:-/src/.ccache}" "${ACPP_APPDB_DIR:-/src/.cache/acpp}" "${CUDA_CACHE_PATH:-/src/.cache/cuda}"
  export CCACHE_DIR="${CCACHE_DIR:-/src/.ccache}"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-2G}"
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
  # acpp is AdaptiveCpp's clang driver; syclcc-launcher keeps
  # CMAKE_CXX_COMPILER_LAUNCHER in front of acpp, so ccache's compiler is acpp.
  # Force clang flag parsing so SYCL TUs are hashable.
  export CCACHE_COMPILERTYPE="${CCACHE_COMPILERTYPE:-clang}"
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-/src}"
  export CCACHE_NOHASHDIR=1
}

cmake_launcher_flags=(
  -DCMAKE_C_COMPILER=clang-20
  -DCMAKE_CXX_COMPILER=clang++-20
  -DCMAKE_C_COMPILER_LAUNCHER=ccache
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
)

case "${cmd}" in
  configure)
    ensure_ccache
    ccache -z || true
    testing="${BUILD_TESTING:-ON}"
    echo "cmake --preset ${preset} BUILD_TESTING=${testing}"
    cmake --preset "${preset}" -DBUILD_TESTING="${testing}" "${cmake_launcher_flags[@]}"
    echo "CMAKE_CXX_COMPILER_LAUNCHER is set; AdaptiveCpp syclcc-launcher preserves it and invokes: ccache acpp <flags>"
    ;;
  product)
    ensure_ccache
    echo "build product targets (${preset})"
    cmake --build --preset "${preset}" --target scram-cli fzb mc mcscram_native
    ;;
  configure-product)
    ensure_ccache
    ccache -z || true
    testing="${BUILD_TESTING:-ON}"
    echo "cmake --preset ${preset} BUILD_TESTING=${testing}"
    cmake --preset "${preset}" -DBUILD_TESTING="${testing}" "${cmake_launcher_flags[@]}"
    echo "CMAKE_CXX_COMPILER_LAUNCHER is set; AdaptiveCpp syclcc-launcher preserves it and invokes: ccache acpp <flags>"
    echo "build product targets (${preset})"
    cmake --build --preset "${preset}" --target scram-cli fzb mc mcscram_native
    ;;
  tests)
    ensure_ccache
    echo "build remaining targets including tests (${preset})"
    cmake --build --preset "${preset}"
    ;;
  ctest-omp)
    echo "ctest --preset ${preset} (OpenMP, quarantine excluded)"
    export ACPP_VISIBILITY_MASK="${ACPP_VISIBILITY_MASK:-omp}"
    export ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}"
    export ACPP_PERSISTENT_RUNTIME="${ACPP_PERSISTENT_RUNTIME:-1}"
    ctest --preset "${preset}" --output-on-failure --no-tests=error
    ;;
  ctest-mc)
    echo "ctest -L mc ACPP_VISIBILITY_MASK=cuda (${preset})"
    export ACPP_VISIBILITY_MASK=cuda
    export ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}"
    export ACPP_PERSISTENT_RUNTIME="${ACPP_PERSISTENT_RUNTIME:-1}"
    ctest --test-dir "build/${preset}" -L mc -LE quarantine --output-on-failure --no-tests=error
    ;;
  cuobjdump-sm70)
    lib="$(find "build/${preset}" -name libmc.so -print -quit)"
    if [[ -z "${lib}" ]]; then
      echo "error: libmc.so not found under build/${preset}" >&2
      exit 1
    fi
    echo "cuobjdump ${lib}"
    dump="$(mktemp)"
    cuobjdump_bin="$(command -v cuobjdump || true)"
    cuobjdump_bin="${cuobjdump_bin:-/usr/local/cuda/bin/cuobjdump}"
    "${cuobjdump_bin}" --list-elf --list-ptx "${lib}" | tee "${dump}"
    if ! grep -Eq 'sm_70' "${dump}"; then
      echo "error: no sm_70 SASS/arch in libmc" >&2
      exit 1
    fi
    if ! grep -Eqi 'ptx|compute_70' "${dump}"; then
      echo "error: no embedded PTX/compute_70 in libmc" >&2
      exit 1
    fi
    echo "ok: libmc contains sm_70 SASS and PTX"
    ;;
  ccache-stats)
    ensure_ccache
    echo "=== ccache stats ==="
    ccache --verbose --show-stats 2>/dev/null || ccache -s
    ;;
  smoke)
    cli="$(find "build/${preset}" -name scram-cli -type f -print -quit)"
    if [[ -z "${cli}" ]]; then
      echo "error: scram-cli not found under build/${preset}" >&2
      exit 1
    fi
    export SCRAM_CLI="${cli}"
    export SMOKE_INPUT_DIR=/src/docker/inputs
    export ACPP_VISIBILITY_MASK=cuda
    echo "smoke scram-cli=${cli}"
    # docker/smoke.sh wraps every model in `timeout 60` inside this container.
    /src/docker/smoke.sh chinese baobab1 das9201
    ;;
  wheel)
    ensure_ccache
    ccache -z || true
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends python3-venv python3-pip python3-dev >/dev/null
    python3 -m venv /tmp/wheel-venv
    /tmp/wheel-venv/bin/pip install -U pip >/dev/null
    /tmp/wheel-venv/bin/pip install 'scikit-build-core>=0.10' 'pybind11>=2.13' ninja cmake build packaging
    mkdir -p /src/dist
    export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
    PATH="/tmp/wheel-venv/bin:${PATH}" FUZZYBITS_VARIANT=generic /src/docker/pip-wheel.sh /src/dist
    ls -lah /src/dist
    ;;
  *)
    echo "unknown command ${cmd}" >&2
    exit 2
    ;;
esac
