#!/usr/bin/env bash
# Commands run *inside* the toolchain container (workdir /src).
# Usage: inner-build.sh <preset> <configure|product|tests|ctest-omp|ctest-mc|cuobjdump-sm70|ccache-stats|smoke|spine|wheel>
set -euo pipefail
cd /src
git config --global --add safe.directory /src || true

# docker runs as root; chown bind-mounted outputs so the runner can cache
# and delete them.
fix_owner() {
  local owner
  owner="$(stat -c '%u:%g' /src 2>/dev/null || true)"
  if [[ -n "${owner}" && "${owner}" != "0:0" ]]; then
    chown -R "${owner}" /src/build /src/.ccache /src/.cache /src/dist 2>/dev/null || true
  fi
}
trap fix_owner EXIT

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
  -DOPTIMIZE_FOR_NATIVE=OFF
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
    native_mod="_native_generic"
    case "${preset}" in
      cuda-sm[0-9]*) native_mod="_native_${preset//-/_}" ;;
    esac
    cmake --build --preset "${preset}" --target scram-cli fzb mc "${native_mod}"
    ;;
  configure-product)
    ensure_ccache
    ccache -z || true
    testing="${BUILD_TESTING:-ON}"
    echo "cmake --preset ${preset} BUILD_TESTING=${testing}"
    cmake --preset "${preset}" -DBUILD_TESTING="${testing}" "${cmake_launcher_flags[@]}"
    echo "CMAKE_CXX_COMPILER_LAUNCHER is set; AdaptiveCpp syclcc-launcher preserves it and invokes: ccache acpp <flags>"
    echo "build product targets (${preset})"
    native_mod="_native_generic"
    case "${preset}" in
      cuda-sm[0-9]*) native_mod="_native_${preset//-/_}" ;;
    esac
    cmake --build --preset "${preset}" --target scram-cli fzb mc "${native_mod}"
    ;;
  tests)
    ensure_ccache
    echo "build remaining targets including tests (${preset})"
    cmake --build --preset "${preset}"
    ;;
  ctest-omp)
    # The compile-fail tests build through CMAKE_CXX_COMPILER_LAUNCHER=ccache.
    ensure_ccache
    echo "ctest --preset ${preset} (OpenMP, quarantine excluded)"
    export ACPP_VISIBILITY_MASK="${ACPP_VISIBILITY_MASK:-omp}"
    export ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}"
    export ACPP_PERSISTENT_RUNTIME="${ACPP_PERSISTENT_RUNTIME:-1}"
    ctest --preset "${preset}" --output-on-failure --no-tests=error
    ;;
  ctest-mc)
    ensure_ccache
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
  spine)
    export SPINE_BACKEND="${SPINE_BACKEND:-omp}"
    export SPINE_TRIALS="${SPINE_TRIALS:-4194304}"
    export SPINE_TIMEOUT="${SPINE_TIMEOUT:-60}"
    export SPINE_Z_MAX="${SPINE_Z_MAX:-4}"
    echo "spine backend=${SPINE_BACKEND} trials=${SPINE_TRIALS}"
    # scripts/ci/spine.sh wraps every fzb run in `timeout 60` inside this container.
    /src/scripts/ci/spine.sh
    ;;
  wheel)
    ensure_ccache
    ccache -z || true
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      python3-venv python3-pip python3-dev patchelf libatomic1 \
      curl ca-certificates >/dev/null
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="${HOME}/.local/bin:${PATH}"
    read -r -a pythons <<< "${PYTHON_VERSIONS:-3.12}"
    if ((${#pythons[@]} == 0)); then
      echo "error: PYTHON_VERSIONS is empty" >&2
      exit 2
    fi
    echo "wheel CPython versions: ${pythons[*]}"
    uv python install "${pythons[@]}"
    mkdir -p /src/dist
    export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
    orig_path="${PATH}"
    for py in "${pythons[@]}"; do
      echo "=== wheel CPython ${py} ==="
      start=$(date +%s)
      venv="/tmp/wheel-venv-${py}"
      rm -rf "${venv}"
      uv venv --seed --python "${py}" "${venv}"
      "${venv}/bin/pip" install -U pip >/dev/null
      "${venv}/bin/pip" install 'scikit-build-core>=0.10' 'pybind11>=3.1' ninja cmake build packaging auditwheel
      wheelhouse="/tmp/wheelhouse-cp${py//./}"
      rm -rf "${wheelhouse}"
      mkdir -p "${wheelhouse}"
      PATH="${venv}/bin:${orig_path}" FUZZYBITS_VARIANT=generic /src/docker/pip-wheel.sh "${wheelhouse}"
      PATH="${venv}/bin:${orig_path}" /src/docker/auditwheel-repair.sh "${wheelhouse}" /src/dist
      echo "wheel_python_${py}_seconds=$(( $(date +%s) - start ))"
    done
    ls -lah /src/dist
    ;;
  trim)
    rm -rf /src/build /src/dist /src/.cache
    ;;
  *)
    echo "unknown command ${cmd}" >&2
    exit 2
    ;;
esac
