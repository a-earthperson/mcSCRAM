#!/usr/bin/env bash
# Build Boost into BOOST_PREFIX unless it already matches the pins.
# Usage: macos-boost-build.sh [--check]
#   --check  exit 0 if BOOST_PREFIX matches the pins, 1 otherwise; build nothing.
set -euo pipefail

test -n "${BOOST_PREFIX:-}"
test -n "${BOOST_URL:-}"
test -n "${BOOST_SHA256:-}"
: "${BOOST_INCLUDE_LIBRARIES:=program_options;filesystem;system;random;range;exception;multi_index;accumulators;multiprecision;icl;math;dll;regex;test}"
: "${RUNNER_TEMP:=/tmp}"

stamp_file="${BOOST_PREFIX}/.fuzzybits-pin"
expected_stamp="sha256=${BOOST_SHA256}
libraries=${BOOST_INCLUDE_LIBRARIES}"

up_to_date() {
  [[ -f "${BOOST_PREFIX}/include/boost/version.hpp" && -f "${stamp_file}" ]] \
    && [[ "$(cat "${stamp_file}")" == "${expected_stamp}" ]]
}

if [[ "${1:-}" == "--check" ]]; then
  up_to_date
  exit
fi
if up_to_date; then
  echo "Boost at ${BOOST_PREFIX} matches pins; skipping build"
  exit 0
fi

start=$(date +%s)
rm -rf "${BOOST_PREFIX}" "${RUNNER_TEMP}/boost"
mkdir -p "${BOOST_PREFIX}" "${RUNNER_TEMP}/boost"
curl -fsSL -o "${RUNNER_TEMP}/boost/boost.tar.gz" "${BOOST_URL}"
echo "${BOOST_SHA256}  ${RUNNER_TEMP}/boost/boost.tar.gz" | shasum -a 256 -c
mkdir -p "${RUNNER_TEMP}/boost/src"
tar -xzf "${RUNNER_TEMP}/boost/boost.tar.gz" -C "${RUNNER_TEMP}/boost/src" --strip-components=1
cmake -S "${RUNNER_TEMP}/boost/src" -B "${RUNNER_TEMP}/boost/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${BOOST_PREFIX}" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBOOST_INCLUDE_LIBRARIES="${BOOST_INCLUDE_LIBRARIES}"
cmake --build "${RUNNER_TEMP}/boost/build"
cmake --install "${RUNNER_TEMP}/boost/build"
mkdir -p "${BOOST_PREFIX}/include/boost"
cp -a "${RUNNER_TEMP}/boost/src/libs/dll/include/boost/dll.hpp" \
  "${BOOST_PREFIX}/include/boost/dll.hpp"
cp -a "${RUNNER_TEMP}/boost/src/libs/dll/include/boost/dll" \
  "${BOOST_PREFIX}/include/boost/dll"
rm -rf "${RUNNER_TEMP}/boost"
echo "${expected_stamp}" > "${stamp_file}"
elapsed=$(( $(date +%s) - start ))
echo "boost_build_seconds=${elapsed}"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "boost_build_seconds=${elapsed}" >> "${GITHUB_ENV}"
fi
df -h /
