#!/usr/bin/env bash
# Fetch apple/metal-cpp headers (METAL_CPP_VERSION) into METAL_CPP_DIR.
set -euo pipefail

test -n "${METAL_CPP_DIR:-}"
test -n "${METAL_CPP_VERSION:-}"
test "${METAL_CPP_VERSION}" != "TBD"

# A shallow clone has no tag refs, so the branch name is recorded next to it.
stamp="${METAL_CPP_DIR}/.fuzzybits-pin"
if [[ -f "${METAL_CPP_DIR}/Metal/Metal.hpp" && -f "${stamp}" ]] \
    && [[ "$(cat "${stamp}")" == "${METAL_CPP_VERSION}" ]]; then
  echo "metal-cpp ${METAL_CPP_VERSION} present at ${METAL_CPP_DIR}"
else
  rm -rf "${METAL_CPP_DIR}"
  git clone --depth 1 --branch "${METAL_CPP_VERSION}" \
    https://github.com/apple/metal-cpp.git "${METAL_CPP_DIR}"
  echo "${METAL_CPP_VERSION}" > "${stamp}"
fi
test -f "${METAL_CPP_DIR}/Metal/Metal.hpp"
git -C "${METAL_CPP_DIR}" rev-parse HEAD
