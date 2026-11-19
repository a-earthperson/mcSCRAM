#!/usr/bin/env bash
# Build the CI toolchain image on a Docker context and measure
# cmake --preset generic|cuda-sm70 on hosted-runner-sized resources.
# Does not push to GHCR.
set -euo pipefail

CONTEXT="${DOCKER_CONTEXT:-xerxes}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

set -a
# shellcheck disable=SC1091
. ./toolchain.env
set +a

export TOOLCHAIN_SHA="$(sha256sum toolchain.env | cut -c1-12)"
export FUZZYBITS_REVISION="$(git rev-parse HEAD)"
LOCAL="fuzzybits-toolchain:${ACPP_REF}"
GENERIC_TAG="${LOCAL}-generic"
CUDA_TAG="${LOCAL}-cuda12.8"
GENERIC_IMMUTABLE="${LOCAL}-generic-${TOOLCHAIN_SHA}"
CUDA_IMMUTABLE="${LOCAL}-cuda12.8-${TOOLCHAIN_SHA}"
VERIFY_GENERIC="fuzzybits-toolchain-verify:generic"
VERIFY_CUDA="fuzzybits-toolchain-verify:cuda-sm70"

log() { printf '[verify-toolchain] %s\n' "$*"; }

log "building ${GENERIC_TAG} (docker --context ${CONTEXT})"
START=$(date +%s)
docker --context "${CONTEXT}" build \
  -f docker/toolchain.Dockerfile \
  --target toolchain \
  --build-arg ACPP_REF \
  --build-arg ACPP_REPO \
  --build-arg ACPP_COMMIT \
  --build-arg CLANG_VERSION \
  --build-arg CUDA_DEVEL_IMAGE \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE \
  --build-arg BOOST_URL \
  --build-arg BOOST_SHA256 \
  --build-arg BOOST_VERSION \
  --build-arg FUZZYBITS_REVISION \
  --build-arg TOOLCHAIN_SHA \
  -t "${GENERIC_TAG}" \
  -t "${CUDA_TAG}" \
  -t "${GENERIC_IMMUTABLE}" \
  -t "${CUDA_IMMUTABLE}" \
  .
END=$(date +%s)
BUILD_SECS=$((END - START))
log "toolchain image build wall: ${BUILD_SECS}s"

log "inspecting labels / backends / forbidden stacks"
docker --context "${CONTEXT}" inspect "${GENERIC_TAG}" \
  --format '{{json .Config.Labels}}' | python3 -m json.tool
docker --context "${CONTEXT}" run --rm --entrypoint bash "${GENERIC_TAG}" -lc '
set -euo pipefail
echo "acpp-commit=$(cat /usr/local/share/acpp-commit)"
echo "acpp-describe=$(cat /usr/local/share/acpp-describe)"
echo "cmake=$(cmake --version | head -1)"
echo "clang=$(clang-20 --version | head -1)"
test -f /usr/local/lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake
test -d /usr/local/lib/cmake/Boost-1.89.0 -o -d /usr/local/lib/cmake/Boost-1.89
ls /usr/local/lib/hipSYCL/librt-backend-cuda.so /usr/local/lib/hipSYCL/librt-backend-omp.so
test ! -e /usr/local/lib/hipSYCL/librt-backend-rocm.so
test ! -e /usr/local/lib/hipSYCL/librt-backend-ze.so
test ! -e /usr/local/lib/hipSYCL/librt-backend-ocl.so
test ! -d /opt/intel
test ! -d /opt/rocm
if dpkg-query -W 2>/dev/null | grep -Ei "oneapi|level-zero|rocm|intel-opencl"; then
  echo "error: forbidden packages present" >&2
  exit 1
fi
echo "=== acpp-info -l ==="
acpp-info -l
echo "=== plugin-with-sscp-compiler ==="
python3 - <<PY
import json
p="/usr/local/etc/AdaptiveCpp/acpp-core.json"
d=json.load(open(p))
print("plugin-with-sscp-compiler=", d.get("plugin-with-sscp-compiler"))
print("plugin-llvm-version-major=", d.get("plugin-llvm-version-major"))
assert d.get("plugin-with-sscp-compiler") == "true", d
assert d.get("plugin-llvm-version-major") == "20", d
PY
'

log "thin verify image (COPY source); generic and cuda-sm70 share the same toolchain"
docker --context "${CONTEXT}" build \
  -f docker/toolchain-verify.Dockerfile \
  --build-arg TOOLCHAIN="${GENERIC_TAG}" \
  -t "${VERIFY_GENERIC}" \
  -t "${VERIFY_CUDA}" \
  .

log "cmake --preset generic on --cpus=4 -m 16g"
GENERIC_START=$(date +%s)
docker --context "${CONTEXT}" run --rm --cpus=4 -m 16g \
  --entrypoint bash "${VERIFY_GENERIC}" \
  -lc 'cmake --preset generic && cmake --build --preset generic'
GENERIC_SECS=$(( $(date +%s) - GENERIC_START ))
log "preset generic wall: ${GENERIC_SECS}s"

log "cmake --preset cuda-sm70 on --cpus=4 -m 16g"
CUDA_START=$(date +%s)
docker --context "${CONTEXT}" run --rm --cpus=4 -m 16g \
  --entrypoint bash "${VERIFY_CUDA}" \
  -lc 'cmake --preset cuda-sm70 && cmake --build --preset cuda-sm70'
CUDA_SECS=$(( $(date +%s) - CUDA_START ))
log "preset cuda-sm70 wall: ${CUDA_SECS}s"

log "summary"
cat <<EOF
toolchain_image_build_s=${BUILD_SECS}
generic_cmake_build_s=${GENERIC_SECS}
cuda_sm70_cmake_build_s=${CUDA_SECS}
tags=${GENERIC_TAG} ${CUDA_TAG} ${GENERIC_IMMUTABLE} ${CUDA_IMMUTABLE}
toolchain_sha=${TOOLCHAIN_SHA}
acpp_ref=${ACPP_REF}
acpp_commit=${ACPP_COMMIT}
EOF
