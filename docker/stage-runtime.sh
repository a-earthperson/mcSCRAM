#!/usr/bin/env bash
# Stage a slim runtime rootfs from a toolchain+product builder.
# Env:
#   DEST              staging prefix (default /opt/runtime-root)
#   INCLUDE_SSCP_JIT  1 = copy LLVM/SSCP pieces (generic); 0 = SMCP-only (cuda-sm70)
#   CLANG_VERSION     LLVM major (default 20)
#   CMAKE_PRESET      build dir name under /src/build (generic|cuda-sm70)
set -euo pipefail

DEST="${DEST:-/opt/runtime-root}"
INCLUDE_SSCP_JIT="${INCLUDE_SSCP_JIT:-1}"
CLANG_VERSION="${CLANG_VERSION:-20}"
CMAKE_PRESET="${CMAKE_PRESET:-generic}"
SRC="${SRC:-/src}"
BUILDDIR="${SRC}/build/${CMAKE_PRESET}"

mkdir -p "${DEST}"

# Second argument is a rootfs-absolute path (e.g. /usr/local/bin/fzb),
# not a host path under DEST.
stage() {
  local src="$1"
  local rel="${2:-$1}"
  local dest="${DEST}${rel}"
  if [[ ! -e "${src}" ]]; then
    echo "skip missing ${src}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  # libLLVM.so.20.1 is a symlink into the LLVM tree. cp -a would copy a
  # dangling link; dereference files so the runtime image has the bytes.
  if [[ -L "${src}" && ! -d "${src}" ]]; then
    cp -aL "${src}" "${dest}"
  else
    cp -a "${src}" "${dest}"
  fi
  echo "copy ${src} -> ${dest}"
}

stage_glob() {
  local pattern="$1"
  local dest_dir="${2:-}"
  local hits=()
  shopt -s nullglob
  # Unquoted so the glob expands. Quoted "${pattern}" staged a literal
  # `libacpp-rt.so*` path and silently skipped every shared library.
  hits=(${pattern})
  shopt -u nullglob
  if [[ ${#hits[@]} -eq 0 ]]; then
    echo "skip missing ${pattern}"
    return 0
  fi
  local src
  for src in "${hits[@]}"; do
    if [[ -n "${dest_dir}" ]]; then
      stage "${src}" "${dest_dir}/$(basename "${src}")"
    else
      stage "${src}"
    fi
  done
}

require_staged() {
  local rel
  for rel in "$@"; do
    if [[ -e "${DEST}${rel}" ]]; then
      return 0
    fi
  done
  echo "error: required artefact missing in ${DEST}: $*" >&2
  find "${DEST}/usr/local/lib" -maxdepth 2 -type f -printf '%p\n' | sort >&2 || true
  exit 1
}

echo "stage-runtime DEST=${DEST} INCLUDE_SSCP_JIT=${INCLUDE_SSCP_JIT} CMAKE_PRESET=${CMAKE_PRESET}"

# --- product binaries -------------------------------------------------------
stage /usr/local/bin/scram-cli
stage /usr/local/bin/fzb
if [[ ! -e "${DEST}/usr/local/bin/scram-cli" ]]; then
  stage "${BUILDDIR}/targets/scram/scram-cli" /usr/local/bin/scram-cli
fi
if [[ ! -e "${DEST}/usr/local/bin/fzb" ]]; then
  stage "${BUILDDIR}/targets/fzb/fzb" /usr/local/bin/fzb
fi

# --- product shared libs ----------------------------------------------------
for libdir in /usr/local/lib /usr/local/lib/x86_64-linux-gnu; do
  stage_glob "${libdir}/libscram.so*"
  stage_glob "${libdir}/libmc.so*"
  stage_glob "${libdir}/libxml2.so*"
done
stage_glob "${BUILDDIR}/src/libscram.so*" /usr/local/lib
stage_glob "${BUILDDIR}/src/mc/libmc.so*" /usr/local/lib
stage_glob "${BUILDDIR}/_deps/libxml2-build/libxml2.so*" /usr/local/lib

# --- AdaptiveCpp runtime ----------------------------------------------------
stage /usr/local/bin/acpp-info
stage_glob "/usr/local/lib/libacpp-rt.so*"
stage_glob "/usr/local/lib/libacpp-common.so*"
stage /usr/local/etc/AdaptiveCpp
stage /usr/local/share/acpp-commit
stage /usr/local/share/acpp-describe

mkdir -p "${DEST}/usr/local/lib/hipSYCL"
stage /usr/local/lib/hipSYCL/librt-backend-cuda.so
stage /usr/local/lib/hipSYCL/librt-backend-omp.so

# --- CUDA runtime (cudart only; driver comes from nvidia-container-toolkit) -
mkdir -p "${DEST}/usr/local/cuda/lib64" "${DEST}/usr/local/lib"
stage_glob "/usr/local/cuda/lib64/libcudart.so*" /usr/local/cuda/lib64
stage_glob "/usr/local/cuda/lib64/libcudart.so*" /usr/local/lib

# --- OpenMP (AdaptiveCpp always loads the OMP backend) ----------------------
for omp in \
  "/usr/lib/llvm-${CLANG_VERSION}/lib/libomp.so.5" \
  "/usr/lib/llvm-${CLANG_VERSION}/lib/libomp.so" \
  /usr/lib/x86_64-linux-gnu/libomp.so.5; do
  if [[ -e "${omp}" ]]; then
    stage "${omp}" /usr/lib/x86_64-linux-gnu/libomp.so.5
    break
  fi
done

# --- SSCP JIT (generic only) ------------------------------------------------
if [[ "${INCLUDE_SSCP_JIT}" == "1" ]]; then
  echo "including SSCP JIT pieces"
  if [[ -d /usr/local/lib/hipSYCL/llvm-to-backend ]]; then
    cp -a /usr/local/lib/hipSYCL/llvm-to-backend "${DEST}/usr/local/lib/hipSYCL/llvm-to-backend"
  fi
  if [[ -d /usr/local/lib/hipSYCL/bitcode ]]; then
    cp -a /usr/local/lib/hipSYCL/bitcode "${DEST}/usr/local/lib/hipSYCL/bitcode"
  fi
  if [[ -d /usr/local/lib/hipSYCL/ext ]]; then
    cp -a /usr/local/lib/hipSYCL/ext "${DEST}/usr/local/lib/hipSYCL/ext"
  fi
  for llvm in \
    "/usr/lib/llvm-${CLANG_VERSION}/lib/libLLVM.so.${CLANG_VERSION}.1" \
    "/usr/lib/x86_64-linux-gnu/libLLVM.so.${CLANG_VERSION}.1"; do
    if [[ -e "${llvm}" ]]; then
      stage "${llvm}" "/usr/local/lib/libLLVM.so.${CLANG_VERSION}.1"
      stage "${llvm}" "/usr/lib/x86_64-linux-gnu/libLLVM.so.${CLANG_VERSION}.1"
      break
    fi
  done
  stage "/usr/lib/llvm-${CLANG_VERSION}/bin/opt"
  stage "/usr/lib/llvm-${CLANG_VERSION}/bin/llc"
  stage /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
else
  echo "omitting SSCP JIT pieces (SMCP AOT runtime)"
fi

# --- smoke inputs + scripts -------------------------------------------------
mkdir -p "${DEST}/app/input" "${DEST}/usr/local/bin"
stage "${SRC}/docker/inputs/chinese.xml" /app/input/chinese.xml
stage "${SRC}/docker/inputs/baobab1.xml" /app/input/baobab1.xml
stage "${SRC}/docker/inputs/das9201.xml" /app/input/das9201.xml
stage "${SRC}/docker/smoke.sh" /usr/local/bin/mcscram-smoke
stage "${SRC}/docker/image-smoke.sh" /usr/local/bin/fuzzybits-image-smoke
chmod 0755 "${DEST}/usr/local/bin/mcscram-smoke" "${DEST}/usr/local/bin/fuzzybits-image-smoke"
chmod 0755 "${DEST}/usr/local/bin/scram-cli" "${DEST}/usr/local/bin/fzb" || true
if [[ -e "${DEST}/usr/local/bin/acpp-info" ]]; then
  chmod 0755 "${DEST}/usr/local/bin/acpp-info"
fi

printf '%s\n' '#!/bin/sh' \
  'if [ "$1" = "fzb" ]; then shift; exec /usr/local/bin/fzb "$@"; fi' \
  'exec /usr/local/bin/scram-cli "$@"' \
  > "${DEST}/usr/local/bin/scram-or-fzb-entrypoint"
chmod 0755 "${DEST}/usr/local/bin/scram-or-fzb-entrypoint"

# RPATH: drop build-tree paths so the slim image resolves /usr/local/lib.
if command -v patchelf >/dev/null 2>&1; then
  for bin in \
    "${DEST}/usr/local/bin/scram-cli" \
    "${DEST}/usr/local/bin/fzb" \
    "${DEST}/usr/local/lib/libscram.so" \
    "${DEST}/usr/local/lib/libmc.so"; do
    if [[ -f "${bin}" ]]; then
      patchelf --set-rpath "/usr/local/lib:/usr/local/cuda/lib64" "${bin}" || true
      echo "rpath ${bin}"
    fi
  done
  shopt -s nullglob
  for so in "${DEST}/usr/local/lib"/libscram.so* "${DEST}/usr/local/lib"/libmc.so* "${DEST}/usr/local/lib"/libxml2.so*; do
    [[ -f "${so}" ]] || continue
    patchelf --set-rpath "/usr/local/lib:/usr/local/cuda/lib64" "${so}" || true
  done
  shopt -u nullglob
fi

echo "staged rootfs:"
du -sh "${DEST}" || true
find "${DEST}/usr/local/bin" -maxdepth 1 -type f -printf '%p\n' | sort
test -x "${DEST}/usr/local/bin/scram-cli"
test -x "${DEST}/usr/local/bin/fzb"
test -x "${DEST}/usr/local/bin/acpp-info"
test -x "${DEST}/usr/local/bin/mcscram-smoke"
test -f "${DEST}/app/input/chinese.xml"
require_staged /usr/local/lib/libscram.so /usr/local/lib/libscram.so.0
require_staged /usr/local/lib/libmc.so /usr/local/lib/libmc.so.0
require_staged /usr/local/lib/libacpp-rt.so /usr/local/lib/libacpp-rt.so.0
require_staged /usr/local/lib/libacpp-common.so /usr/local/lib/libacpp-common.so.0
require_staged /usr/local/lib/libxml2.so.16 /usr/local/lib/libxml2.so
require_staged /usr/local/cuda/lib64/libcudart.so.12 /usr/local/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so
if [[ "${INCLUDE_SSCP_JIT}" == "1" ]]; then
  require_staged "/usr/local/lib/libLLVM.so.${CLANG_VERSION}.1" \
    "/usr/lib/x86_64-linux-gnu/libLLVM.so.${CLANG_VERSION}.1"
  require_staged "/usr/lib/llvm-${CLANG_VERSION}/bin/opt"
  require_staged /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
fi
echo "required libs present"
