#!/usr/bin/env bash
# Fail if shipped product ELFs contain instructions above x86-64-v2.
# Floor is Ivy Bridge / Xeon E5 v2 (AVX, SSE4.2, POPCNT; no AVX2/FMA/BMI).
#
# Usage: isa-baseline-check.sh [rootfs] [wheelhouse]
#   rootfs     staged runtime root (default /opt/runtime-root)
#   wheelhouse directory of .whl files (optional)
#
# Skips libLLVM and llvm-to-* plugins: LLVM is built with function
# multi-versioning, so objdump hits on AVX2 there are not SIGILL on Ivy Bridge.
set -euo pipefail

ROOT="${1:-/opt/runtime-root}"
WHEELHOUSE="${2:-}"

# BMI2/BMI1/AVX2/FMA that SIGILL on Ivy Bridge. tzcnt is omitted: it is
# encoded as REP BSF and executes as BSF on pre-BMI1. pextrw/andnpd/pandn
# are SSE and must not match (andn is BMI1; andnpd is a different mnemonic).
PAT='[[:space:]](vpbroadcast[a-z0-9]*|vperm[dq][a-z0-9]*|vfmadd[a-z0-9]*|vfmsub[a-z0-9]*|vfnmadd[a-z0-9]*|vpgather[a-z0-9]*|shlx|sarx|shrx|andn|bextr|pdep|pext|lzcnt|bzhi|mulx|rorx|blsi|blsmsk|blsr)[[:space:]]'

need_objdump() {
  if ! command -v objdump >/dev/null 2>&1; then
    echo "error: objdump not found (apt binutils)" >&2
    exit 2
  fi
}

is_elf() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  [[ -L "$f" ]] && return 1
  local magic
  magic="$(od -An -N4 -tx1 "$f" 2>/dev/null | tr -d ' \n')"
  [[ "$magic" == "7f454c46" ]]
}

scan_file() {
  local f="$1"
  local hits
  hits="$(objdump -d "$f" 2>/dev/null | grep -E "$PAT" || true)"
  if [[ -n "$hits" ]]; then
    echo "ISA-BASELINE FAIL: $f" >&2
    echo "$hits" | head -20 >&2
    return 1
  fi
  echo "ISA-BASELINE ok  $f"
}

should_skip() {
  local base
  base="$(basename "$1")"
  case "$base" in
    libLLVM.so*|opt|llc) return 0 ;;
  esac
  case "$1" in
    */llvm-to-backend/*|*/libLLVM.so*) return 0 ;;
  esac
  return 1
}

need_objdump
fail=0
echo "ISA-BASELINE scanning rootfs ${ROOT} (x86-64-v2; no AVX2/FMA/BMI)"

if [[ ! -d "$ROOT" ]]; then
  echo "error: rootfs not a directory: $ROOT" >&2
  exit 2
fi

while IFS= read -r -d '' f; do
  should_skip "$f" && continue
  is_elf "$f" || continue
  if ! scan_file "$f"; then
    fail=1
  fi
done < <(find "${ROOT}/usr/local" -type f \( -name '*.so*' -o -name 'fzb' -o -name 'scram-cli' -o -name 'acpp-info' \) -print0 2>/dev/null)

if [[ -n "$WHEELHOUSE" && -d "$WHEELHOUSE" ]]; then
  echo "ISA-BASELINE scanning wheels in ${WHEELHOUSE}"
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' EXIT
  shopt -s nullglob
  for whl in "${WHEELHOUSE}"/*.whl; do
    echo "ISA-BASELINE wheel $(basename "$whl")"
    python3 -c 'import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
      "$whl" "${tmp}/whl"
    while IFS= read -r -d '' f; do
      case "$f" in
        */fuzzybitslib/*) continue ;;
      esac
      should_skip "$f" && continue
      is_elf "$f" || continue
      if ! scan_file "$f"; then
        fail=1
      fi
    done < <(find "${tmp}/whl" -type f -name '*.so*' -print0)
    rm -rf "${tmp}/whl"
  done
  shopt -u nullglob
fi

if [[ "$fail" -ne 0 ]]; then
  echo "error: shipped ELFs contain instructions above x86-64-v2 (Ivy Bridge)" >&2
  exit 1
fi
echo "ISA-BASELINE OK"
