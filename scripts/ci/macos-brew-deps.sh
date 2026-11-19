#!/usr/bin/env bash
# Homebrew llvm@20 + libomp + lld for AdaptiveCpp on GitHub-hosted macOS.
# Intel Homebrew is /usr/local; Apple Silicon is /opt/homebrew.
set -euo pipefail

export HOMEBREW_NO_AUTO_UPDATE="${HOMEBREW_NO_AUTO_UPDATE:-1}"
export HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK="${HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK:-1}"

brew install llvm@20 libomp ninja ccache coreutils cmake
brew install lld@20 || brew install lld || true

LLVM_PREFIX="$(brew --prefix llvm@20)"
OMP_PREFIX="$(brew --prefix libomp)"
LLD_BIN=""
for c in \
    "$(brew --prefix lld@20 2>/dev/null)/bin/ld64.lld" \
    "$(brew --prefix lld 2>/dev/null)/bin/ld64.lld" \
    "${LLVM_PREFIX}/bin/ld64.lld"; do
  if [[ -x "${c}" ]]; then
    LLD_BIN="${c}"
    break
  fi
done
test -n "${LLD_BIN}"

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "${LLVM_PREFIX}/bin" >> "${GITHUB_PATH}"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "LLVM_PREFIX=${LLVM_PREFIX}"
    echo "OMP_PREFIX=${OMP_PREFIX}"
    echo "ACPP_LLD_PATH=${LLD_BIN}"
  } >> "${GITHUB_ENV}"
fi
export LLVM_PREFIX OMP_PREFIX ACPP_LLD_PATH="${LLD_BIN}"
echo "LLVM_PREFIX=${LLVM_PREFIX}"
echo "OMP_PREFIX=${OMP_PREFIX}"
echo "ACPP_LLD_PATH=${LLD_BIN}"
ccache -z || true
df -h /
