#!/usr/bin/env bash
# Build one fuzzybits wheel. FUZZYBITS_VARIANT=generic|cuda-smNN (cuda-sm70, cuda-sm86, cuda-sm120)
set -euo pipefail

variant="${FUZZYBITS_VARIANT:-generic}"
dest="${1:-/out/wheelhouse}"
mkdir -p "${dest}"

cfg=(
  --config-settings=cmake.define.MALLOC_TYPE=malloc
  --config-settings=cmake.define.FUZZYBITS_WITH_MC=ON
  --config-settings=cmake.define.BUILD_TESTING=OFF
  --config-settings=cmake.define.OPTIMIZE_FOR_NATIVE=OFF
  --config-settings=cmake.define.WITH_PROFILE=OFF
  --config-settings=cmake.define.WITH_COVERAGE=OFF
)

case "${variant}" in
  generic)
    cfg+=(--config-settings=cmake.define.ACPP_TARGETS=generic)
    ;;
  cuda-sm[0-9]*)
    arch="${variant#cuda-sm}"
    if [[ ! "${arch}" =~ ^[0-9]+$ ]]; then
      echo "unknown FUZZYBITS_VARIANT=${variant}" >&2
      exit 2
    fi
    cfg+=(
      --config-settings=cmake.define.ACPP_TARGETS=cuda:sm_${arch}
      --config-settings=cmake.define.ACPP_USE_ACCELERATED_CPU=off
      --config-settings=cmake.define.ACPP_STDPAR=off
      --config-settings=cmake.define.ACPP_EXTRA_COMPILE_OPTIONS=--cuda-include-ptx=sm_${arch}
    )
    ;;
  *)
    echo "unknown FUZZYBITS_VARIANT=${variant}" >&2
    exit 2
    ;;
esac

echo "pip wheel variant=${variant}"
exec pip wheel . -w "${dest}" --no-deps "${cfg[@]}"
