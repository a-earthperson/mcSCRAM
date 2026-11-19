# syntax=docker/dockerfile:1.10.0
#
# CI toolchain image: LLVM 20 + AdaptiveCpp (full = SSCP + clang CUDA SMCP)
# + CUDA 12.8 devel + CMake/Ninja + Boost 1.89. Serves both
#   cmake --preset generic
#   cmake --preset cuda-sm70
# so one image is published under both -generic and -cuda12.8 tags.
#
# Stages base → minimal → llvm-clang → adaptivecpp-clang are the CUDA-only
# recipe from Dockerfile.cuda12.8 (no oneAPI / Level Zero / ROCm). Keep the
# AdaptiveCpp cmake/build RUN in lockstep with that file. Default profile
# here is `full`; Dockerfile.cuda12.8 stays `minimal` for slim SMCP runtimes.
#
# ARG defaults match toolchain.env. CI must pass --build-arg from that file:
#   set -a && . ./toolchain.env && set +a
#   docker build -f docker/toolchain.Dockerfile --target toolchain \
#     --build-arg ACPP_REF --build-arg ACPP_REPO --build-arg ACPP_COMMIT \
#     --build-arg CLANG_VERSION \
#     --build-arg CUDA_DEVEL_IMAGE \
#     --build-arg ACPP_COMPILER_FEATURE_PROFILE \
#     --build-arg BOOST_URL --build-arg BOOST_SHA256 --build-arg BOOST_VERSION \
#     --build-arg FUZZYBITS_REVISION --build-arg TOOLCHAIN_SHA \
#     -t ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic .
#
# See docs/ci-toolchain.md.

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04

FROM ${CUDA_DEVEL_IMAGE} AS base

ENV CODENAME="noble" \
    DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:apt-fast/stable && \
    apt-get install -y --no-install-recommends apt-fast && \
    apt-fast upgrade -y --no-install-recommends && \
    apt-fast -y autoremove

# CUDA-only: no OpenCL ICD, intel-gpu-tools, oneAPI, Level Zero, or ROCm.
ENV BUILD_PACKAGES="ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    python3 \
    python3-dev \
    sudo \
    wget"

ENV SRC_BUILD_PACKAGES="cmake \
    libboost-all-dev \
    libedit-dev \
    libhwloc-dev \
    libnuma-dev \
    libxml2-dev \
    libzstd-dev \
    ninja-build \
    pkg-config"

ENV SCRAM_BUILD_PACKAGES="g++ \
    libgoogle-perftools-dev \
    libjemalloc-dev \
    libomp-dev \
    make"

ENV SCRAM_RUNTIME_PACKAGES="libxml2 \
    libtcmalloc-minimal4 \
    libjemalloc2 \
    python3"

FROM base AS minimal
WORKDIR /build
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-fast install -y --no-install-recommends \
    $BUILD_PACKAGES \
    $SRC_BUILD_PACKAGES \
    $SCRAM_BUILD_PACKAGES \
    $SCRAM_RUNTIME_PACKAGES && \
    update-ca-certificates

# clang 20: CUDA 12.8 is the last officially supported toolkit (warns on 12.9).
# AdaptiveCpp supports official LLVM releases 15–21; 20 is the CUDA-12.8 match.
FROM minimal AS llvm-clang
WORKDIR /build
ARG CLANG_VERSION=20
ENV CLANG_VERSION=${CLANG_VERSION}
ENV LLVM_PACKAGES="libclang-${CLANG_VERSION}-dev \
    clang-tools-${CLANG_VERSION} \
    libclang-rt-${CLANG_VERSION}-dev \
    libomp-${CLANG_VERSION}-dev \
    llvm-${CLANG_VERSION}-dev \
    lld-${CLANG_VERSION}"
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    wget -q https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${CLANG_VERSION} && \
    apt-fast update && \
    apt-fast install -y --no-install-recommends $LLVM_PACKAGES

# AdaptiveCpp CUDA SMCP (clang CUDA) + SSCP. Do NOT pass -DWITH_CUDA_NVCXX_ONLY=*
# (deprecated; forces ACPP_COMPILER_FEATURE_PROFILE=custom-deprecated).
# `full` builds SSCP (generic JIT) and the clang CUDA plugin (cuda-sm70).
FROM llvm-clang AS adaptivecpp-clang
ARG CMAKE_BUILD_TYPE=Release
ARG ACPP_REPO=https://github.com/AdaptiveCpp/AdaptiveCpp.git
ARG ACPP_REF=v25.10.0
# Default matches Dockerfile.cuda12.8 so BuildKit cache hits the AdaptiveCpp
# layer. toolchain.env / CI pass ACPP_COMPILER_FEATURE_PROFILE=full; the
# toolchain stage asserts that below (minimal cannot configure `generic`).
ARG ACPP_COMPILER_FEATURE_PROFILE=minimal
ENV CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    CMAKE_C_COMPILER=/usr/bin/clang-${CLANG_VERSION} \
    CC=/usr/bin/clang-${CLANG_VERSION} \
    CMAKE_CXX_COMPILER=/usr/bin/clang++-${CLANG_VERSION} \
    CXX=/usr/bin/clang++-${CLANG_VERSION} \
    ADAPTIVE_CPP_INSTALL_DIR=/usr/local \
    CLANG_EXECUTABLE_PATH=/usr/bin/clang++-${CLANG_VERSION} \
    LLVM_DIR=/usr/lib/llvm-${CLANG_VERSION}/cmake \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
WORKDIR /build
RUN git clone --filter=blob:none --depth 1 --branch "${ACPP_REF}" "${ACPP_REPO}" AdaptiveCpp || \
    (git clone --filter=blob:none "${ACPP_REPO}" AdaptiveCpp && git -C AdaptiveCpp fetch --depth 1 origin "${ACPP_REF}" && git -C AdaptiveCpp checkout FETCH_HEAD) && \
    git -C AdaptiveCpp rev-parse HEAD | tee /usr/local/share/acpp-commit && \
    git -C AdaptiveCpp describe --always --tags --dirty | tee /usr/local/share/acpp-describe && \
    cmake -S AdaptiveCpp -B AdaptiveCpp/build -G Ninja \
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
          -DCMAKE_INSTALL_PREFIX=${ADAPTIVE_CPP_INSTALL_DIR} \
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
          -DCMAKE_C_FLAGS="-march=x86-64-v2" \
          -DCMAKE_CXX_FLAGS="-march=x86-64-v2" \
          -DCLANG_EXECUTABLE_PATH=${CLANG_EXECUTABLE_PATH} \
          -DLLVM_DIR=${LLVM_DIR} \
          -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
          -DWITH_CUDA_BACKEND=ON \
          -DWITH_OPENCL_BACKEND=OFF \
          -DWITH_ROCM_BACKEND=OFF \
          -DWITH_LEVEL_ZERO_BACKEND=OFF \
          -DACPP_COMPILER_FEATURE_PROFILE=${ACPP_COMPILER_FEATURE_PROFILE} && \
    cmake --build AdaptiveCpp/build -j "$(nproc)" && \
    cmake --install AdaptiveCpp/build && \
    rm -rf AdaptiveCpp

# CI compiler image: AdaptiveCpp + LLVM + CMake/Ninja + Boost 1.89 (static,
# PIC) so `cmake --preset generic|cuda-sm70` does not FetchContent-compile
# Boost. Not copied into Dockerfile.cuda12.8 scramruntime.
FROM adaptivecpp-clang AS toolchain
ARG ACPP_COMPILER_FEATURE_PROFILE=full
ARG BOOST_VERSION=1.89.0
ARG BOOST_URL=https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.gz
ARG BOOST_SHA256=954a01219bf818c7fb850fa610c2c8c71a4fa28fa32a1900056bcb6ff58cf908
# Must match CMakeLists.txt BOOST_INCLUDE_LIBRARIES.
ARG BOOST_INCLUDE_LIBRARIES="program_options;filesystem;system;random;range;exception;multi_index;accumulators;multiprecision;icl;math;dll;regex;test"

# Refuse a SMCP-only AdaptiveCpp install: `cmake --preset generic` needs SSCP.
# Checks the built compiler, not the ARG default (stage ARG defaults can differ).
RUN python3 -c 'import json,sys; d=json.load(open("/usr/local/etc/AdaptiveCpp/acpp-core.json")); v=d.get("plugin-with-sscp-compiler"); sys.exit(0 if v=="true" else 1)'

WORKDIR /tmp/boost
RUN wget -q -O boost.tar.gz "${BOOST_URL}" && \
    echo "${BOOST_SHA256}  boost.tar.gz" | sha256sum -c - && \
    mkdir src && tar -xzf boost.tar.gz -C src --strip-components=1 && \
    cmake -S src -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_C_FLAGS="-march=x86-64-v2" \
          -DCMAKE_CXX_FLAGS="-march=x86-64-v2" \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DBUILD_SHARED_LIBS=OFF \
          -DBOOST_INCLUDE_LIBRARIES="${BOOST_INCLUDE_LIBRARIES}" && \
    cmake --build build -j "$(nproc)" && \
    cmake --install build && \
    mkdir -p /usr/local/include/boost && \
    cp -a src/libs/dll/include/boost/dll.hpp /usr/local/include/boost/dll.hpp && \
    cp -a src/libs/dll/include/boost/dll /usr/local/include/boost/dll && \
    rm -rf /tmp/boost

ENV CMAKE_PREFIX_PATH=/usr/local \
    AdaptiveCpp_DIR=/usr/local/lib/cmake/AdaptiveCpp \
    ACPP_COMPILER_FEATURE_PROFILE=${ACPP_COMPILER_FEATURE_PROFILE}

WORKDIR /src

# Declare revision/hash ARGs after Boost so a new fuzzybits commit does not
# rebuild Boost. CI still passes them as --build-arg.
ARG ACPP_REF=v25.10.0
ARG ACPP_COMMIT=9f842c701a599107cc6d117d3539f971036363a1
ARG FUZZYBITS_REVISION=unknown
ARG TOOLCHAIN_SHA=unknown
ARG CLANG_VERSION=20
LABEL org.opencontainers.image.source="https://github.com/fuzzybit-org/fuzzybits" \
      org.opencontainers.image.revision="${FUZZYBITS_REVISION}" \
      org.opencontainers.image.version="${ACPP_REF}" \
      org.opencontainers.image.title="fuzzybits-toolchain" \
      org.opencontainers.image.description="LLVM ${CLANG_VERSION} + AdaptiveCpp ${ACPP_REF} (full SSCP + CUDA SMCP) for cmake --preset generic|cuda-sm70" \
      dev.fuzzybits.acpp.commit="${ACPP_COMMIT}" \
      dev.fuzzybits.toolchain.sha="${TOOLCHAIN_SHA}" \
      dev.fuzzybits.acpp.profile="${ACPP_COMPILER_FEATURE_PROFILE}"
