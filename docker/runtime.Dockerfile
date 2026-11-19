# syntax=docker/dockerfile:1.10.0
#
# Slim fuzzybits runtime image for GHCR (#35).
#
# Built on top of the CI toolchain image (LLVM 20 + AdaptiveCpp full + Boost
# 1.89 + CUDA 12.8 devel) so hosted-runner jobs do not rebuild AdaptiveCpp.
# The final image is NOT the toolchain: it is ubuntu:24.04 plus
#   scram-cli, fzb, AdaptiveCpp runtime, optional SSCP JIT, pip-installed
#   fuzzybits wheel, acpp-info, mcscram-smoke.
#
# Why not FROM ${CUDA_RUNTIME_IMAGE}? nvidia/cuda:*-runtime is ~3 GB because
# of libcublas/cufft/cusolver/… AdaptiveCpp only needs libcudart; the NVIDIA
# *driver* (libcuda.so.1) is injected by nvidia-container-toolkit (`--gpus`).
# Copying cudart from the toolchain (same CUDA 12.8.1 pin) keeps generic
# ≤ ~4 GB and cuda-sm70 ≤ ~3.5 GB.
#
# ARG defaults match toolchain.env. CI passes --build-arg from that file.
#
#   docker build -f docker/runtime.Dockerfile \
#     --build-arg TOOLCHAIN_IMAGE=ghcr.io/fuzzybit-org/fuzzybits-toolchain:v25.10.0-generic \
#     --build-arg CMAKE_PRESET=generic \
#     --build-arg FUZZYBITS_VARIANT=generic \
#     --build-arg INCLUDE_SSCP_JIT=1 \
#     --build-arg ACPP_TARGETS=generic \
#     -t ghcr.io/fuzzybit-org/fuzzybits-runtime:generic .

ARG TOOLCHAIN_IMAGE=ghcr.io/fuzzybit-org/fuzzybits-toolchain:v25.10.0-generic
FROM ${TOOLCHAIN_IMAGE} AS builder

ARG CMAKE_PRESET=generic
ARG FUZZYBITS_VARIANT=generic
ARG INCLUDE_SSCP_JIT=1
ARG CLANG_VERSION=20
ARG BUILD_TESTING=OFF
ARG CUDA_EMBED_PTX=highest
ARG ACPP_TARGETS=generic

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    FUZZYBITS_VARIANT=${FUZZYBITS_VARIANT} \
    CMAKE_PRESET=${CMAKE_PRESET} \
    INCLUDE_SSCP_JIT=${INCLUDE_SSCP_JIT} \
    CLANG_VERSION=${CLANG_VERSION} \
    ACPP_TARGETS=${ACPP_TARGETS} \
    CUDA_EMBED_PTX=${CUDA_EMBED_PTX}

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      python3-venv \
      python3-dev \
      python3-packaging \
      patchelf \
      binutils \
      libatomic1

COPY . /src
WORKDIR /src

# Product CLI (scram-cli, fzb). BUILD_PYTHON_BINDINGS stays ON from the
# preset; the wheel is a separate SKBUILD so the CLI is not SKBUILD-stripped.
RUN cmake --preset "${CMAKE_PRESET}" \
          -DBUILD_TESTING="${BUILD_TESTING}" \
          -DBUILD_TESTS="${BUILD_TESTING}" \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_INSTALL_RPATH="/usr/local/lib;/usr/local/cuda/lib64" && \
    cmake --build --preset "${CMAKE_PRESET}" --parallel "$(nproc)" && \
    cmake --install "build/${CMAKE_PRESET}"

# Wheel (scikit-build-core). SKBUILD skips fzb; that is why the CLI build
# above is a separate compile. Hosted 4-core budget is ~8 min each; they
# share the toolchain layers.
RUN python3 -m venv /opt/wheel-venv && \
    /opt/wheel-venv/bin/pip install -U pip && \
    /opt/wheel-venv/bin/pip install \
      'scikit-build-core>=0.10' \
      'pybind11>=2.13' \
      ninja \
      cmake \
      auditwheel \
      patchelf \
      build \
      packaging && \
    mkdir -p /out/wheelhouse /out/dist && \
    PATH="/opt/wheel-venv/bin:${PATH}" /src/docker/pip-wheel.sh /out/wheelhouse && \
    PATH="/opt/wheel-venv/bin:${PATH}" /src/docker/auditwheel-repair.sh /out/wheelhouse /out/dist && \
    ls -lah /out/dist

RUN chmod +x /src/docker/stage-runtime.sh /src/docker/image-smoke.sh \
      /src/docker/smoke.sh /src/docker/isa-baseline-check.sh && \
    DEST=/opt/runtime-root /src/docker/stage-runtime.sh && \
    /src/docker/isa-baseline-check.sh /opt/runtime-root /out/dist

# -----------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG USER=coder
ARG UID=1000
ARG GID=1000
ARG CLANG_VERSION=20
ARG ACPP_REF=v25.10.0
ARG ACPP_COMMIT=9f842c701a599107cc6d117d3539f971036363a1
ARG ACPP_TARGETS=generic
ARG CUDA_EMBED_PTX=n/a
ARG CUDA_VERSION=12.8.1
ARG FUZZYBITS_REVISION=unknown
ARG FUZZYBITS_VERSION=0.17.0
ARG INCLUDE_SSCP_JIT=1
ARG CMAKE_PRESET=generic

ENV DEBIAN_FRONTEND=noninteractive \
    ACPP_VISIBILITY_MASK=cuda \
    ACPP_DEBUG_LEVEL=0 \
    ACPP_ADAPTIVITY_LEVEL=2 \
    ACPP_ALLOCATION_TRACKING=0 \
    ACPP_PERSISTENT_RUNTIME=1 \
    ACPP_APPDB_DIR=/home/coder/.cache/fuzzybits/acpp \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

LABEL org.opencontainers.image.source="https://github.com/fuzzybit-org/fuzzybits" \
      org.opencontainers.image.revision="${FUZZYBITS_REVISION}" \
      org.opencontainers.image.version="${FUZZYBITS_VERSION}" \
      org.opencontainers.image.title="fuzzybits-runtime" \
      org.opencontainers.image.description="slim fuzzybits runtime (scram-cli, fzb, fuzzybits Python)" \
      dev.fuzzybits.acpp.ref="${ACPP_REF}" \
      dev.fuzzybits.acpp.commit="${ACPP_COMMIT}" \
      dev.fuzzybits.acpp.targets="${ACPP_TARGETS}" \
      dev.fuzzybits.cuda.embed_ptx="${CUDA_EMBED_PTX}" \
      dev.fuzzybits.cuda.version="${CUDA_VERSION}" \
      dev.fuzzybits.clang.version="${CLANG_VERSION}" \
      dev.fuzzybits.sscp="${INCLUDE_SSCP_JIT}" \
      dev.fuzzybits.cmake.preset="${CMAKE_PRESET}"

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libstdc++6 \
      libc6 \
      libgcc-s1 \
      libatomic1 \
      libedit2 \
      libzstd1 \
      zlib1g \
      liblzma5 \
      libhwloc15 \
      libnuma1 \
      libtinfo6 \
      libgomp1 \
      libffi8 \
      libxml2 \
      python3 \
      python3-pip \
      python3-venv && \
    apt-mark manual libxml2 && \
    if getent passwd ${UID}; then userdel -f "$(getent passwd ${UID} | cut -d: -f1)"; fi && \
    if getent group ${GID}; then groupdel "$(getent group ${GID} | cut -d: -f1)"; fi && \
    groupadd -g ${GID} ${USER} && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USER}

COPY --from=builder /opt/runtime-root /
COPY --from=builder /out/dist /tmp/wheels
RUN ls -lh /usr/local/bin /usr/local/lib && \
    test -x /usr/local/bin/scram-cli && test -x /usr/local/bin/fzb && \
    ls /usr/local/lib/libscram.so* /usr/local/lib/libmc.so* \
       /usr/local/lib/libacpp-rt.so* /usr/local/lib/libacpp-common.so* \
       /usr/local/lib/libxml2.so*

RUN python3 -m pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels && \
    apt-get purge -y python3-pip python3-venv && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-recommends libxml2 && \
    apt-mark manual libxml2 && \
    printf '%s\n' /usr/local/lib /usr/local/cuda/lib64 > /etc/ld.so.conf.d/fuzzybits.conf && \
    ldconfig && \
    mkdir -p /home/${USER}/.cache/fuzzybits/acpp && \
    chown -R ${USER}:${USER} /app /home/${USER} && \
    chmod 0755 /usr/local/bin/scram-cli /usr/local/bin/fzb \
      /usr/local/bin/mcscram-smoke /usr/local/bin/fuzzybits-image-smoke \
      /usr/local/bin/scram-or-fzb-entrypoint && \
    test -x /usr/local/bin/scram-cli && \
    test -x /usr/local/bin/fzb && \
    python3 -c "import fuzzybits, importlib.util; assert importlib.util.find_spec('mcscram_native')" && \
    test -z "${PYTHONPATH:-}" && \
    test -z "${LD_PRELOAD:-}" && \
    test -e /lib/x86_64-linux-gnu/libxml2.so.2 && \
    missing="$(ldd /usr/local/lib/hipSYCL/librt-backend-omp.so \
                   /usr/local/lib/hipSYCL/librt-backend-cuda.so \
                   /usr/local/lib/hipSYCL/llvm-to-backend/*.so \
                 | grep 'not found' | grep -v 'libcuda.so' || true)" && \
    if [ -n "${missing}" ]; then echo "${missing}" >&2; exit 1; fi && \
    ACPP_VISIBILITY_MASK=omp acpp-info -l && \
    scram-cli --help >/dev/null && \
    fzb --help >/dev/null

USER ${USER}
WORKDIR /app
ENTRYPOINT ["/usr/local/bin/scram-or-fzb-entrypoint"]
CMD ["--help"]
