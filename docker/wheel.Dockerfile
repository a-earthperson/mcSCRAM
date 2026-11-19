# syntax=docker/dockerfile:1.10.0
#
# Build a self-contained fuzzybits wheel from an AdaptiveCpp+clang toolchain
# image (Dockerfile.cuda12.8 --target adaptivecpp-clang, full profile).
#
#   docker --context xerxes build -f Dockerfile.cuda12.8 \
#     --build-arg ACPP_COMPILER_FEATURE_PROFILE=full \
#     --target adaptivecpp-clang -t fuzzybits-acpp:full .
#
#   docker --context xerxes build -f docker/wheel.Dockerfile \
#     --build-arg TOOLCHAIN_IMAGE=fuzzybits-acpp:full \
#     --build-arg FUZZYBITS_VARIANT=generic \
#     --output type=local,dest=dist/generic .
#
# FUZZYBITS_VARIANT=generic | cuda-sm70
# Remote daemons cannot bind-mount the worktree; COPY the source in.

ARG TOOLCHAIN_IMAGE=fuzzybits-acpp:full
FROM ${TOOLCHAIN_IMAGE} AS wheel-build

ARG FUZZYBITS_VARIANT=generic
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive \
    FUZZYBITS_VARIANT=${FUZZYBITS_VARIANT} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      python3-packaging \
      patchelf \
      binutils \
      ca-certificates \
      libatomic1 \
      git

COPY . /src
WORKDIR /src

# Isolated venv: Ubuntu 24.04 is PEP 668 managed.
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
      packaging

COPY docker/pip-wheel.sh docker/auditwheel-repair.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/pip-wheel.sh /usr/local/bin/auditwheel-repair.sh && \
    mkdir -p /out/wheelhouse /out/dist && \
    PATH="/opt/wheel-venv/bin:${PATH}" /usr/local/bin/pip-wheel.sh /out/wheelhouse && \
    ls -la /out/wheelhouse

# auditwheel on Ubuntu 24.04 emits manylinux_2_39. Exclude the NVIDIA driver
# and AdaptiveCpp/JIT libs we already laid out under fuzzybits/lib/ (moving
# them to .libs would break plugin discovery relative to libacpp-rt).
RUN PATH="/opt/wheel-venv/bin:${PATH}" /usr/local/bin/auditwheel-repair.sh /out/wheelhouse /out/dist && \
    ls -lah /out/dist && \
    python3 - <<'PY'
from pathlib import Path
for p in sorted(Path("/out/dist").glob("*.whl")):
    print(f"WHEEL {p.name} {p.stat().st_size} bytes ({p.stat().st_size/1e6:.1f} MB)")
PY

FROM scratch AS export
COPY --from=wheel-build /out/dist/ /
