# syntax=docker/dockerfile:1.10.0
# Clean-runtime acceptance image: no AdaptiveCpp/clang toolchain.
# Build context is docker/wheel-test-context/ (populated by the host script).
# Ubuntu 24.04 / glibc 2.39 matches manylinux_2_39. python:3.12-slim is
# currently Debian bookworm (glibc 2.36) and rejects the wheel.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      libstdc++6 \
      libgomp1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY *.whl /tmp/
COPY models /models
COPY tests /tests
COPY wheel-smoke.sh /usr/local/bin/wheel-smoke.sh
RUN chmod +x /usr/local/bin/wheel-smoke.sh && \
    pip install --no-cache-dir /tmp/*.whl pytest && \
    rm -f /tmp/*.whl

WORKDIR /tests
CMD ["pytest", "-q"]
