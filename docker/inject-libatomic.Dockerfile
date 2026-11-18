# syntax=docker/dockerfile:1.10.0
# Drop libatomic.so.1 into an already-built wheel (gcc runtime; AdaptiveCpp OMP).
FROM ubuntu:24.04 AS inject
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      unzip zip patchelf libatomic1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY *.whl /tmp/wheels/
RUN mkdir -p /w /out && \
    whl="$(ls /tmp/wheels/*.whl)" && \
    unzip -q "${whl}" -d /w && \
    src="$(readlink -f /usr/lib/x86_64-linux-gnu/libatomic.so.1)" && \
    cp -a "${src}" /w/fuzzybits/lib/libatomic.so.1 && \
    patchelf --set-rpath '$ORIGIN' /w/fuzzybits/lib/libatomic.so.1 && \
    name="$(basename "${whl}")" && \
    (cd /w && zip -qr "/out/${name}" .)
FROM scratch
COPY --from=inject /out/ /
