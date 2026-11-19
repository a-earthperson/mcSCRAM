# syntax=docker/dockerfile:1.10.0
#
# Thin source overlay for measuring
#   cmake --preset generic|cuda-sm70 && cmake --build --preset ...
# on top of a toolchain image. Not published.
ARG TOOLCHAIN=fuzzybits-toolchain:v25.10.0-generic
FROM ${TOOLCHAIN}
COPY . /src
WORKDIR /src
