# syntax=docker/dockerfile:1.10.0
#
# FastAPI server image on top of the slim fuzzybits runtime (#46 step E).
# Does NOT rebuild AdaptiveCpp / scram-cli / the fuzzybits wheel: those come
# from RUNTIME_IMAGE. uv installs fuzzybits-server into a venv with
# --system-site-packages so `import fuzzybits` hits
# the runtime's system Python 3.12 site.
#
# UI: Node build stage emits CSR assets into /app/server/webui/dist.
# No Node at runtime. Disable with FUZZYBITS_SERVER_UI=0.
#
#   docker build -f tools/server/Dockerfile \
#     --build-arg RUNTIME_IMAGE=ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic \
#     --build-arg VARIANT=generic \
#     -t ghcr.io/fuzzybit-org/fuzzybits-server:ci-local-generic \
#     tools/server
#
# Context is tools/server/ (pyproject.toml, uv.lock, src/, scripts/, webui/).

ARG RUNTIME_IMAGE=ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142

FROM node:22-bookworm-slim AS webui
WORKDIR /webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ ./
RUN npm run build

FROM ${UV_IMAGE} AS uv
FROM ${RUNTIME_IMAGE}

ARG RUNTIME_IMAGE=ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic
ARG RUNTIME_DIGEST=unknown
ARG VARIANT=generic
ARG FUZZYBITS_REVISION=unknown
ARG FUZZYBITS_VERSION=0.18.4
ARG SERVER_VERSION=0.18.4

LABEL org.opencontainers.image.source="https://github.com/fuzzybit-org/fuzzybits" \
      org.opencontainers.image.revision="${FUZZYBITS_REVISION}" \
      org.opencontainers.image.version="${FUZZYBITS_VERSION}" \
      org.opencontainers.image.title="fuzzybits-server" \
      org.opencontainers.image.description="Compile OpenPSA/MEF fault trees to .pcfb and sample the top event" \
      org.opencontainers.image.base.name="${RUNTIME_IMAGE}" \
      org.opencontainers.image.base.digest="${RUNTIME_DIGEST}" \
      dev.fuzzybits.variant="${VARIANT}" \
      dev.fuzzybits.server.version="${SERVER_VERSION}"

USER root

COPY --from=uv /uv /uvx /usr/local/bin/

ENV HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/server/.venv \
    PATH="/app/server/.venv/bin:/usr/local/bin:${PATH}" \
    FUZZYBITS_SERVER_DATA_DIR=/data \
    FUZZYBITS_SERVER_CACHE_DIR=/home/coder/.cache/fuzzybits/server \
    FUZZYBITS_SERVER_UI_DIR=/app/server/webui/dist \
    FUZZYBITS_SERVER_UI=1 \
    ACPP_VISIBILITY_MASK=omp \
    ACPP_APPDB_DIR=/home/coder/.cache/fuzzybits/acpp

RUN mkdir -p /app/server /data \
      /home/coder/.cache/fuzzybits/server \
      /home/coder/.cache/fuzzybits/acpp && \
    chown -R coder:coder /app/server /data /home/coder/.cache/fuzzybits

WORKDIR /app/server
COPY --chown=coder:coder pyproject.toml uv.lock hatch_build.py README.md ./
COPY --chown=coder:coder src ./src
COPY --chown=coder:coder scripts/docker-entrypoint.sh /usr/local/bin/server-entrypoint
COPY --from=webui --chown=coder:coder /webui/dist /app/server/webui/dist
RUN chmod 0755 /usr/local/bin/server-entrypoint /usr/local/bin/uv /usr/local/bin/uvx

USER coder
# --system-site-packages: the runtime pip-installed fuzzybits wheel lives in
# the image's dist-packages. A .pth fallback covers Debian venvs that miss
# /usr/local/lib/python3.12/dist-packages even with the flag.
RUN set -eux; \
    uv venv --system-site-packages /app/server/.venv; \
    uv sync --frozen --no-dev --no-install-package fuzzybits; \
    venv_site="$(/app/server/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"; \
    sys_site="$(python3 -c 'import fuzzybits, pathlib; p=pathlib.Path(fuzzybits.__file__).resolve(); print(p.parent if p.name != "__init__.py" else p.parent.parent)')"; \
    printf '%s\n' "${sys_site}" > "${venv_site}/zz-system-fuzzybits.pth"; \
    /app/server/.venv/bin/python -c \
      "import fuzzybits, fuzzybits_server, uvicorn; import importlib.util as u; \
assert u.find_spec('fuzzybits'); \
assert u.find_spec('fuzzybits_server'); \
native = fuzzybits.native(); \
assert hasattr(native, 'compile_pcfb'), 'runtime image lacks compile_pcfb (#71)'; \
assert hasattr(fuzzybits, 'compile') and hasattr(fuzzybits, 'inspect') and hasattr(fuzzybits, 'Session'), 'runtime image lacks fuzzybits.compile/inspect/Session (#71)'"

VOLUME ["/data", "/home/coder/.cache/fuzzybits"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python3 -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/v1/healthz' % os.environ.get('PORT','8000'), timeout=3)"

STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/local/bin/server-entrypoint"]
