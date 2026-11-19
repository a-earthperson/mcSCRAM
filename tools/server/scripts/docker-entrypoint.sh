#!/bin/sh
# uvicorn wrapper so HOST / PORT / LOG_LEVEL stay env-driven at runtime.
# Keep this a plain exec (no extra shell loop) so signals reach uvicorn.
# Docker binds 0.0.0.0; the `fuzzybits-server` CLI defaults to 127.0.0.1.
exec uvicorn fuzzybits_server.app:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --log-level "${LOG_LEVEL:-info}"
