#!/bin/sh
# uvicorn wrapper so HOST / PORT / LOG_LEVEL stay env-driven at runtime.
# Keep this a plain exec (no extra shell loop) so signals reach uvicorn.
exec uvicorn mcscram_server.app:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --log-level "${LOG_LEVEL:-info}"
