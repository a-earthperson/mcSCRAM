#!/usr/bin/env bash
# Install the generic wheel into a clean Ubuntu 24.04 venv and run tests/python.
# Intended for `docker run ubuntu:24.04` — no AdaptiveCpp toolchain.
set -euo pipefail
cd /src
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip libgomp1 libstdc++6 libatomic1 libomp5 >/dev/null
shopt -s nullglob
whls=(dist/fuzzybits-*.whl)
if [[ ${#whls[@]} -eq 0 ]]; then
  echo "error: no wheel in dist/" >&2
  exit 1
fi
whl="${whls[0]}"
python3 -m venv /tmp/fb-venv
/tmp/fb-venv/bin/pip install -U pip >/dev/null
/tmp/fb-venv/bin/pip install "${whl}" pytest
# Drop pyproject pythonpath so the *installed* package is imported, not python/.
export ACPP_VISIBILITY_MASK=omp
export FUZZYBITS_BACKEND=omp
/tmp/fb-venv/bin/pytest tests/python -q -m 'not gpu' -o pythonpath=
