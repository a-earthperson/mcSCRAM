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
# PR CI and the release 3.12 job leave one cp312 wheel; a multi-version dist/
# must still pick cp312 so ubuntu:24.04's system Python can import it.
whls=(dist/fuzzybits-*-cp312-*.whl)
if [[ ${#whls[@]} -eq 0 ]]; then
  whls=(dist/fuzzybits-*.whl)
fi
if [[ ${#whls[@]} -eq 0 ]]; then
  echo "error: no wheel in dist/" >&2
  exit 1
fi
if [[ ${#whls[@]} -gt 1 ]]; then
  echo "error: expected one cp312 wheel in dist/, got ${#whls[@]}" >&2
  ls -la dist >&2
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

# Issue #45 Python smoke: exact public API, timeout 60 inside this container.
echo "python spine: chinese compile → run (omp, timeout 60)"
timeout 60 /tmp/fb-venv/bin/python - <<'PY'
import math
import sys

import fuzzybits

REF = 1.17058e-3
Z_MAX = 4.0
TRIALS = 1 << 20  # 2^20: chinese finishes in seconds on hosted CPU

pcfb = fuzzybits.compile(["input/Aralia/chinese.xml"])
summary = fuzzybits.inspect(pcfb)
if summary.get("diagnostics"):
    print("inspect diagnostics:", summary["diagnostics"], file=sys.stderr)
    sys.exit(1)
rows = fuzzybits.run(pcfb, trials=TRIALS, backend="omp")
if not rows:
    print("error: fuzzybits.run returned no queries", file=sys.stderr)
    sys.exit(1)
row = rows[0]
n = float(row.n)
sigma = math.sqrt(max(row.estimate * (1.0 - row.estimate), 0.0) / n) if n else float("nan")
z = (row.estimate - REF) / sigma if sigma > 0 else float("inf")
ok = abs(z) <= Z_MAX
print(
    f"chinese  p̂={row.estimate:.6e}  p_ref={REF:.6e}  |z|={abs(z):.3f}  "
    f"n={row.n}  ok={ok}"
)
sys.exit(0 if ok else 1)
PY
