#!/usr/bin/env bash
# Smoke one manylinux wheel in a clean Ubuntu 24.04 container.
# PYTHON_VERSION=3.12 (default). Expects dist/fuzzybits-*-cpXYY-*.manylinux*.whl
set -euo pipefail
cd /src

py="${PYTHON_VERSION:-3.12}"
tag="cp${py//./}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  curl ca-certificates libgomp1 libstdc++6 libatomic1 libomp5 >/dev/null
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
uv python install "${py}"
venv="/tmp/fb-venv-${py}"
rm -rf "${venv}"
uv venv --python "${py}" "${venv}"
# shellcheck disable=SC1091
. "${venv}/bin/activate"

shopt -s nullglob
whls=(dist/fuzzybits-*-"${tag}"-*.whl)
if [[ ${#whls[@]} -ne 1 ]]; then
  echo "error: expected one ${tag} wheel in dist/, got ${#whls[@]}" >&2
  ls -la dist >&2
  exit 1
fi
whl="${whls[0]}"
python -m pip install -U pip >/dev/null
python -m pip install "${whl}"
export ACPP_VISIBILITY_MASK=omp
export FUZZYBITS_BACKEND=omp

python - <<'PY'
import fuzzybits

print("version", fuzzybits.__version__)
print("variant", fuzzybits.active_variant())
devices = fuzzybits.list_devices()
print("devices", devices)
if not devices:
    raise SystemExit("list_devices() returned empty")
print("import+list_devices ok")
PY

xml=input/Aralia/chinese.xml
pcfb=/tmp/chinese.pcfb
test -f "${xml}"
fzb compile "${xml}" -o "${pcfb}"
timeout 180 fzb run "${pcfb}" --trials 1e7 --backend omp --json >/tmp/chinese.json

python - <<'PY'
import json
import math
import sys

REF = 1.17058e-3
Z_MAX = 4.0
payload = json.loads(open("/tmp/chinese.json", encoding="utf-8").read())
queries = payload.get("queries") or []
if not queries:
    sys.stderr.write("error: no queries in fzb --json output\n")
    sys.exit(3)
row = queries[0]
phat = float(row["estimate"])
n = float(row["n"])
if n <= 0:
    sys.stderr.write(f"error: n={n} from fzb run\n")
    sys.exit(3)
se = math.sqrt(max(phat * (1.0 - phat), 0.0) / n)
z = (phat - REF) / se if se > 0 else float("inf")
ok = abs(z) <= Z_MAX
print(
    f"chinese  p̂={phat:.6e}  p_ref={REF:.6e}  |z|={abs(z):.3f}  "
    f"n={int(n)}  ok={ok}"
)
sys.exit(0 if ok else 1)
PY
echo "linux ${tag} smoke ok (${whl})"
