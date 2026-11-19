#!/usr/bin/env bash
# In-image smoke: mcscram-smoke + fzb compile/run (chinese) + Python import.
# Every Monte Carlo invocation is wrapped in `timeout` (default 60s).
set -euo pipefail

TIMEOUT_SEC="${SMOKE_TIMEOUT:-60}"
FZB_TRIALS="${FZB_TRIALS:-1e8}"
INPUT_DIR="${SMOKE_INPUT_DIR:-/app/input}"

echo "fuzzybits-image-smoke  timeout=${TIMEOUT_SEC}s  fzb_trials=${FZB_TRIALS}"
echo "uid=$(id -u) user=$(id -un) ACPP_APPDB_DIR=${ACPP_APPDB_DIR:-unset}"
test "$(id -u)" -ne 0
test -n "${ACPP_APPDB_DIR:-}"
test -w "${ACPP_APPDB_DIR}"
test -z "${PYTHONPATH:-}"
test -z "${LD_PRELOAD:-}"

echo "== binaries =="
command -v scram-cli
command -v fzb
command -v acpp-info
command -v mcscram-smoke
scram-cli --help >/dev/null
fzb --help >/dev/null
acpp-info -l || true

echo "== python (import mcscram_native shim, no PYTHONPATH/LD_PRELOAD) =="
python3 - <<'PY'
import os, sys
assert not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset"
assert not os.environ.get("LD_PRELOAD"), "LD_PRELOAD must be unset"
import fuzzybits
fuzzybits.init()
print("fuzzybits", fuzzybits.__version__,
      "variant", fuzzybits.active_variant(),
      "backend", fuzzybits.active_backend())
print("devices", fuzzybits.list_devices())
import mcscram_native
assert hasattr(mcscram_native, "run_probability")
print("mcscram_native ok")
PY

echo "== mcscram-smoke =="
SMOKE_TIMEOUT="${TIMEOUT_SEC}" mcscram-smoke

echo "== fzb compile + run chinese =="
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
fzb compile "${INPUT_DIR}/chinese.xml" -c 2 --no-kn -o "${tmp}/chinese.pcfb"
timeout "${TIMEOUT_SEC}" fzb run "${tmp}/chinese.pcfb" --trials "${FZB_TRIALS}" --backend cuda --json

echo "IMAGE SMOKE OK"
