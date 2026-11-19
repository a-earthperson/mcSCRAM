#!/usr/bin/env bash
# pip-install the released engine + server wheels into a clean Ubuntu 24.04
# venv, start `fuzzybits-server`, and check the UI and a chinese sample.
# Intended for `docker run -v "$PWD:/src" -w /src ubuntu:24.04` with the
# wheels downloaded into ./wheels.
set -euo pipefail
cd /src
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip libgomp1 libstdc++6 libatomic1 libomp5 >/dev/null

shopt -s nullglob
engine=(wheels/fuzzybits-*manylinux*.whl)
server=(wheels/fuzzybits_server-*.whl)
if [[ ${#engine[@]} -ne 1 || ${#server[@]} -ne 1 ]]; then
  echo "error: expected one engine and one server wheel in wheels/" >&2
  ls -la wheels >&2
  exit 1
fi

python3 -m venv /tmp/venv
/tmp/venv/bin/pip install -q -U pip
/tmp/venv/bin/pip install -q "${engine[0]}" "${server[0]}"

export ACPP_VISIBILITY_MASK=omp
/tmp/venv/bin/fuzzybits-server --host 127.0.0.1 --port 8000 --data-dir input/Aralia \
  >/tmp/server.log 2>&1 &
trap 'kill %1 2>/dev/null || true' EXIT

if ! /tmp/venv/bin/python - <<'PY'; then
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
REF = 1.17058e-3  # chinese, BDD


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.status, resp.read()


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


deadline = time.time() + 60
while True:
    try:
        get("/api/v1/healthz", timeout=2)
        break
    except Exception:
        if time.time() > deadline:
            raise SystemExit("server did not become healthy within 60 s")
        time.sleep(1)

status, html = get("/")
asset = re.search(rb'src="(/assets/[^"]+)"', html)
assert status == 200 and asset, f"UI / unexpected: {status} {html[:200]!r}"
assert get(asset.group(1).decode())[0] == 200
print(f"ui ok  asset={asset.group(1).decode()}")

get("/api/v1/model/dag?model_id=chinese", timeout=120)
accepted = post("/api/v1/start", {"model_id": "chinese", "num_trials": 1e7,
                                  "requested_trials_per_pass": 1e6,
                                  "early_stop": False, "policy": "wald"})
assert accepted.get("accepted"), accepted
deadline = time.time() + 180
while time.time() < deadline:
    st = json.loads(get("/api/v1/status")[1])
    if st.get("status") in ("completed", "failed", "stopped"):
        break
    time.sleep(0.5)
assert st.get("status") == "completed", st
update, result = st.get("last_update") or {}, st.get("last_result") or {}
mean = result.get("mean", update.get("mean"))
se = update.get("std_err", result.get("std_err"))
z = (mean - REF) / se
print(f"chinese p={mean:.6e} ref={REF:.6e} se={se:.3e} z={z:+.2f}")
assert abs(z) <= 4, "chinese estimate outside 4 sigma"
PY
  echo "---- server log ----" >&2
  cat /tmp/server.log >&2
  exit 1
fi
echo "server smoke ok"
