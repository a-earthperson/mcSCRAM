#!/usr/bin/env bash
# Smoke an installed `fuzzybits-server` wheel (packaged UI + API + chinese).
#
# Starts `fuzzybits-server --data-dir <Aralia>` in this process namespace,
# then:
#   1. GET /  (playground HTML) and one hashed /assets/* file
#   2. GET /api/v1/healthz
#   3. compile+sample chinese; |p̂ − 1.17058e-3| ≤ 4·σ̂
#
# Usage:
#   tools/server/scripts/wheel-smoke.sh [ARALIA_DIR]
#
# Env: SMOKE_N (default 1e6), SMOKE_TPP (default 1e5), SMOKE_TIMEOUT (90),
#      SMOKE_Z_MAX (4), SMOKE_HOST (127.0.0.1), SMOKE_PORT (8000).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARALIA="${1:-${SMOKE_ARALIA:-${ROOT}/input/Aralia}}"
HOST="${SMOKE_HOST:-127.0.0.1}"
PORT="${SMOKE_PORT:-8000}"
N="${SMOKE_N:-1e6}"
TPP="${SMOKE_TPP:-100000}"
TIMEOUT_SEC="${SMOKE_TIMEOUT:-90}"
Z_MAX="${SMOKE_Z_MAX:-4}"
BASE="http://${HOST}:${PORT}"

if [[ ! -d "${ARALIA}" ]]; then
  echo "error: Aralia models not found at ${ARALIA}" >&2
  exit 2
fi
if [[ ! -f "${ARALIA}/chinese.xml" ]]; then
  echo "error: expected chinese.xml under ${ARALIA}" >&2
  exit 2
fi
if ! command -v fuzzybits-server >/dev/null 2>&1; then
  echo "error: fuzzybits-server not on PATH" >&2
  exit 2
fi

server_pid=""
cleanup() {
  local rc=$?
  if [[ -n "${server_pid}" ]]; then
    kill -INT "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
  exit "${rc}"
}
trap cleanup EXIT

echo "wheel-smoke  data-dir=${ARALIA}  N=${N}  tpp=${TPP}  timeout=${TIMEOUT_SEC}s  |z|<=${Z_MAX}"

fuzzybits-server \
  --host "${HOST}" \
  --port "${PORT}" \
  --data-dir "${ARALIA}" \
  --backend omp \
  --engine pcfb \
  &
server_pid=$!

python3 - "${BASE}" "${TIMEOUT_SEC}" <<'PY'
import sys, time, urllib.request, urllib.error
base, timeout_s = sys.argv[1], float(sys.argv[2])
deadline = time.time() + min(timeout_s, 60.0)
last = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base}/api/v1/healthz", timeout=3) as resp:
            print(f"healthz ok  {resp.read().decode().strip()}", flush=True)
            sys.exit(0)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        last = str(exc)
        time.sleep(0.4)
sys.stderr.write(f"error: healthz not ready after wait: {last}\n")
sys.exit(1)
PY

python3 - "${BASE}" "${N}" "${TPP}" "${TIMEOUT_SEC}" "${Z_MAX}" <<'PY'
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

base, n_s, tpp_s, timeout_s, zmax_s = sys.argv[1:6]
n = float(n_s)
tpp = float(tpp_s)
timeout = float(timeout_s)
zmax = float(zmax_s)
REF = 1.17058e-3


class ApiError(RuntimeError):
    pass


def _request(method, path, *, data=None, content_type=None, timeout_s=60.0):
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"{base}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            if not body:
                return resp.status, {}
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw.decode("utf-8", "replace")
        raise ApiError(f"{method} {path} -> HTTP {exc.code}: {payload}") from exc


def poll_until_done(label):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        _, last = _request("GET", "/api/v1/status", timeout_s=15)
        if not isinstance(last, dict):
            raise ApiError(f"{label}: status not an object: {last!r}")
        st = last.get("status")
        if st == "completed":
            return last
        if st == "failed":
            raise ApiError(f"{label}: calculation failed: {last.get('last_error')}")
        if st == "stopped":
            raise ApiError(f"{label}: calculation stopped early: {last}")
        time.sleep(0.5)
    raise ApiError(f"{label}: timed out after {timeout}s  last={last!r}")


def extract_estimate(status):
    update = status.get("last_update") or {}
    result = status.get("last_result") or {}
    mean = result.get("mean", update.get("mean"))
    if mean is None and isinstance(result.get("probabilities"), list) and result["probabilities"]:
        mean = result["probabilities"][0].get("p_total")
    std_err = update.get("std_err")
    if std_err is None:
        std_err = result.get("std_err")
    trials = result.get("trials", update.get("trials") or 0)
    if mean is None or std_err is None:
        raise ApiError(f"missing mean/std_err in status: {json.dumps(status)[:800]}")
    return float(mean), float(std_err), int(trials or 0)


def main():
    status, home = _request("GET", "/", timeout_s=15)
    html = home.decode() if isinstance(home, (bytes, bytearray)) else str(home)
    if status != 200 or ("Playground" not in html and "<!doctype html>" not in html.lower()):
        raise ApiError(f"UI / unexpected {status} {html[:300]!r}")
    asset_m = re.search(r'src="(/assets/[^"]+)"', html)
    if asset_m is None:
        raise ApiError(f"UI / has no hashed /assets/ script: {html[:300]!r}")
    ast, asset_body = _request("GET", asset_m.group(1), timeout_s=15)
    asset_len = len(asset_body) if isinstance(asset_body, (bytes, bytearray, str)) else "?"
    if ast != 200:
        raise ApiError(f"UI asset {asset_m.group(1)} -> {ast}")
    print(f"ui / ok  asset={asset_m.group(1)}  bytes={asset_len}", flush=True)

    _, health = _request("GET", "/api/v1/healthz", timeout_s=15)
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise ApiError(f"healthz unexpected: {health!r}")
    print(f"healthz json ok  {health}", flush=True)

    _, catalog = _request("GET", "/api/v1/model/xml-inputs", timeout_s=15)
    ids = catalog.get("modelIds") if isinstance(catalog, dict) else []
    print(
        f"catalog  {len(ids) if isinstance(ids, list) else '?'} models  sample={ids[:8] if isinstance(ids, list) else catalog}",
        flush=True,
    )
    if not isinstance(ids, list) or not any(i.endswith("chinese") or i == "chinese" for i in ids):
        raise ApiError(f"chinese not in catalog: {catalog!r}")

    print("-- compile chinese", flush=True)
    st, payload = _request("GET", "/api/v1/model/dag?model_id=chinese", timeout_s=timeout)
    if st != 200 or not isinstance(payload, dict):
        raise ApiError(f"compile chinese: unexpected {st} {payload!r}")
    print(f"   dag ok  keys={sorted(payload.keys())[:8]}", flush=True)

    body = json.dumps(
        {
            "model_id": "chinese",
            "num_trials": n,
            "requested_trials_per_pass": tpp,
            "early_stop": False,
            "policy": "wald",
        }
    ).encode()
    print(f"-- start chinese  N={n:g}  tpp={tpp:g}", flush=True)
    st, started = _request("POST", "/api/v1/start", data=body, content_type="application/json", timeout_s=30)
    if st != 200 or not isinstance(started, dict) or not started.get("accepted"):
        raise ApiError(f"start rejected: {started!r}")
    done = poll_until_done("chinese")
    mean, std_err, trials = extract_estimate(done)
    if std_err <= 0.0:
        raise ApiError(f"chinese: std_err={std_err} (need a positive sigma)")
    delta = abs(mean - REF)
    bound = zmax * std_err
    z = (mean - REF) / std_err
    ok = delta <= bound
    line = (
        f"chinese          p={mean:.6e}  p_bdd={REF:.6e}  se={std_err:.6e}  "
        f"|d|={delta:.6e}  z*se={bound:.6e}  z={z:+.3f}  N={trials}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(line, flush=True)
    if not ok:
        raise ApiError(line)
    print("WHEEL SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
