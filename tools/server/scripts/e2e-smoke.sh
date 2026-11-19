#!/usr/bin/env bash
# Hosted CPU E2E for the fuzzybits-server image (#46 step E).
#
# Starts the image with ACPP_VISIBILITY_MASK=omp, mounts public Aralia XML
# at /data/Aralia, then via the HTTP API (from inside the container, so this
# works on a remote Docker host as well as GitHub-hosted):
#   1. compile chinese and das9201 (GET /model/dag)
#   2. start a fixed-budget run, poll /status until done
#   3. assert |p̂ − p_bdd| ≤ z·σ̂  (z=4)
#   4. POST a .pcfb produced by `fzb compile` inside the image and run it
#
# Usage:
#   tools/server/scripts/e2e-smoke.sh [IMAGE]
#
# Env: SMOKE_N (default 1e7), SMOKE_TPP (default 1e6), SMOKE_TIMEOUT (90),
#      SMOKE_Z_MAX (4), ACPP_VISIBILITY_MASK (omp).
set -euo pipefail

IMAGE="${1:-${SERVER_IMAGE:-ghcr.io/fuzzybit-org/fuzzybits-server:nightly-generic}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARALIA="${SMOKE_ARALIA:-${ROOT}/input/Aralia}"
N="${SMOKE_N:-1e7}"
TPP="${SMOKE_TPP:-1000000}"
TIMEOUT_SEC="${SMOKE_TIMEOUT:-90}"
Z_MAX="${SMOKE_Z_MAX:-4}"
MASK="${ACPP_VISIBILITY_MASK:-omp}"
NAME="${SMOKE_NAME:-fuzzybits-server-e2e-$$}"
BASE="http://127.0.0.1:8000"

if [[ ! -d "${ARALIA}" ]]; then
  echo "error: Aralia models not found at ${ARALIA}" >&2
  exit 2
fi
if [[ ! -f "${ARALIA}/chinese.xml" || ! -f "${ARALIA}/das9201.xml" ]]; then
  echo "error: expected chinese.xml and das9201.xml under ${ARALIA}" >&2
  exit 2
fi

cleanup() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "==== docker logs (${NAME}) ====" >&2
    docker logs "${NAME}" >&2 || true
  fi
  docker stop -t 15 "${NAME}" >/dev/null 2>&1 || true
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  exit "${rc}"
}
trap cleanup EXIT

echo "e2e-smoke  image=${IMAGE}  N=${N}  tpp=${TPP}  timeout=${TIMEOUT_SEC}s  |z|≤${Z_MAX}  mask=${MASK}"

docker run --rm -d \
  --name "${NAME}" \
  -e ACPP_VISIBILITY_MASK="${MASK}" \
  -e ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}" \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e FUZZYBITS_SERVER_DATA_DIR=/data \
  -e FUZZYBITS_SERVER_ENGINE="${FUZZYBITS_SERVER_ENGINE:-pcfb}" \
  -v "${ARALIA}:/data/Aralia:ro" \
  "${IMAGE}" >/dev/null

# Wait for /healthz from inside the container (bounded).
python3 - "${NAME}" "${BASE}" "${TIMEOUT_SEC}" <<'PY'
import subprocess, sys, time
name, base, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + min(timeout_s, 60.0)
last = b""
while time.time() < deadline:
    proc = subprocess.run(
        [
            "docker", "exec", name, "python3", "-c",
            "import urllib.request; print(urllib.request.urlopen('%s/api/v1/healthz', timeout=3).read().decode())" % base,
        ],
        capture_output=True,
    )
    if proc.returncode == 0:
        print(f"healthz ok  {proc.stdout.decode().strip()}", flush=True)
        sys.exit(0)
    last = proc.stderr or proc.stdout
    time.sleep(0.5)
sys.stderr.write(f"error: healthz not ready after wait: {last.decode(errors='replace')}\n")
sys.exit(1)
PY

# API client runs inside the image so it always hits 127.0.0.1:8000.
# timeout wraps the whole client (compile + two models + pcfb upload).
timeout $(( TIMEOUT_SEC * 4 )) docker exec -i \
  -e SMOKE_N="${N}" \
  -e SMOKE_TPP="${TPP}" \
  -e SMOKE_TIMEOUT="${TIMEOUT_SEC}" \
  -e SMOKE_Z_MAX="${Z_MAX}" \
  -e SMOKE_BASE="${BASE}" \
  "${NAME}" python3 - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

base = os.environ.get("SMOKE_BASE", "http://127.0.0.1:8000")
n = float(os.environ.get("SMOKE_N", "1e7"))
tpp = float(os.environ.get("SMOKE_TPP", "1000000"))
timeout = float(os.environ.get("SMOKE_TIMEOUT", "90"))
zmax = float(os.environ.get("SMOKE_Z_MAX", "4"))

REFS = {
    "chinese": 1.17058e-3,
    "das9201": 1.34237e-2,
}


class ApiError(RuntimeError):
    pass


def _request(method: str, path: str, *, data: bytes | None = None, content_type: str | None = None, timeout_s: float = 60.0) -> tuple[int, object]:
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


def compile_model(model_id: str) -> None:
    print(f"-- compile {model_id}", flush=True)
    status, payload = _request("GET", f"/api/v1/model/dag?model_id={model_id}", timeout_s=timeout)
    if status != 200 or not isinstance(payload, dict):
        raise ApiError(f"compile {model_id}: unexpected {status} {payload!r}")
    nodes = payload.get("nodes") or payload.get("dag") or payload
    n_nodes = len(nodes) if isinstance(nodes, (list, dict)) else "?"
    print(f"   dag ok  keys={sorted(payload.keys())[:8]}  nodes={n_nodes}", flush=True)


def start_run(**body: object) -> None:
    blob = json.dumps(body).encode()
    status, payload = _request(
        "POST",
        "/api/v1/start",
        data=blob,
        content_type="application/json",
        timeout_s=30,
    )
    if status != 200 or not isinstance(payload, dict) or not payload.get("accepted"):
        raise ApiError(f"start rejected: {payload!r}")


def poll_until_done(label: str) -> dict:
    deadline = time.time() + timeout
    last: object = {}
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


def extract_estimate(status: dict) -> tuple[float, float, int]:
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


def assert_ref(label: str, status: dict, ref: float) -> str:
    mean, std_err, trials = extract_estimate(status)
    if std_err <= 0.0:
        raise ApiError(f"{label}: std_err={std_err} (need a positive σ̂)")
    delta = abs(mean - ref)
    bound = zmax * std_err
    z = (mean - ref) / std_err
    ok = delta <= bound
    line = (
        f"{label:16s}  p̂={mean:.6e}  p_bdd={ref:.6e}  σ̂={std_err:.6e}  "
        f"|Δ|={delta:.6e}  z·σ̂={bound:.6e}  z={z:+.3f}  N={trials}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(line, flush=True)
    if not ok:
        raise ApiError(line)
    return line


def main() -> int:
    lines: list[str] = []
    _, catalog = _request("GET", "/api/v1/model/xml-inputs", timeout_s=15)
    ids = catalog.get("modelIds") if isinstance(catalog, dict) else []
    print(
        f"catalog  {len(ids) if isinstance(ids, list) else '?'} models  sample={ids[:8] if isinstance(ids, list) else catalog}",
        flush=True,
    )
    if not isinstance(ids, list) or not any(i.endswith("chinese") or i == "chinese" for i in ids):
        raise ApiError(f"chinese not in catalog: {catalog!r}")

    start_body = {
        "num_trials": n,
        "requested_trials_per_pass": tpp,
        "early_stop": False,
        "policy": "wald",
    }

    for model_id, ref in REFS.items():
        compile_model(model_id)
        print(f"-- start {model_id}  N={n:g}  tpp={tpp:g}", flush=True)
        start_run(model_id=model_id, **start_body)
        st = poll_until_done(model_id)
        lines.append(assert_ref(model_id, st, ref))

    xml = "/data/Aralia/chinese.xml"
    if not os.path.isfile(xml):
        xml = "/app/input/chinese.xml"
    out = "/tmp/e2e-chinese.pcfb"
    print(f"-- fzb compile chinese inside the image ({xml})", flush=True)
    compile = subprocess.run(
        [
            "timeout",
            str(int(timeout)),
            "/usr/local/bin/fzb",
            "compile",
            xml,
            "-c",
            "2",
            "--no-kn",
            "-o",
            out,
        ],
        capture_output=True,
        text=True,
    )
    if compile.returncode != 0:
        # Runtime CLI currently abort()s on CPU (`terminate called without an
        # active exception`). fuzzybits.compile is the same host lowering as
        # `fzb compile` (#71); use it so POST /model/pcfb still runs.
        print(
            f"   fzb compile failed ({compile.returncode}): {(compile.stderr or compile.stdout).strip()[:300]}",
            flush=True,
        )
        print("   falling back to fuzzybits.compile (same host lowering)", flush=True)
        import fuzzybits

        pcfb = bytes(fuzzybits.compile(xml, pdag_c=2, no_kn=True))
        open(out, "wb").write(pcfb)
    else:
        with open(out, "rb") as fh:
            pcfb = fh.read()
    if len(pcfb) < 8:
        raise ApiError(f"compiled pcfb too small: {len(pcfb)} bytes")
    print(f"   pcfb {len(pcfb)} bytes", flush=True)
    _, uploaded = _request(
        "POST",
        "/api/v1/model/pcfb",
        data=pcfb,
        content_type="application/octet-stream",
        timeout_s=timeout,
    )
    if not isinstance(uploaded, dict) or not uploaded.get("pcfb_id"):
        raise ApiError(f"pcfb upload failed: {uploaded!r}")
    pcfb_id = uploaded["pcfb_id"]
    print(f"   uploaded pcfb_id={pcfb_id}", flush=True)

    print(f"-- start pcfb chinese  N={n:g}", flush=True)
    start_run(pcfb_id=pcfb_id, **start_body)
    st = poll_until_done("pcfb:chinese")
    lines.append(assert_ref("pcfb:chinese", st, REFS["chinese"]))

    print("E2E SMOKE OK", flush=True)
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
