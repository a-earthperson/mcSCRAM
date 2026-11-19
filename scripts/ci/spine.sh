#!/usr/bin/env bash
# fzb compile → inspect --validate → run against Aralia BDD references.
#
# Every Monte Carlo invocation is wrapped in `timeout ${SPINE_TIMEOUT:-60}`
# and this script must run *inside* the toolchain container (a timeout around
# `docker run` leaves GPU-resident orphans).
#
# Env:
#   SPINE_BACKEND   omp|cuda   (default omp)
#   SPINE_TRIALS    count      (default 4194304 = 2^22; GPU jobs use 1e8)
#   SPINE_TIMEOUT   seconds    (default 60)
#   SPINE_Z_MAX     bound      (default 4)
#   SPINE_INPUT_DIR Aralia dir (default /src/input/Aralia)
#   PRESET          build dir  (default generic)
set -euo pipefail

BACKEND="${SPINE_BACKEND:-omp}"
TRIALS="${SPINE_TRIALS:-4194304}"
TIMEOUT_SEC="${SPINE_TIMEOUT:-60}"
Z_MAX="${SPINE_Z_MAX:-4}"
INPUT_DIR="${SPINE_INPUT_DIR:-/src/input/Aralia}"
PRESET="${PRESET:-generic}"

# BDD references: tests/mc/cli_correctness_tests.cc / issue #45.
declare -A REF=(
  [chinese]=1.17058e-3
  [das9201]=1.34237e-2
  [baobab2]=7.13018e-4
)
MODELS=(chinese das9201 baobab2)

fzb="$(find "build/${PRESET}" -name fzb -type f -print -quit || true)"
if [[ -z "${fzb}" || ! -x "${fzb}" ]]; then
  echo "error: fzb not found under build/${PRESET}" >&2
  exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "error: missing input dir ${INPUT_DIR}" >&2
  exit 2
fi

export ACPP_VISIBILITY_MASK="${BACKEND}"
export ACPP_DEBUG_LEVEL="${ACPP_DEBUG_LEVEL:-0}"
export ACPP_PERSISTENT_RUNTIME="${ACPP_PERSISTENT_RUNTIME:-1}"

echo "fzb-spine  backend=${BACKEND}  trials=${TRIALS}  timeout=${TIMEOUT_SEC}s  |z|≤${Z_MAX}  fzb=${fzb}"

check_json() {
  local model="$1" pref="$2" json_path="$3" wall="$4" zmax="$5"
  python3 - "$model" "$pref" "$json_path" "$wall" "$zmax" <<'PY'
import json, math, sys
model, pref_s, json_path, wall_s, zmax_s = sys.argv[1:6]
pref = float(pref_s)
zmax = float(zmax_s)
wall = float(wall_s)
payload = json.loads(open(json_path, encoding="utf-8").read())
queries = payload.get("queries") or []
if not queries:
    sys.stderr.write(f"error: {model}: no queries in fzb --json output\n")
    sys.exit(3)
row = queries[0]
phat = float(row["estimate"])
n = float(row["n"])
if n <= 0:
    sys.stderr.write(f"error: {model}: n={n} from fzb run\n")
    sys.exit(3)
se = math.sqrt(max(phat * (1.0 - phat), 0.0) / n)
z = (phat - pref) / se if se > 0 else (float("inf") if phat != pref else 0.0)
absz = abs(z)
status = "PASS" if absz <= zmax else "FAIL"
print(
    f"{model:10s}  p̂={phat:.6e}  p_ref={pref:.6e}  |z|={absz:.3f}  "
    f"n={int(n)}  wall={wall:.2f}s  {status}"
)
if absz > zmax:
    sys.exit(4)
PY
}

rc=0
workdir="$(mktemp -d /tmp/fzb-spine.XXXXXX)"
trap 'rm -rf "${workdir}"' EXIT

for model in "${MODELS[@]}"; do
  xml="${INPUT_DIR}/${model}.xml"
  if [[ ! -f "${xml}" ]]; then
    echo "error: missing ${xml}" >&2
    rc=2
    continue
  fi
  pcfb="${workdir}/${model}.pcfb"
  if ! "${fzb}" compile "${xml}" -c 2 --no-kn -o "${pcfb}"; then
    echo "error: ${model}: fzb compile failed" >&2
    rc=1
    continue
  fi
  if ! "${fzb}" inspect --validate "${pcfb}" >/dev/null; then
    echo "error: ${model}: fzb inspect --validate failed" >&2
    rc=1
    continue
  fi
  json_path="${workdir}/${model}.json"
  err_path="${workdir}/${model}.err"
  start_ns="$(date +%s%N)"
  set +e
  timeout "${TIMEOUT_SEC}" "${fzb}" run "${pcfb}" \
    --trials "${TRIALS}" --backend "${BACKEND}" --json \
    >"${json_path}" 2>"${err_path}"
  run_rc=$?
  set -e
  end_ns="$(date +%s%N)"
  wall="$(python3 -c "print('{:.3f}'.format((int('${end_ns}') - int('${start_ns}')) / 1e9))")"
  if [[ ${run_rc} -eq 124 ]]; then
    echo "${model}  TIMEOUT after ${TIMEOUT_SEC}s" >&2
    tail -n 20 "${err_path}" >&2 || true
    rc=124
    continue
  fi
  if [[ ${run_rc} -ne 0 ]]; then
    echo "error: ${model}: fzb run exit ${run_rc}" >&2
    cat "${err_path}" >&2 || true
    cat "${json_path}" >&2 || true
    rc=${run_rc}
    continue
  fi
  if ! check_json "${model}" "${REF[${model}]}" "${json_path}" "${wall}" "${Z_MAX}"; then
    cat "${err_path}" >&2 || true
    rc=4
  fi
done
exit "${rc}"
