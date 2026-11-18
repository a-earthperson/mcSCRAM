#!/usr/bin/env bash
# Smoke-test mcSCRAM Monte Carlo estimates against Aralia BDD references.
#
# Usage (inside a runtime/profiler image):
#   mcscram-smoke
#   SMOKE_N=1e8 SMOKE_TIMEOUT=60 mcscram-smoke [model ...]
#
# Exit non-zero if any run times out, scram-cli fails, or |z| > SMOKE_Z_MAX (default 4).
set -euo pipefail

INPUT_DIR="${SMOKE_INPUT_DIR:-/app/input}"
N="${SMOKE_N:-1e8}"
TIMEOUT_SEC="${SMOKE_TIMEOUT:-60}"
Z_MAX="${SMOKE_Z_MAX:-4}"
SCRAM="${SCRAM_CLI:-/usr/local/bin/scram-cli}"
if [[ ! -x "${SCRAM}" ]]; then
  SCRAM="$(command -v scram-cli)"
fi

# BDD references: benchmarks/input.csv (Aralia) and issue #24/#26.
declare -A REF=(
  [chinese]=1.17058e-3
  [baobab1]=1.01708e-4
  [das9201]=1.34237e-2
  [das9701]=0.07447
)

DEFAULT_MODELS=(chinese baobab1 das9201)
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ ! -x "${SCRAM}" ]]; then
  echo "error: scram-cli not found (SCRAM_CLI=${SCRAM_CLI:-})" >&2
  exit 2
fi

parse_and_check() {
  local model="$1" pref="$2" log="$3" n="$4" zmax="$5"
  python3 - "$model" "$pref" "$log" "$n" "$zmax" <<'PY'
import math, re, sys
model, pref_s, log_path, n_s, zmax_s = sys.argv[1:6]
pref = float(pref_s)
n = float(n_s)
zmax = float(zmax_s)
text = open(log_path, encoding="utf-8", errors="replace").read()
m = re.search(r"Root Event Probability:\s*([0-9.eE+-]+)", text)
if not m:
    m = re.search(r"\bmean[=:\s]+([0-9.eE+-]+)", text)
if not m:
    sys.stderr.write(f"error: {model}: could not parse estimate from scram-cli output\n")
    sys.exit(3)
phat = float(m.group(1))
zm = re.search(r"z=\s*([0-9.eE+-]+)", text)
se = math.sqrt(max(phat * (1.0 - phat), 0.0) / n) if n > 0 else float("nan")
if zm:
    z = float(zm.group(1))
elif se > 0:
    z = (phat - pref) / se
else:
    z = float("inf") if phat != pref else 0.0
absz = abs(z)
status = "PASS" if absz <= zmax else "FAIL"
print(f"{model:10s}  p̂={phat:.6e}  p_bdd={pref:.6e}  |z|={absz:.3f}  {status}")
if absz > zmax:
    sys.exit(4)
PY
}

rc=0
echo "mcscram-smoke  N=${N}  timeout=${TIMEOUT_SEC}s  |z|≤${Z_MAX}  scram=${SCRAM}"
for model in "${MODELS[@]}"; do
  if [[ -z "${REF[${model}]+x}" ]]; then
    echo "error: no BDD reference for model '${model}'" >&2
    rc=2
    continue
  fi
  xml="${INPUT_DIR}/${model}.xml"
  if [[ ! -f "${xml}" ]]; then
    echo "error: missing ${xml}" >&2
    rc=2
    continue
  fi
  log="$(mktemp)"
  set +e
  timeout "${TIMEOUT_SEC}" "${SCRAM}" \
    --monte-carlo --probability --pdag -c 2 --no-kn \
    -N "${N}" -P wald -p "${REF[${model}]}" \
    --no-report -V 3 \
    "${xml}" >"${log}" 2>&1
  run_rc=$?
  set -e
  if [[ ${run_rc} -eq 124 ]]; then
    echo "${model}  TIMEOUT after ${TIMEOUT_SEC}s" >&2
    tail -n 20 "${log}" >&2 || true
    rm -f "${log}"
    rc=124
    continue
  fi
  if [[ ${run_rc} -ne 0 ]]; then
    echo "error: ${model}: scram-cli exit ${run_rc}" >&2
    tail -n 40 "${log}" >&2 || true
    rm -f "${log}"
    rc=${run_rc}
    continue
  fi
  if ! parse_and_check "${model}" "${REF[${model}]}" "${log}" "${N}" "${Z_MAX}"; then
    tail -n 40 "${log}" >&2 || true
    rc=4
  fi
  rm -f "${log}"
done
exit "${rc}"
