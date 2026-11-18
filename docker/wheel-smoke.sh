#!/usr/bin/env bash
# Run inside a clean ubuntu:24.04 / python:3.12-slim container after pip install.
# Usage: wheel-smoke.sh [--backend cuda|omp] [--xml DIR] [--trials N]
set -euo pipefail

backend="${FUZZYBITS_BACKEND:-cuda}"
xml_dir="${FUZZYBITS_XML_DIR:-/models}"
trials="${FUZZYBITS_TRIALS:-10000000}"
seed="${FUZZYBITS_SEED:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) backend="$2"; shift 2 ;;
    --xml) xml_dir="$2"; shift 2 ;;
    --trials) trials="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

export FUZZYBITS_BACKEND="${backend}"

python3 - <<'PY'
import os
os.environ["FUZZYBITS_BACKEND"] = os.environ.get("FUZZYBITS_BACKEND", "cuda")
import fuzzybits
fuzzybits.init(backend=os.environ["FUZZYBITS_BACKEND"])
print("version", fuzzybits.__version__)
print("variant", fuzzybits.active_variant())
print("backend", fuzzybits.active_backend())
print("devices", fuzzybits.list_devices())
import mcscram_native
print("mcscram_native", hasattr(mcscram_native, "run_probability"))
PY

Z=1.959963984540054
failed=0
for name_ref in "das9201:1.34237e-2" "chinese:1.17058e-3"; do
  name="${name_ref%%:*}"
  ref="${name_ref##*:}"
  path="${xml_dir}/${name}.xml"
  if [[ ! -f "${path}" ]]; then
    echo "SKIP ${name}: missing ${path}"
    continue
  fi
  if ! timeout 60 python3 - "${path}" "${name}" "${ref}" "${trials}" "${seed}" "${Z}" <<'PY'
import os, sys
os.environ.setdefault("FUZZYBITS_BACKEND", os.environ.get("FUZZYBITS_BACKEND", "cuda"))
import fuzzybits
path, name, ref_s, trials_s, seed_s, z_s = sys.argv[1:]
ref = float(ref_s)
result = fuzzybits.probability(
    path, trials=float(trials_s), seed=int(seed_s), pdag_c=2, no_kn=True, policy="wald",
)
delta = abs(result.estimate - ref)
bound = float(z_s) * result.std_err
ok = delta <= bound
print(
    f"{name}: estimate={result.estimate:.6e} ci={result.ci} "
    f"se={result.std_err:.6e} ref={ref:.6e} |Δ|={delta:.6e} z·σ̂={bound:.6e} ok={ok} "
    f"variant={fuzzybits.active_variant()} backend={fuzzybits.active_backend()}"
)
sys.exit(0 if ok else 1)
PY
  then
    echo "FAIL ${name}"
    failed=$((failed + 1))
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
echo "SMOKE OK"
