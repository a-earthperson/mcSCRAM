#!/usr/bin/env bash
# Host-side helper for fuzzybits#44 acceptance runs on xerxes.
# Usage:
#   verify_fzb_run.sh <image> <backend> <gpu-arg|none> <trials> <model-xml> [lock-dir]
set -euo pipefail

IMAGE="${1:?image}"
BACKEND="${2:?cuda|omp}"
GPU_ARG="${3:?--gpus device=N or none}"
TRIALS="${4:?trial count}"
MODEL="${5:?xml path inside the container}"
LOCK="${6:-}"

run() {
  if [[ "${GPU_ARG}" == "none" ]]; then
    docker --context xerxes run --rm \
      -e ACPP_VISIBILITY_MASK="${BACKEND}" \
      "${IMAGE}" \
      bash -lc "$*"
  else
    docker --context xerxes run --rm ${GPU_ARG} \
      -e ACPP_VISIBILITY_MASK="${BACKEND}" \
      "${IMAGE}" \
      bash -lc "$*"
  fi
}

inner=$(cat <<EOF
set -euo pipefail
cd /app
export ACPP_VISIBILITY_MASK=${BACKEND}
FZB=/usr/local/bin/fzb
if [[ ! -x "\$FZB" ]]; then
  FZB=/app/build/targets/fzb/fzb
fi
STEM=\$(basename ${MODEL} .xml)
\$FZB compile ${MODEL} -c 2 --no-kn -o /tmp/\${STEM}.pcfb
TIMEFORMAT='WALL_SEC=%R'
{ time timeout 60 \$FZB run /tmp/\${STEM}.pcfb --trials ${TRIALS} --backend ${BACKEND} --json; } 2>&1
EOF
)

if [[ -n "${LOCK}" ]]; then
  until mkdir "${LOCK}" 2>/dev/null; do sleep 5; done
  set +e
  run "${inner}"
  rc=$?
  rmdir "${LOCK}"
  exit $rc
else
  run "${inner}"
fi
