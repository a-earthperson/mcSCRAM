#!/usr/bin/env bash
# Chown leftover root-owned bind mounts from previous docker --user 0 runs.
# Uses a public image only (no GHCR token). Safe before or after checkout.
#
# Usage: reclaim-workspace.sh [chown|trim]
#   chown — chown the workspace to the runner user (default)
#   trim  — chown, then delete build/ dist/ .cache/ (keep .ccache for actions/cache)
set -euo pipefail
mode="${1:-chown}"
ws="${GITHUB_WORKSPACE:-$PWD}"
mkdir -p "${ws}"
uid="$(id -u)"
gid="$(id -g)"
docker run --rm --user 0 -v "${ws}:/src" alpine:3.21 \
  sh -c "chown -R ${uid}:${gid} /src"
if [[ "${mode}" == trim ]]; then
  rm -rf "${ws}/build" "${ws}/dist" "${ws}/.cache" || true
fi
