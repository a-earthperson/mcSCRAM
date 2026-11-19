#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
whl="${1:?path to fuzzybits-*.whl}"
ctx="${root}/docker/wheel-test-context"
rm -rf "${ctx}"
mkdir -p "${ctx}/models" "${ctx}/tests"
cp "${whl}" "${ctx}/$(basename "${whl}")"
echo "$(basename "${whl}")" > "${ctx}/WHEEL_NAME"
cp "${root}/docker/inputs/das9201.xml" "${root}/docker/inputs/chinese.xml" "${ctx}/models/"
cp -R "${root}/tests/python" "${ctx}/tests/python"
cp "${root}/docker/wheel-smoke.sh" "${ctx}/wheel-smoke.sh"
cat > "${ctx}/tests/pytest.ini" <<'EOF'
[pytest]
testpaths = python
addopts = -q
markers =
    slow: longer Monte Carlo runs
    gpu: requires a CUDA device
EOF
echo "${ctx}"
