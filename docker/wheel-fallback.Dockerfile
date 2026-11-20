# syntax=docker/dockerfile:1.10.0
# Install the generic wheel, drop _native_cuda_sm70.so beside it, and check
# auto-select falls back to generic when no sm_70 device is present.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libstdc++6 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY generic.whl /tmp/generic.whl
COPY sm70.whl /tmp/sm70.whl
# pip rejects filenames that are not PEP 427 tags; rename before install.
RUN mv /tmp/generic.whl /tmp/fuzzybits-0.18.3-cp312-cp312-manylinux_2_39_x86_64.whl && \
    pip install --no-cache-dir /tmp/fuzzybits-0.18.3-cp312-cp312-manylinux_2_39_x86_64.whl && \
    python3 - <<'PY'
import pathlib, sysconfig, zipfile
plat = pathlib.Path(sysconfig.get_path("platlib"))
dest = plat / "fuzzybits"
extracted = []
with zipfile.ZipFile("/tmp/sm70.whl") as z:
    for name in z.namelist():
        if "_native_cuda_sm70" in name and name.endswith(".so"):
            target = dest / pathlib.Path(name).name
            target.write_bytes(z.read(name))
            extracted.append(str(target))
if not extracted:
    raise SystemExit("cuda-sm70 native extension missing from sm70.whl")
print("extracted", extracted)
PY
