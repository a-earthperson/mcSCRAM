#!/usr/bin/env python3
"""GPU-less Windows wheel smoke: import, devices, CUDA DLL present, chinese 4σ, pytest.

Intended for a clean venv that has only the wheel (+ pytest) installed.
Every Monte Carlo invocation is bounded by a 60 s timeout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

CHINESE_REF = 1.17058e-3
Z = 4.0
TRIALS = 1 << 20  # 1 048 576; hosted OpenMP stays well under 60 s
MC_TIMEOUT = 60


def _wheel_names(whl: Path) -> list[str]:
    with zipfile.ZipFile(whl) as zf:
        return zf.namelist()


def check_wheel_cuda_dll(whl: Path) -> None:
    names = _wheel_names(whl)
    cuda_dlls = [
        n
        for n in names
        if n.replace("\\", "/").lower().endswith("rt-backend-cuda.dll")
        or n.replace("\\", "/").endswith("librt-backend-cuda.so")
    ]
    driver = [n for n in names if Path(n).name.lower() in {"nvcuda.dll", "libcuda.so", "libcuda.so.1"}]
    print(f"wheel={whl.name} bytes={whl.stat().st_size} entries={len(names)}")
    print("cuda backend artefacts:")
    for n in cuda_dlls:
        print(f"  {n}")
    if driver:
        raise SystemExit(f"wheel contains NVIDIA driver DLL(s): {driver}")
    if not cuda_dlls:
        raise SystemExit("CUDA backend DLL missing from wheel (expected rt-backend-cuda.dll)")
    print("bundled DLLs (name only):")
    for n in sorted(names):
        lower = n.lower()
        if lower.endswith((".dll", ".exe", ".pyd", ".so", ".bc")):
            print(f"  {n}")


def smoke_import_and_devices() -> None:
    os.environ.setdefault("ACPP_DEBUG_LEVEL", "2")
    os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
    os.environ.setdefault("FUZZYBITS_BACKEND", "omp")
    import fuzzybits

    fuzzybits.init(backend="omp")
    print("version", fuzzybits.__version__)
    print("variant", fuzzybits.active_variant())
    print("backend", fuzzybits.active_backend())
    devices = fuzzybits.list_devices()
    print("devices", devices)
    if not devices:
        raise SystemExit("list_devices() returned empty; OpenMP host device should be present")
    backends = {str(d.get("backend", "")).lower() for d in devices}
    types = {str(d.get("type", "")).lower() for d in devices}
    if "omp" not in backends and "cpu" not in types and not any("omp" in str(d).lower() for d in devices):
        print("warning: host device listing did not mention omp; continuing", file=sys.stderr)
    print("import+list_devices ok")


def smoke_chinese(xml: Path) -> None:
    # Sibling interpreter so a hung kernel is killed by MC_TIMEOUT.
    script = r"""
import math, os, sys
os.environ.setdefault("ACPP_DEBUG_LEVEL", "2")
os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
os.environ["FUZZYBITS_BACKEND"] = "omp"
import fuzzybits
fuzzybits.init(backend="omp")
xml, trials, ref, z = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
blob = fuzzybits.compile([xml], pdag_c=2, no_kn=True)
rows = fuzzybits.run(blob, trials=trials, backend="omp")
row = rows[0]
p = float(row.estimate)
n = float(row.n) if row.n else float(trials)
se = math.sqrt(p * (1.0 - p) / n) if 0.0 < p < 1.0 else float("nan")
delta = abs(p - ref)
bound = z * se
print(
    f"chinese: estimate={p:.6e} n={int(n)} se={se:.6e} "
    f"ref={ref:.6e} |d|={delta:.6e} 4s={bound:.6e} ok={delta <= bound}"
)
sys.exit(0 if delta <= bound else 1)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(xml), str(TRIALS), str(CHINESE_REF), str(Z)],
        timeout=MC_TIMEOUT,
        check=True,
    )


def run_pytest(repo: Path) -> None:
    env = os.environ.copy()
    env["ACPP_VISIBILITY_MASK"] = "omp"
    env["FUZZYBITS_BACKEND"] = "omp"
    env["ACPP_DEBUG_LEVEL"] = env.get("ACPP_DEBUG_LEVEL", "0")
    subprocess.run(
        [sys.executable, "-m", "pytest", str(repo / "tests" / "python"), "-q", "-m", "not gpu", "-o", "pythonpath="],
        cwd=str(repo),
        env=env,
        timeout=300,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    whl = args.wheel.resolve()
    repo = args.repo.resolve()
    xml = repo / "input" / "Aralia" / "chinese.xml"
    if not xml.is_file():
        raise SystemExit(f"missing {xml}")
    check_wheel_cuda_dll(whl)
    smoke_import_and_devices()
    smoke_chinese(xml)
    run_pytest(repo)
    print("WINDOWS SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"timed out: {exc}") from exc
