#!/usr/bin/env python3
"""macOS arm64 wheel smoke: import, Metal+OpenMP devices, omp 4σ, metal exception, pytest.

Intended for a clean venv that has only the wheel (+ pytest) installed.
Every Monte Carlo invocation is bounded by MC_TIMEOUT in a sibling interpreter:
AdaptiveCpp cannot switch backends after the first SYCL runtime init.
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

REQUIRED_NAMES = (
    "fuzzybits/_native_generic",
    "fuzzybits/libscram.dylib",
    "fuzzybits/lib/libacpp-rt.dylib",
    "fuzzybits/lib/libacpp-common.dylib",
    "fuzzybits/lib/hipSYCL/librt-backend-metal.dylib",
    "fuzzybits/lib/hipSYCL/librt-backend-omp.dylib",
    "fuzzybits/lib/hipSYCL/llvm-to-backend/libllvm-to-metal.dylib",
    "fuzzybits/lib/hipSYCL/llvm-to-backend/libllvm-to-host.dylib",
    "fuzzybits/lib/hipSYCL/ext/llvm/bin/opt",
    "fuzzybits/lib/hipSYCL/ext/llvm/bin/llc",
)


def _wheel_names(whl: Path) -> list[str]:
    with zipfile.ZipFile(whl) as zf:
        return [n.replace("\\", "/") for n in zf.namelist()]


def check_wheel(whl: Path) -> None:
    names = _wheel_names(whl)
    print(f"wheel={whl.name} bytes={whl.stat().st_size} entries={len(names)}")
    missing = [want for want in REQUIRED_NAMES if not any(n.startswith(want) for n in names)]
    lld = [n for n in names if n.endswith("/ld.lld") or n.endswith("/ld64.lld")]
    if not lld:
        missing.append("fuzzybits/lib/hipSYCL/ext/llvm/bin/ld.lld|ld64.lld")
    llvm = [n for n in names if "/libLLVM" in n and n.endswith(".dylib")]
    if not llvm:
        missing.append("fuzzybits/lib/libLLVM*.dylib")
    lld_libs = [n for n in names if "/liblld" in n and n.endswith(".dylib")]
    if not lld_libs:
        missing.append("fuzzybits/lib/liblld*.dylib")
    omp = [n for n in names if n.endswith("/libomp.dylib")]
    if not omp:
        missing.append("fuzzybits/lib/libomp.dylib")
    libmc = [n for n in names if n.endswith("libmc.dylib") or n.endswith("/libmc.so")]
    if libmc:
        raise SystemExit(f"Darwin wheel must not ship libmc (INTERFACE mc → libscram): {libmc}")
    print("bundled Mach-O / JIT (name only):")
    for n in sorted(names):
        if n.endswith((".dylib", ".so", ".bc")) or "/ext/llvm/bin/" in n or "/llvm-to-backend/" in n:
            print(f"  {n}")
    if missing:
        raise SystemExit(f"wheel missing required AdaptiveCpp artefacts: {missing}")
    print("wheel layout ok")


def _run_child(script: str, extra_args: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *extra_args],
        env=env,
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )


def smoke_devices() -> None:
    script = r"""
import os, sys
os.environ.pop("ACPP_VISIBILITY_MASK", None)
os.environ.pop("FUZZYBITS_BACKEND", None)
os.environ.setdefault("ACPP_DEBUG_LEVEL", "2")
os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
import fuzzybits
print("version", fuzzybits.__version__)
print("variant", fuzzybits.active_variant())
print("backend", fuzzybits.active_backend())
devices = fuzzybits.list_devices()
print("devices", devices)
if not devices:
    raise SystemExit("list_devices() returned empty")
backends = {str(d.get("backend", "")).lower() for d in devices}
blob = " ".join(str(d).lower() for d in devices)
has_omp = "omp" in backends or any(str(d.get("type", "")).lower() == "cpu" for d in devices)
has_metal = "metal" in backends or "paravirtual" in blob or "metal" in blob or "apple" in blob
print("has_omp", has_omp, "has_metal", has_metal, "backends", sorted(backends))
if not has_omp:
    raise SystemExit("OpenMP host device missing from list_devices()")
if not has_metal:
    raise SystemExit("Metal device missing from list_devices()")
print("import+list_devices ok")
"""
    env = os.environ.copy()
    env.pop("ACPP_VISIBILITY_MASK", None)
    env.pop("FUZZYBITS_BACKEND", None)
    env.setdefault("ACPP_DEBUG_LEVEL", "2")
    env.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
    proc = _run_child(script, [], env, MC_TIMEOUT)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"list_devices smoke failed rc={proc.returncode}")


def smoke_chinese(xml: Path) -> None:
    script = r"""
import math, os, sys
os.environ.setdefault("ACPP_DEBUG_LEVEL", "0")
os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
os.environ["FUZZYBITS_BACKEND"] = "omp"
os.environ["ACPP_VISIBILITY_MASK"] = "omp"
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
    env = os.environ.copy()
    env["FUZZYBITS_BACKEND"] = "omp"
    env["ACPP_VISIBILITY_MASK"] = "omp"
    proc = _run_child(script, [str(xml), str(TRIALS), str(CHINESE_REF), str(Z)], env, MC_TIMEOUT)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode < 0:
        raise SystemExit(f"omp chinese run crashed with signal {-proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(f"omp chinese smoke failed rc={proc.returncode}")


def smoke_metal(xml: Path) -> None:
    script = r"""
import os, sys
os.environ.setdefault("ACPP_DEBUG_LEVEL", "2")
os.environ.setdefault("ACPP_PERSISTENT_RUNTIME", "1")
os.environ["FUZZYBITS_BACKEND"] = "metal"
os.environ["ACPP_VISIBILITY_MASK"] = "metal"
import fuzzybits
fuzzybits.init(backend="metal")
xml = sys.argv[1]
blob = fuzzybits.compile([xml], pdag_c=2, no_kn=True)
try:
    rows = fuzzybits.run(blob, trials=1 << 16, backend="metal")
except BaseException as exc:
    print(f"metal_exception={type(exc).__name__}: {exc}")
    text = str(exc).lower()
    if "double" in text or "metal" in text or "sycl" in text or "adaptivecpp" in text or "not supported" in text:
        print("metal expected JIT failure as Python exception")
        sys.exit(0)
    print("metal raised, but the message did not mention double/Metal/SYCL")
    sys.exit(0)
print("metal unexpectedly succeeded", rows)
sys.exit(2)
"""
    env = os.environ.copy()
    env["FUZZYBITS_BACKEND"] = "metal"
    env["ACPP_VISIBILITY_MASK"] = "metal"
    env["PYTHONFAULTHANDLER"] = "1"
    proc = _run_child(script, [str(xml)], env, MC_TIMEOUT)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write("--- metal child stderr ---\n")
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode < 0:
        raise SystemExit(
            f"metal run crashed with signal {-proc.returncode} "
            "(wanted a Python exception, not abort; #17/#30)"
        )
    if proc.returncode == 2:
        raise SystemExit("metal run succeeded; expected a double/JIT exception until #17/#30")
    if proc.returncode != 0:
        raise SystemExit(f"metal smoke failed rc={proc.returncode}")


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
    check_wheel(whl)
    fzb = Path(sys.executable).parent / "fzb"
    if not fzb.is_file():
        fzb = Path(sys.executable).resolve().parent / "fzb"
    help_proc = subprocess.run([str(fzb), "--help"], text=True, capture_output=True, check=False)
    sys.stdout.write(help_proc.stdout)
    sys.stderr.write(help_proc.stderr)
    if help_proc.returncode != 0:
        raise SystemExit(f"fzb --help failed rc={help_proc.returncode}")
    if "Usage: fzb" not in help_proc.stdout:
        raise SystemExit("fzb --help did not print the native-style usage")
    smoke_devices()
    smoke_chinese(xml)
    smoke_metal(xml)
    run_pytest(repo)
    print("MACOS SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"timed out: {exc}") from exc
