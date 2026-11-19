#!/usr/bin/env python3
"""Rewrite DT_RUNPATH on a staged fuzzybits wheel prefix to $ORIGIN-relative paths.

Layout after CMake install + AdaptiveCpp --acpp-deploy:

    <prefix>/fuzzybits/_native_*.so
    <prefix>/fuzzybits/libscram.so
    <prefix>/fuzzybits/libmc.so
    <prefix>/fuzzybits/lib/libacpp-rt.so
    <prefix>/fuzzybits/lib/hipSYCL/...
    <prefix>/fuzzybits/lib/hipSYCL/ext/llvm/bin/{opt,llc,ld.lld}

AdaptiveCpp locates backend plugins as ``dirname(libacpp-rt)/hipSYCL`` and the
SSCP redistributable package as ``dirname(libacpp-common)/hipSYCL/ext``. Keep
that tree intact; only RPATH is rewritten. NVIDIA driver libs (libcuda.so*)
are deleted if the deploy step copied them.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

ORIGIN_RPATHS = {
    "ext": "$ORIGIN:$ORIGIN/lib",
    "lib": "$ORIGIN",
    "backend": "$ORIGIN:$ORIGIN/..:$ORIGIN/llvm-to-backend",
    "llvm_to_backend": "$ORIGIN:$ORIGIN/../..",
    "llvm_bin": "$ORIGIN/../../../..",
}

SKIP_NAMES = {
    "libcuda.so",
    "libcuda.so.1",
}


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def _patchelf(path: Path, rpath: str) -> None:
    result = _run(["patchelf", "--set-rpath", rpath, str(path)])
    if result.returncode != 0:
        sys.stderr.write(f"patchelf failed on {path}: {result.stderr}\n")
        raise SystemExit(result.returncode or 1)


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _chmod_u_write(path: Path) -> None:
    mode = path.stat().st_mode
    if not mode & stat.S_IWUSR:
        path.chmod(mode | stat.S_IWUSR)


def _classify(rel: Path) -> str | None:
    parts = rel.parts
    if len(parts) == 1 and parts[0].endswith((".so", ".so.1")) or (
        len(parts) == 1 and ".so" in parts[0]
    ):
        return "ext"
    if len(parts) >= 2 and parts[0] == "lib" and "hipSYCL" not in parts:
        return "lib"
    if "ext" in parts and "llvm" in parts and "bin" in parts:
        return "llvm_bin"
    if "llvm-to-backend" in parts:
        return "llvm_to_backend"
    if "hipSYCL" in parts:
        return "backend"
    if parts and parts[0] == "lib":
        return "lib"
    return "ext"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(f"usage: {argv[0]} <prefix>/fuzzybits\n")
        return 2
    root = Path(argv[1]).resolve()
    if not root.is_dir():
        sys.stderr.write(f"not a directory: {root}\n")
        return 1

    for driver in root.rglob("libcuda.so*"):
        print(f"removing driver library {driver}")
        driver.unlink()

    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in SKIP_NAMES or path.name.startswith("libcuda.so"):
            continue
        if not _is_elf(path):
            continue
        rel = path.relative_to(root)
        kind = _classify(rel)
        if kind is None:
            continue
        rpath = ORIGIN_RPATHS[kind]
        _chmod_u_write(path)
        print(f"RPATH {rpath} -> {rel}")
        _patchelf(path, rpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
