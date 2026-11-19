#!/usr/bin/env python3
"""Copy AdaptiveCpp's runtime/JIT tree into fuzzybits/lib for a self-contained wheel.

Mirrors `acpp --acpp-deploy=core,cuda:<dest>` but does not abort when an optional
file (ld.lld, libcudart.so symlink, …) is missing. NVIDIA driver libraries
(libcuda.so*) are never copied.

Layout (AdaptiveCpp v25.10 plugin discovery):

    <dest>/libacpp-rt.so
    <dest>/libacpp-common.so
    <dest>/hipSYCL/librt-backend-{cuda,omp}.so
    <dest>/hipSYCL/llvm-to-backend/libllvm-to-{backend,host,ptx}.so
    <dest>/hipSYCL/bitcode/*.bc
    <dest>/hipSYCL/ext/llvm/bin/{opt,llc,ld.lld}
    <dest>/hipSYCL/ext/bitcode/ptx/libdevice.10.bc
    <dest>/libLLVM.so*
    <dest>/libomp.so*
    <dest>/libcudart.so*
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _copy_file(src: Path, dest_dir: Path) -> Path | None:
    if not src.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.is_symlink():
        # Copy the real file under the symlink name so RPATH lookup of the
        # SONAME (libcudart.so.12, libLLVM.so.20.1, …) still works.
        real = src.resolve()
        if real.is_file():
            shutil.copy2(real, dest)
            return dest
    shutil.copy2(src, dest)
    return dest


def _copy_glob(directory: Path, pattern: str, dest_dir: Path) -> list[Path]:
    copied: list[Path] = []
    if not directory.is_dir():
        return copied
    for src in directory.glob(pattern):
        if src.is_file() or src.is_symlink():
            out = _copy_file(src, dest_dir)
            if out is not None:
                copied.append(out)
    return copied


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def bundle(acpp_prefix: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lib = acpp_prefix / "lib"
    hipsycl = lib / "hipSYCL"

    copied: list[str] = []

    def record(src: Path, dest_dir: Path) -> None:
        out = _copy_file(src, dest_dir)
        if out is None:
            print(f"skip missing {src}")
        else:
            print(f"copy {src} -> {out}")
            copied.append(str(out))

    for name in ("libacpp-rt.so", "libacpp-common.so"):
        record(lib / name, dest)

    backends = dest / "hipSYCL"
    for name in ("librt-backend-cuda.so", "librt-backend-omp.so"):
        record(hipsycl / name, backends)

    l2b = dest / "hipSYCL" / "llvm-to-backend"
    for name in ("libllvm-to-backend.so", "libllvm-to-host.so", "libllvm-to-ptx.so"):
        record(hipsycl / "llvm-to-backend" / name, l2b)

    bitcode_src = hipsycl / "bitcode"
    bitcode_dst = dest / "hipSYCL" / "bitcode"
    for src in sorted(bitcode_src.glob("*.bc")) if bitcode_src.is_dir() else []:
        record(src, bitcode_dst)

    llvm_bin_dst = dest / "hipSYCL" / "ext" / "llvm" / "bin"
    clang_ver = os.environ.get("CLANG_VERSION", "20")
    llvm_bin = Path(f"/usr/lib/llvm-{clang_ver}/bin")
    for tool in ("opt", "llc", "ld.lld"):
        record(_first_existing([llvm_bin / tool, Path("/usr/bin") / tool]) or llvm_bin / tool, llvm_bin_dst)

    libdevice = _first_existing(
        [
            Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"),
            Path("/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc"),
        ]
    )
    if libdevice is not None:
        record(libdevice, dest / "hipSYCL" / "ext" / "bitcode" / "ptx")
    else:
        print("skip missing libdevice.10.bc")

    llvm_lib = Path(f"/usr/lib/llvm-{clang_ver}/lib")
    # Only the pinned LLVM SONAME. Globbing libLLVM.so* also picks up leftover
    # distro LLVM 18 in /usr/lib/x86_64-linux-gnu and triplicate unversioned
    # symlinks that inflate the wheel by hundreds of MB.
    record(llvm_lib / f"libLLVM.so.{clang_ver}.1", dest)
    omp = _first_existing(
        [
            llvm_lib / "libomp.so.5",
            llvm_lib / "libomp.so",
            Path("/usr/lib/x86_64-linux-gnu/libomp.so.5"),
        ]
    )
    if omp is not None:
        record(omp, dest)
    else:
        print("skip missing libomp")

    atomic = _first_existing(
        [
            Path("/usr/lib/x86_64-linux-gnu/libatomic.so.1"),
            Path("/usr/lib/gcc/x86_64-linux-gnu/13/libatomic.so.1"),
            Path("/usr/lib/gcc/x86_64-linux-gnu/14/libatomic.so.1"),
        ]
    )
    if atomic is not None:
        record(atomic, dest)
    else:
        print("skip missing libatomic.so.1")

    cuda_lib = Path("/usr/local/cuda/lib64")
    for pattern in ("libcudart.so*",):
        hits = _copy_glob(cuda_lib, pattern, dest)
        if not hits:
            print(f"skip missing {pattern}")
        else:
            for hit in hits:
                print(f"copy {hit}")

    for driver in dest.glob("libcuda.so*"):
        print(f"removing driver library {driver}")
        driver.unlink()

    if not (dest / "libacpp-rt.so").is_file():
        raise SystemExit(f"bundle failed: {dest}/libacpp-rt.so missing")
    print(f"bundled {len(copied)} AdaptiveCpp artefacts into {dest}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(f"usage: {argv[0]} <acpp-prefix> <dest-lib-dir>\n")
        return 2
    bundle(Path(argv[1]).resolve(), Path(argv[2]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
