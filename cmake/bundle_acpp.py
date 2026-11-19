#!/usr/bin/env python3
"""Copy AdaptiveCpp's runtime/JIT tree into fuzzybits/lib for a self-contained wheel.

Mirrors `acpp --acpp-deploy=core,cuda:<dest>` but does not abort when an optional
file (ld.lld, libcudart.so symlink, …) is missing. NVIDIA driver libraries
(libcuda.so*) are never copied.

Layout (AdaptiveCpp v25.10 plugin discovery):

    Linux:
      <dest>/libacpp-rt.so
      <dest>/hipSYCL/librt-backend-{cuda,omp}.so
      <dest>/hipSYCL/llvm-to-backend/libllvm-to-{backend,host,ptx}.so
      <dest>/hipSYCL/ext/llvm/bin/{opt,llc,ld.lld}

    Darwin (Metal + OpenMP SSCP):
      <dest>/libacpp-rt.dylib
      <dest>/hipSYCL/librt-backend-{metal,omp}.dylib
      <dest>/hipSYCL/llvm-to-backend/libllvm-to-{backend,host,metal}.dylib
      <dest>/hipSYCL/ext/llvm/bin/{opt,llc,ld.lld,ld64.lld}
      <dest>/libLLVM.dylib
      <dest>/libomp.dylib
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


def _copy_tree(src_dir: Path, dest_dir: Path, copied: list[str]) -> None:
    if not src_dir.is_dir():
        print(f"skip missing tree {src_dir}")
        return
    for src in src_dir.rglob("*"):
        if not src.is_file() and not src.is_symlink():
            continue
        rel = src.relative_to(src_dir)
        out = _copy_file(src, dest_dir / rel.parent)
        if out is None:
            print(f"skip missing {src}")
        else:
            print(f"copy {src} -> {out}")
            copied.append(str(out))


def _homebrew_prefix(formula: str) -> Path | None:
    import shutil
    import subprocess

    brew = shutil.which("brew")
    if brew is None:
        return None
    result = subprocess.run(
        [brew, "--prefix", formula],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    prefix = Path(result.stdout.strip())
    return prefix if prefix.is_dir() else None


def _bundle_darwin(acpp_prefix: Path, dest: Path) -> None:
    """Darwin AdaptiveCpp (Metal + OpenMP, Homebrew llvm@20)."""
    lib = acpp_prefix / "lib"
    copied: list[str] = []

    def record(src: Path, dest_dir: Path) -> None:
        out = _copy_file(src, dest_dir)
        if out is None:
            print(f"skip missing {src}")
        else:
            print(f"copy {src} -> {out}")
            copied.append(str(out))

    for pattern in ("libacpp-rt*.dylib", "libacpp-common*.dylib"):
        hits = _copy_glob(lib, pattern, dest)
        if not hits:
            print(f"skip missing {pattern}")
        else:
            for hit in hits:
                copied.append(str(hit))
                print(f"copy {hit}")

    hipsycl_src = lib / "hipSYCL"
    if hipsycl_src.is_dir():
        _copy_tree(hipsycl_src, dest / "hipSYCL", copied)
    else:
        print(f"skip missing tree {hipsycl_src}")
    # Executables install to prefix/bin/hipSYCL on Darwin.
    hipsycl_bin = acpp_prefix / "bin" / "hipSYCL"
    if hipsycl_bin.is_dir():
        _copy_tree(hipsycl_bin, dest / "hipSYCL", copied)

    llvm_prefix = Path(os.environ.get("LLVM_PREFIX") or "")
    if not llvm_prefix.is_dir():
        llvm_prefix = _homebrew_prefix("llvm@20") or Path("/opt/homebrew/opt/llvm@20")

    llvm_bin_dst = dest / "hipSYCL" / "ext" / "llvm" / "bin"
    llvm_bin = llvm_prefix / "bin"
    for tool in ("opt", "llc"):
        record(_first_existing([llvm_bin / tool]) or (llvm_bin / tool), llvm_bin_dst)

    lld_candidates = [
        llvm_bin / "ld64.lld",
        llvm_bin / "ld.lld",
    ]
    if os.environ.get("ACPP_LLD_PATH"):
        lld_candidates.insert(0, Path(os.environ["ACPP_LLD_PATH"]))
    lld = _first_existing(lld_candidates)
    brew_lld = _homebrew_prefix("lld@20") or _homebrew_prefix("lld")
    if lld is None and brew_lld is not None:
        lld = _first_existing([brew_lld / "bin" / "ld64.lld", brew_lld / "bin" / "ld.lld"])
    if lld is not None:
        record(lld, llvm_bin_dst)
        alias = llvm_bin_dst / ("ld.lld" if lld.name != "ld.lld" else "ld64.lld")
        if not alias.exists() and (llvm_bin_dst / lld.name).exists():
            src_copied = llvm_bin_dst / lld.name
            print(f"copy {src_copied} -> {alias}")
            shutil.copy2(src_copied, alias)
            copied.append(str(alias))
        lld_lib = lld.resolve().parent.parent / "lib"
        if not lld_lib.is_dir() and brew_lld is not None:
            lld_lib = brew_lld / "lib"
        if lld_lib.is_dir():
            lld_libs = _copy_glob(lld_lib, "liblld*.dylib", dest)
            if not lld_libs:
                print(f"skip missing {lld_lib}/liblld*.dylib")
            else:
                for hit in lld_libs:
                    copied.append(str(hit))
                    print(f"copy {hit}")
    else:
        print("skip missing ld.lld / ld64.lld")

    llvm_lib = llvm_prefix / "lib"
    llvm_files = [llvm_lib / "libLLVM.dylib", *sorted(llvm_lib.glob("libLLVM.[0-9]*.dylib"))]
    llvm_copied = False
    for src in llvm_files:
        if src.is_file() or src.is_symlink():
            record(src, dest)
            llvm_copied = True
    if not llvm_copied:
        print("skip missing libLLVM.dylib")

    omp_prefix = Path(os.environ.get("OMP_PREFIX") or "")
    if not omp_prefix.is_dir():
        omp_prefix = _homebrew_prefix("libomp") or Path("/opt/homebrew/opt/libomp")
    omp = _first_existing(
        [
            omp_prefix / "lib" / "libomp.dylib",
            llvm_lib / "libomp.dylib",
        ]
    )
    if omp is not None:
        record(omp, dest)
    else:
        print("skip missing libomp.dylib")

    for tool_dir in (dest / "hipSYCL" / "ext" / "llvm" / "bin", dest / "hipSYCL" / "llvm-to-backend"):
        if not tool_dir.is_dir():
            continue
        for tool in tool_dir.iterdir():
            if tool.is_file():
                tool.chmod(tool.stat().st_mode | 0o111)

    metal = dest / "hipSYCL" / "librt-backend-metal.dylib"
    omp_be = dest / "hipSYCL" / "librt-backend-omp.dylib"
    rt = dest / "libacpp-rt.dylib"
    lld_ok = (llvm_bin_dst / "ld64.lld").is_file() or (llvm_bin_dst / "ld.lld").is_file()
    lld_libs_ok = any(dest.glob("liblld*.dylib"))
    if not rt.is_file():
        raise SystemExit(f"bundle failed: {rt} missing")
    if not omp_be.is_file():
        raise SystemExit(f"bundle failed: {omp_be} missing")
    if not metal.is_file():
        raise SystemExit(f"bundle failed: {metal} missing")
    if not lld_ok:
        raise SystemExit(f"bundle failed: {llvm_bin_dst}/ld.lld missing")
    if not lld_libs_ok:
        raise SystemExit(f"bundle failed: {dest}/liblld*.dylib missing (Homebrew lld@20)")
    print(f"bundled {len(copied)} AdaptiveCpp artefacts into {dest}")


def bundle(acpp_prefix: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        _bundle_darwin(acpp_prefix, dest)
        return

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
