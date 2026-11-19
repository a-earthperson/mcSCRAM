#!/usr/bin/env python3
"""Rewrite RPATH on a staged fuzzybits wheel prefix to $ORIGIN / @loader_path.

Layout after CMake install + AdaptiveCpp bundle:

    <prefix>/fuzzybits/_native_*.so
    <prefix>/fuzzybits/libscram.so|.dylib
    <prefix>/fuzzybits/libmc.so          (Linux only)
    <prefix>/fuzzybits/lib/libacpp-rt.so|.dylib
    <prefix>/fuzzybits/lib/hipSYCL/...
    <prefix>/fuzzybits/lib/hipSYCL/ext/llvm/bin/{opt,llc,ld.lld}

AdaptiveCpp locates backend plugins as ``dirname(libacpp-rt)/hipSYCL`` and the
SSCP redistributable package as ``dirname(libacpp-common)/hipSYCL/ext``. Keep
that tree intact; only RPATH / install names are rewritten. NVIDIA driver
libs (libcuda.so*) are deleted if the deploy step copied them.
"""

from __future__ import annotations

import os
import re
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

LOADER_RPATHS = {
    "ext": "@loader_path:@loader_path/lib",
    "lib": "@loader_path",
    "backend": "@loader_path:@loader_path/..:@loader_path/llvm-to-backend",
    "llvm_to_backend": "@loader_path:@loader_path/../..",
    "llvm_bin": "@loader_path/../../../..",
}

MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
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


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) in MACHO_MAGICS
    except OSError:
        return False


def _otool_libs(path: Path) -> list[str]:
    result = _run(["otool", "-L", str(path)])
    if result.returncode != 0:
        return []
    libs: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        text = line.strip()
        if not text:
            continue
        name = text.split(" (compatibility", 1)[0].strip()
        if name:
            libs.append(name)
    return libs


def _otool_id(path: Path) -> str | None:
    result = _run(["otool", "-D", str(path)])
    if result.returncode != 0:
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[-1]
    return None


def _otool_rpaths(path: Path) -> list[str]:
    result = _run(["otool", "-l", str(path)])
    if result.returncode != 0:
        return []
    rpaths: list[str] = []
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "LC_RPATH" not in line:
            continue
        for follow in lines[i + 1 : i + 6]:
            match = re.search(r"path\s+(\S+)", follow)
            if match:
                rpaths.append(match.group(1))
                break
    return rpaths


def _codesign(path: Path) -> None:
    _run(["codesign", "--force", "--sign", "-", str(path)])


def _install_name_tool(args: list[str], path: Path) -> None:
    result = _run(["install_name_tool", *args, str(path)])
    if result.returncode != 0:
        sys.stderr.write(f"install_name_tool {args} failed on {path}: {result.stderr}\n")
        raise SystemExit(result.returncode or 1)


def _relocate_darwin(root: Path) -> None:
    macho: list[Path] = []
    bundled: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if not _is_macho(path):
            continue
        macho.append(path)
        bundled[path.name] = path

    for path in macho:
        rel = path.relative_to(root)
        kind = _classify(rel) or "ext"
        rpath = LOADER_RPATHS[kind]
        _chmod_u_write(path)

        current_id = _otool_id(path)
        if current_id and path.suffix in {".dylib", ".so"} and not path.name.startswith("_native"):
            new_id = f"@rpath/{path.name}"
            if current_id != new_id:
                print(f"ID {new_id} -> {rel}")
                _install_name_tool(["-id", new_id], path)

        for dep in _otool_libs(path):
            dep_name = Path(dep).name
            target = bundled.get(dep_name)
            if target is None or target == path:
                continue
            new_dep = "@loader_path/" + os.path.relpath(target, path.parent)
            if dep == new_dep:
                continue
            print(f"CHANGE {dep} -> {new_dep} [{rel}]")
            _install_name_tool(["-change", dep, new_dep], path)

        for old in _otool_rpaths(path):
            _install_name_tool(["-delete_rpath", old], path)
        for entry in rpath.split(":"):
            print(f"RPATH {entry} -> {rel}")
            add = _run(["install_name_tool", "-add_rpath", entry, str(path)])
            if add.returncode != 0 and "would duplicate path" not in (add.stderr or ""):
                sys.stderr.write(f"install_name_tool -add_rpath failed on {path}: {add.stderr}\n")
                raise SystemExit(add.returncode or 1)
        _codesign(path)


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

    if sys.platform == "darwin":
        _relocate_darwin(root)
        return 0

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
