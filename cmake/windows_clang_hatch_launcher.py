#!/usr/bin/env python3
"""CMAKE_{C,CXX}_COMPILER_LAUNCHER: inject the MSVC STL Clang-version hatch.

Boost FetchContent (and other nested project() trees) compile TUs without the
superbuild's CMAKE_CXX_FLAGS / directory compile definitions. Inserting
-D_ALLOW_COMPILER_VERSION_MISMATCH on the real compiler command line is
independent of that. No-op on non-Clang compilers.
"""

from __future__ import annotations

import os
import subprocess
import sys

HATCH = "-D_ALLOW_COMPILER_VERSION_MISMATCH"
NOMINMAX = "-DNOMINMAX"


def _is_clang(compiler: str) -> bool:
    name = os.path.basename(compiler).lower()
    return "clang" in name


def main(argv: list[str]) -> int:
    if not argv:
        return 2
    compiler, rest = argv[0], argv[1:]
    extra: list[str] = []
    if _is_clang(compiler):
        joined = " ".join(rest)
        if HATCH not in joined and "/D_ALLOW_COMPILER_VERSION_MISMATCH" not in joined:
            extra.append(HATCH)
        if NOMINMAX not in joined and "/DNOMINMAX" not in joined:
            extra.append(NOMINMAX)
    return subprocess.call([compiler, *extra, *rest])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
