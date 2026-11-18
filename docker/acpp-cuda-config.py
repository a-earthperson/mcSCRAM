#!/usr/bin/env python3
"""Resolve AdaptiveCpp CUDA SMCP target string and clang PTX flags.

AdaptiveCpp's CUDA SMCP driver (v25.10 / develop) passes
`--cuda-gpu-arch=sm_XX` for cuda.integrated-multipass and
`--cuda-device-only --cuda-gpu-arch=sm_XX -S` for cuda.explicit-multipass
(PTX text, driver-JIT). It has no --cuda-include-ptx of its own; we forward
clang's flags via ACPP_EXTRA_COMPILE_OPTIONS.

CUDA_EMBED_PTX:
  none     SASS only (--no-cuda-include-ptx=all)
  highest  SASS for each sm_XX + PTX of the highest arch (default)
  all      SASS + PTX for every arch (--cuda-include-ptx=all)
  ptx-only rewrite cuda / cuda.integrated-multipass -> cuda.explicit-multipass
           (clang -S PTX; clang cannot emit compute_XX virtual-arch fatbins)
"""
from __future__ import annotations

import os
import re
import sys


def normalize_targets(targets: str) -> str:
    """Accept comma-separated ACPP_TARGETS (Docker ARG cannot easily pass ';')."""
    parts = [p.strip() for p in targets.replace(",", ";").split(";")]
    return ";".join(p for p in parts if p)


def sm_archs(targets: str) -> list[int]:
    return [int(x) for x in re.findall(r"sm_([0-9]+)", targets)]


def rewrite_ptx_only(targets: str) -> str:
    parts: list[str] = []
    for raw in targets.split(";"):
        p = raw.strip()
        if not p:
            continue
        if (
            p == "generic"
            or p.startswith("omp")
            or p.startswith("hip")
            or p.startswith("cuda-nvcxx")
            or p.startswith("cuda.explicit-multipass")
        ):
            parts.append(p)
            continue
        m = re.match(r"cuda(?:\.integrated-multipass)?:(.+)$", p)
        if m:
            parts.append("cuda.explicit-multipass:" + m.group(1))
            continue
        parts.append(p)
    return ";".join(parts)


def ptx_flags(mode: str, targets: str) -> str:
    if mode == "none":
        return "--no-cuda-include-ptx=all"
    if mode == "all":
        return "--cuda-include-ptx=all"
    if mode == "highest":
        nums = sm_archs(targets)
        return f"--cuda-include-ptx=sm_{max(nums)}" if nums else ""
    if mode == "ptx-only":
        return ""
    raise SystemExit(f"error: CUDA_EMBED_PTX must be none|highest|all|ptx-only, got {mode!r}")


def main() -> None:
    mode = os.environ.get("CUDA_EMBED_PTX", "highest")
    targets = normalize_targets(os.environ.get("ACPP_TARGETS", "cuda:sm_86"))
    profile = os.environ.get("ACPP_COMPILER_FEATURE_PROFILE", "minimal")
    reject_volta = os.environ.get("REJECT_VOLTA_OFFLINE", "0") == "1"

    if mode not in {"none", "highest", "all", "ptx-only"}:
        raise SystemExit(f"error: CUDA_EMBED_PTX must be none|highest|all|ptx-only, got {mode!r}")

    if reject_volta and re.search(r"sm_(50|52|53|60|61|62|70|72)([^0-9]|$)", targets):
        raise SystemExit(
            "error: CUDA 13.x removed offline compilation for sm_50–sm_72 "
            f"(Volta and older). ACPP_TARGETS={targets!r} is not supported by "
            "Dockerfile.cuda. Use Dockerfile.cuda12.8 (CUDA 12.8) for sm_70."
        )

    if re.search(r"(^|;)generic(;|$)", targets) and profile != "full":
        raise SystemExit(
            "error: ACPP_TARGETS contains 'generic' (SSCP) but AdaptiveCpp was "
            f"built with ACPP_COMPILER_FEATURE_PROFILE={profile}. Rebuild with "
            "--build-arg ACPP_COMPILER_FEATURE_PROFILE=full."
        )

    effective = rewrite_ptx_only(targets) if mode == "ptx-only" else targets
    flags = ptx_flags(mode, targets)

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/share"
    os.makedirs(out_dir, exist_ok=True)
    (open(os.path.join(out_dir, "acpp-targets"), "w")).write(effective + "\n")
    (open(os.path.join(out_dir, "cuda-embed-ptx"), "w")).write(mode + "\n")
    (open(os.path.join(out_dir, "cuda-ptx-flags"), "w")).write(flags + "\n")
    print(f"ACPP_TARGETS={effective}")
    print(f"CUDA_EMBED_PTX={mode} flags={flags or '<explicit-multipass PTX>'}")


if __name__ == "__main__":
    main()
