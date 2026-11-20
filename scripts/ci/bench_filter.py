#!/usr/bin/env python3
"""Classify PR paths for the bench gate: product rebuild vs A/B vs docs-only."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

# Same product paths as images.yml's develop gate, plus the C++/python tree.
PRODUCT_PREFIXES = (
    "src/",
    "python/",
    "targets/",
    "cmake/",
    "share/",
    "docker/",
)
PRODUCT_FILES = frozenset(
    {
        "CMakeLists.txt",
        "CMakePresets.json",
        "toolchain.env",
        "pyproject.toml",
        "Dockerfile",
        "Dockerfile.cuda",
        "Dockerfile.cuda12.8",
        "Dockerfile.fzb",
        ".github/workflows/images.yml",
    }
)
# Harness / gate: run A/B but reuse develop's runtime image when product is unchanged.
BENCH_PREFIXES = (
    "benchmarks/",
    "scripts/ci/bench-",
    "scripts/ci/bench_",
)
BENCH_FILES = frozenset(
    {
        "scripts/ci/check-self-hosted-reachability.py",
        ".github/workflows/bench.yml",
    }
)


def _matches(path: str, prefixes: tuple[str, ...], files: frozenset[str]) -> bool:
    if path in files:
        return True
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def classify(files: Iterable[str]) -> dict[str, bool | str]:
    paths = [p.strip() for p in files if p and p.strip()]
    product = any(_matches(p, PRODUCT_PREFIXES, PRODUCT_FILES) for p in paths)
    bench = any(_matches(p, BENCH_PREFIXES, BENCH_FILES) for p in paths)
    run_benches = product or bench
    return {
        "product": product,
        "bench": bench,
        "run_benches": run_benches,
        "docs_only": bool(paths) and not run_benches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Changed paths (or read newline-separated names from stdin)",
    )
    parser.add_argument(
        "--github-output",
        default=None,
        help="Append key=value lines for GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)
    files = list(args.files)
    if not files and not sys.stdin.isatty():
        files = [line.strip() for line in sys.stdin if line.strip()]
    result = classify(files)
    lines = [
        f"product={'true' if result['product'] else 'false'}",
        f"bench={'true' if result['bench'] else 'false'}",
        f"run_benches={'true' if result['run_benches'] else 'false'}",
        f"docs_only={'true' if result['docs_only'] else 'false'}",
    ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
