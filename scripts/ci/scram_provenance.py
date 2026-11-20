#!/usr/bin/env python3
"""Report the SCRAM-derived part of the tree and keep it from growing (#145).

Counts, over tracked files:

* code tagged ``SPDX-License-Identifier: GPL-3.0-or-later`` (derived from SCRAM);
* upstream SCRAM data, fixtures and docs, by the globs in scram_provenance.json;
* files outside the tagged set that ``#include`` a tagged header.

Fails when a file carries the upstream copyright without the GPL tag or with
AGPL wording, or when any count rises above the baseline in
scram_provenance.json. The baseline only moves down: after removing SCRAM code,
run with ``--update-baseline`` and commit the result.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("scram_provenance.json")
GPL_TAG = "SPDX-License-Identifier: GPL-3.0-or-later"
UPSTREAM_COPYRIGHT = re.compile(r"Copyright \(C\) [0-9, -]+ Olzhas Rakhimov")
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
CXX_SUFFIXES = (".h", ".hpp", ".cc", ".cpp", ".tpp", ".inl", ".cc.in", ".h.in")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p and (ROOT / p).is_file()]


def read(path: str) -> str:
    return (ROOT / path).read_bytes().decode("utf-8", "replace")


def resolve(includer: str, target: str, candidates: set[str]) -> str | None:
    for base in (os.path.dirname(includer), "src", ""):
        path = os.path.normpath(os.path.join(base, target))
        if path in candidates:
            return path
    return None


def measure(upstream_globs: list[str]) -> dict:
    files = tracked_files()
    gpl, lines, untagged, agpl = set(), 0, [], []
    for path in files:
        try:
            text = read(path)
        except OSError:
            continue
        header = text[:2048]
        if GPL_TAG in header:
            gpl.add(path)
            lines += text.count("\n")
            if UPSTREAM_COPYRIGHT.search(header) and "Affero" in header:
                agpl.append(path)
        elif UPSTREAM_COPYRIGHT.search(header) and not path.endswith((".md", "NOTICE")):
            untagged.append(path)

    upstream_data = sorted(
        p for p in files if p not in gpl and any(fnmatch.fnmatchcase(p, g) for g in upstream_globs)
    )

    including: dict[str, list[str]] = {}
    for path in files:
        if path in gpl or not path.endswith(CXX_SUFFIXES):
            continue
        hits = {h for t in INCLUDE.findall(read(path)) if (h := resolve(path, t, gpl))}
        if hits:
            including[path] = sorted(hits)

    return {
        "gpl_files": len(gpl),
        "gpl_lines": lines,
        "upstream_data_files": len(upstream_data),
        "including": including,
        "untagged": sorted(untagged),
        "agpl_wording": sorted(agpl),
    }


def summary(m: dict, base: dict) -> str:
    rows = [
        ("GPL-3.0-or-later code files", m["gpl_files"], base.get("gpl_files")),
        ("GPL-3.0-or-later code lines", m["gpl_lines"], None),
        ("Upstream data, fixture and doc files", m["upstream_data_files"], base.get("upstream_data_files")),
        ("Other files including SCRAM headers", len(m["including"]), len(base.get("including_files", []))),
    ]
    out = ["| SCRAM provenance (#145) | now | baseline |", "|---|---:|---:|"]
    out += [f"| {name} | {now} | {'' if b is None else b} |" for name, now, b in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update-baseline", action="store_true", help="write the current counts as the baseline")
    args = ap.parse_args()

    config = json.loads(CONFIG.read_text())
    m = measure(config["upstream_paths"])
    base = config.get("baseline", {})

    if args.update_baseline:
        config["baseline"] = {
            "gpl_files": m["gpl_files"],
            "upstream_data_files": m["upstream_data_files"],
            "including_files": sorted(m["including"]),
        }
        CONFIG.write_text(json.dumps(config, indent=2) + "\n")
        print(f"baseline written to {CONFIG.relative_to(ROOT)}")
        return 0

    table = summary(m, base)
    print(table)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as f:
            f.write(table + "\n")

    errors = []
    errors += [f"{p}: upstream SCRAM copyright without '{GPL_TAG}'" for p in m["untagged"]]
    errors += [f"{p}: upstream SCRAM notice must say GNU General Public License, not Affero" for p in m["agpl_wording"]]
    for key in ("gpl_files", "upstream_data_files"):
        if m[key] > base.get(key, 0):
            errors.append(f"{key} rose from {base.get(key, 0)} to {m[key]}")
    allowed = set(base.get("including_files", []))
    for path, headers in sorted(m["including"].items()):
        if path not in allowed:
            errors.append(f"{path}: new include of SCRAM-derived {', '.join(headers)}")

    if errors:
        print("\nSCRAM provenance check failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    shrunk = (
        m["gpl_files"] < base.get("gpl_files", 0)
        or m["upstream_data_files"] < base.get("upstream_data_files", 0)
        or set(m["including"]) < allowed
    )
    if shrunk:
        print("\nCounts are below the baseline; lock that in with "
              "scripts/ci/scram_provenance.py --update-baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
