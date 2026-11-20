#!/usr/bin/env python3
"""Compose (and optionally post) the sticky PR comment for the bench gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MARKER = "<!-- fuzzybits-bench-gate -->"

# Jobs whose harness verdict can fail the required summary. CPU is report-only (#124).
# rtx5060ti is shared-device report-only on shelly (never blocks).
BLOCKING_JOBS = ("accuracy", "v100s", "rtx3070")
REPORT_ONLY_JOBS = ("cpu", "rtx5060ti")

_FAIL = {"block", "error"}
# inconclusive: host busy — fail the gate so an untested PR is not green.
_FAIL_GPU = {"block", "error", "inconclusive"}


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1,
            "kind": "error",
            "verdict": "error",
            "error": f"could not read {path.name}: {exc}",
        }
    if not isinstance(payload, dict):
        return {"kind": "error", "verdict": "error", "error": f"{path.name} is not an object"}
    return payload


def _verdict(payload: Mapping[str, Any] | None, *, missing: str = "skipped") -> str:
    if payload is None:
        return missing
    return str(payload.get("verdict") or missing)


def _job_fails(name: str, verdict: str) -> bool:
    if name == "rtx5060ti":
        return False
    if name in REPORT_ONLY_JOBS:
        return verdict == "error"
    if name in BLOCKING_JOBS:
        return verdict in _FAIL_GPU
    return verdict in _FAIL


def overall_verdict(jobs: Mapping[str, Mapping[str, Any] | None]) -> str:
    """Worst blocking verdict. CPU warn/block does not fail the gate."""
    rank = {
        "pass": 0,
        "pass (low power)": 0,
        "skipped": 0,
        "infeasible": 1,
        "known_failure": 1,
        "unsupported": 1,
        "timeout": 2,
        "warn": 3,
        "block": 4,
        "error": 5,
        "inconclusive": 6,
    }
    display = "pass"
    failing: str | None = None
    for name, payload in jobs.items():
        verdict = _verdict(payload)
        if name == "rtx5060ti" and verdict in {
            "block",
            "error",
            "inconclusive",
            "unsupported",
        }:
            # Shared GPU, report-only: surface as warn, never fail the gate.
            verdict = "warn"
        elif name in REPORT_ONLY_JOBS and verdict == "block":
            verdict = "warn"
        if _job_fails(name, verdict):
            if failing is None or rank.get(verdict, 0) > rank.get(failing, 0):
                failing = verdict
        if verdict != "skipped" and rank.get(verdict, 0) > rank.get(display, 0):
            display = verdict
    return failing or display


def _context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    ctx = payload.get("context")
    return ctx if isinstance(ctx, dict) else {}


def _summarize_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "_no results_\n"
    try:
        from benchmarks.harness.summarize import summarize_path
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmarks.harness.summarize import summarize_path
    try:
        return summarize_path(path)
    except Exception as exc:  # noqa: BLE001 — comment must still post
        return f"_summarize failed: {exc}_\n"


def render(
    *,
    jobs: Mapping[str, Path | None],
    run_url: str,
    notices: list[str],
    inject: str | None = None,
    trusted: bool = True,
    run_benches: bool = True,
) -> tuple[str, str]:
    """Return ``(markdown, overall_verdict)``."""
    loaded = {name: _load(path) for name, path in jobs.items()}
    overall = overall_verdict(loaded)
    if not trusted:
        overall = "pass"
    elif not run_benches:
        overall = "pass"

    ctx = {}
    for payload in loaded.values():
        ctx = _context(payload)
        if ctx:
            break

    lines = [
        MARKER,
        "# Bench gate",
        "",
        f"- overall: **{overall}**",
        f"- [Actions run]({run_url})" if run_url else "- Actions run: _(none)_",
    ]
    if ctx.get("base_sha") or ctx.get("head_sha"):
        lines.append(f"- base SHA: `{ctx.get('base_sha') or '—'}`")
        lines.append(f"- head SHA: `{ctx.get('head_sha') or '—'}`")
    if ctx.get("base_image") or ctx.get("head_image"):
        lines.append(f"- base image: `{ctx.get('base_image') or '—'}`")
        lines.append(f"- base digest: `{ctx.get('base_digest') or '—'}`")
        lines.append(f"- head image: `{ctx.get('head_image') or '—'}`")
        lines.append(f"- head digest: `{ctx.get('head_digest') or '—'}`")
    if inject:
        lines.append(f"- inject: `{inject}`")
    lines.append(
        "- CPU A/B is **report-only** (`CPU_GATE_BLOCKS = False`, "
        "[#124](https://github.com/fuzzybit-org/fuzzybits/issues/124)); "
        "it does not fail this check."
    )
    try:
        from benchmarks.harness.summarize import shared_device_note
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmarks.harness.summarize import shared_device_note
    rtx_note = shared_device_note(_context(loaded.get("rtx3070")))
    if rtx_note:
        lines.append(f"- **RTX 3070:** {rtx_note} (co-runs; a block may be interference)")
    rtx5060_note = shared_device_note(_context(loaded.get("rtx5060ti")))
    if rtx5060_note:
        lines.append(f"- **RTX 5060 Ti:** {rtx5060_note} (never blocks)")
    elif loaded.get("rtx5060ti"):
        lines.append("- **RTX 5060 Ti:** shared GPU, report-only (never blocks)")
    if notices:
        lines.append("")
        for notice in notices:
            lines.append(f"- notice: {notice}")

    labels = {
        "accuracy": "Accuracy (head, CPU)",
        "v100s": "V100S A/B (`gpu_total_ms`, blocks at ci_lo > 1.03)",
        "rtx3070": "RTX 3070 A/B (`gpu_total_ms`, blocks at ci_lo > 1.03; shared GPU)",
        "rtx5060ti": "RTX 5060 Ti A/B (shelly, shared GPU, report-only)",
        "cpu": "CPU A/B (`cpu_sample_s`, report-only)",
    }
    for name, title in labels.items():
        payload = loaded.get(name)
        path = jobs.get(name)
        lines += ["", f"## {title}", ""]
        if payload is None:
            lines.append("_no artifact (job skipped or failed before upload)_")
            continue
        kind = payload.get("kind")
        if kind == "skip":
            lines.append(f"- skipped: {payload.get('reason') or 'n/a'}")
            lines.append(f"- verdict: **{payload.get('verdict') or 'pass'}**")
            continue
        if kind == "error" and payload.get("error"):
            lines.append(f"- error: {payload['error']}")
            continue
        lines.append(_summarize_file(path).rstrip())

    lines += [
        "",
        "## Artifacts",
        "",
        "JSON results are uploaded on the Actions run (accuracy, v100s, rtx3070, rtx5060ti, cpu).",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n", overall


def write_skip(path: Path, *, job: str, reason: str, verdict: str = "pass") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "skip",
        "job": job,
        "reason": reason,
        "verdict": verdict,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _gh(args: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def post_comment(*, repo: str, pr: int, body: str) -> None:
    raw = _gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/issues/{pr}/comments",
        ]
    )
    comments: list[dict[str, Any]] = []
    # --paginate may concatenate JSON arrays.
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                comments = parsed
        except json.JSONDecodeError:
            # Concatenated arrays: ][
            blob = "[" + raw.replace("][", ",") + "]"
            parsed = json.loads(blob)
            comments = parsed if isinstance(parsed, list) else []
    existing = next((c for c in comments if MARKER in str(c.get("body") or "")), None)
    payload = json.dumps({"body": body})
    if existing:
        comment_id = existing["id"]
        _gh(
            ["api", "-X", "PATCH", f"repos/{repo}/issues/comments/{comment_id}", "--input", "-"],
            input_text=payload,
        )
        return
    _gh(
        ["api", f"repos/{repo}/issues/{pr}/comments", "--input", "-"],
        input_text=payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, default=None)
    parser.add_argument("--v100s", type=Path, default=None)
    parser.add_argument("--rtx3070", type=Path, default=None)
    parser.add_argument("--rtx5060ti", type=Path, default=None)
    parser.add_argument("--cpu", type=Path, default=None)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--notice", action="append", default=[])
    parser.add_argument("--inject", default=None)
    parser.add_argument("--trusted", action="store_true", default=False)
    parser.add_argument("--run-benches", action="store_true", default=False)
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail if a blocking job artifact is missing (same-repo bench run)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--post-pr", type=int, default=None)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--write-skip",
        type=Path,
        default=None,
        help="Write a skip JSON and exit (use with --skip-job/--skip-reason)",
    )
    parser.add_argument("--skip-job", default="")
    parser.add_argument("--skip-reason", default="")
    parser.add_argument("--skip-verdict", default="pass")
    args = parser.parse_args(argv)

    if args.write_skip is not None:
        write_skip(
            args.write_skip,
            job=args.skip_job,
            reason=args.skip_reason,
            verdict=args.skip_verdict,
        )
        return 0

    jobs = {
        "accuracy": args.accuracy,
        "v100s": args.v100s,
        "rtx3070": args.rtx3070,
        "rtx5060ti": args.rtx5060ti,
        "cpu": args.cpu,
    }
    markdown, overall = render(
        jobs=jobs,
        run_url=args.run_url,
        notices=list(args.notice),
        inject=args.inject,
        trusted=args.trusted,
        run_benches=args.run_benches,
    )
    if args.require and args.trusted and args.run_benches:
        missing = [name for name in BLOCKING_JOBS if jobs.get(name) is None or not jobs[name].is_file()]
        if missing:
            overall = "error"
            markdown += f"\n- missing blocking artifacts: {', '.join(missing)}\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    if args.post_pr:
        if not args.repo:
            raise SystemExit("post-pr requires --repo or GITHUB_REPOSITORY")
        post_comment(repo=args.repo, pr=int(args.post_pr), body=markdown)
    if overall in _FAIL_GPU:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
