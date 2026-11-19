"""``python -m benchmarks.history merge|index|publish``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .index import (
    INDEX_REL,
    build_index,
    merge_runs,
    publish_payload,
)


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_merge(args: argparse.Namespace) -> int:
    payloads: list[dict[str, Any]] = []
    for path in args.inputs:
        payloads.append(_load(path))
    if args.accuracy:
        payloads.append(_load(args.accuracy))
    if args.perf:
        payloads.append(_load(args.perf))
    if not payloads:
        raise SystemExit("history merge: pass --accuracy/--perf and/or positional inputs")
    merged = merge_runs(
        payloads,
        version=args.version,
        driver=args.driver or None,
        runner_name=args.runner_name or None,
    )
    _write(Path(args.out), merged)
    print(f"merged {len(payloads)} run(s) -> {args.out} verdict={merged.get('verdict')}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.root)
    index = build_index(root)
    out = Path(args.out) if args.out else root / INDEX_REL
    _write(out, index)
    print(f"index {index.get('n_points')} point(s) -> {out}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    payload = _load(args.results)
    written = publish_payload(
        payload,
        dest=Path(args.dest),
        version=args.version,
    )
    print(f"published {written['path']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.history",
        description="Merge harness runs and publish bench/v1 history (#123).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_merge = sub.add_parser("merge", help="combine accuracy + perf JSON into one run")
    p_merge.add_argument("inputs", nargs="*", help="additional results JSON files")
    p_merge.add_argument("--accuracy", default=None)
    p_merge.add_argument("--perf", default=None)
    p_merge.add_argument("--version", required=True, help="v0.17.1 or develop")
    p_merge.add_argument("--driver", default="")
    p_merge.add_argument("--runner-name", default="")
    p_merge.add_argument("--out", required=True)
    p_merge.set_defaults(func=cmd_merge)

    p_index = sub.add_parser("index", help="rebuild bench/v1/index.json from results/")
    p_index.add_argument("--root", required=True, help="checkout of the bench-results branch")
    p_index.add_argument("--out", default=None)
    p_index.set_defaults(func=cmd_index)

    p_pub = sub.add_parser("publish", help="copy one merged JSON without rewriting older files")
    p_pub.add_argument("--results", required=True)
    p_pub.add_argument("--dest", required=True, help="checkout of the bench-results branch")
    p_pub.add_argument("--version", default=None)
    p_pub.set_defaults(func=cmd_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
