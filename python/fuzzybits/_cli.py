"""Python ``fzb`` console script. Flag surface matches ``targets/fzb/main.cpp``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from ._api import compile as compile_pcfb
from ._api import exact, inspect, pcfb_to_nombrilo, run

ROOT_USAGE = (
    "Usage: fzb <command> [options]\n\n"
    "Commands:\n"
    "  compile   OpenPSA/MEF XML or nombrilo.circuit → .pcfb\n"
    "  export    .pcfb → nombrilo.circuit (.safetensors)\n"
    "  inspect   Summarise, validate, or exactly evaluate a .pcfb\n"
    "  run       Monte Carlo a .pcfb (estimate and CI per query)\n"
)

INSPECT_JSON_KEYS = (
    "schema_version",
    "tensors",
    "operators",
    "samples",
    "queries",
    "opcodes",
    "sample_source",
    "query_names",
    "metadata",
)

SEED_REJECTED = (
    "fzb run: --seed is not supported yet; results are deterministic "
    "per model and trial count (Philox counters are "
    "{pdag_idx+1, event_idx+1, sample_idx+1, (iteration+1)<<6}, "
    "not Settings::seed). The only accepted value is 372, the "
    "scram-cli default.\n"
)

SCRAM_CLI_DEFAULT_SEED = 372


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse hook
        self.print_help(sys.stderr)
        print(f"Option error: {message}", file=sys.stderr)
        raise SystemExit(1)


def _print_root_usage(out: TextIO) -> None:
    out.write(ROOT_USAGE)


def _cli_error(command: str, exc: BaseException) -> int:
    msg = str(exc)
    for prefix in (f"{command}: ", "compile: ", "export: ", "inspect: ", "run: "):
        if msg.startswith(prefix):
            msg = msg[len(prefix) :]
            break
    print(f"fzb {command}: {msg}", file=sys.stderr)
    return 1


def _parse_trials(text: str) -> int:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid --trials '{text}'") from exc
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"invalid --trials '{text}'")
    return int(value)


def cmd_compile(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb compile",
        usage="fzb compile model.xml|model.safetensors [-c N] [--no-kn] [--target NAME] [--weights FILE] [-o model.pcfb]",
        description="fzb compile options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("-o", "--output", help="output .pcfb path")
    parser.add_argument(
        "-c",
        "--compilation-passes",
        type=int,
        default=2,
        help="0=off 1=null-only 2=optimize 4-8=multipass",
    )
    parser.add_argument("--no-kn", action="store_true", help="expand k/n to and/or")
    parser.add_argument("--no-xor", action="store_true", help="expand xor to and/or")
    parser.add_argument(
        "--recognize-atleast",
        action="store_true",
        help="recognize expanded threshold groups (default on)",
    )
    parser.add_argument(
        "--no-recognize-atleast",
        action="store_true",
        help="disable k/n group recognition",
    )
    parser.add_argument("--recognize-xor", action="store_true", help="recognize expanded xor (default on)")
    parser.add_argument("--no-recognize-xor", action="store_true", help="disable xor group recognition")
    parser.add_argument("--keep-null", action="store_true", help="keep NULL/pass-through gates")
    parser.add_argument("--ccf", action="store_true", help="expand common-cause failure groups")
    parser.add_argument("--allow-extern", action="store_true", help="**UNSAFE** allow external libraries")
    parser.add_argument("--target", help="top-event / gate / nombrilo name")
    parser.add_argument("--weights", help="nombrilo variant weights file")
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        help="extra MEF names / PDAG indices as additional queries (repeatable)",
    )
    parser.add_argument("input_file", nargs="*", help="MEF XML or nombrilo.circuit file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb compile: missing input file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    options: dict[str, Any] = {
        "no_xor": bool(args.no_xor),
        "keep_null": bool(args.keep_null),
        "ccf": bool(args.ccf),
        "allow_extern": bool(args.allow_extern),
    }
    if args.no_recognize_atleast:
        options["recognize_atleast_groups"] = False
    elif args.recognize_atleast:
        options["recognize_atleast_groups"] = True
    if args.no_recognize_xor:
        options["recognize_xor_groups"] = False
    elif args.recognize_xor:
        options["recognize_xor_groups"] = True
    if args.target:
        options["target"] = args.target
    if args.weights:
        options["weights"] = args.weights
    watch = [part for item in args.watch for part in str(item).replace(";", ",").split(",") if part.strip()]
    if watch:
        options["watch"] = watch

    try:
        blob = compile_pcfb(
            args.input_file,
            pdag_c=int(args.compilation_passes),
            no_kn=bool(args.no_kn),
            **options,
        )
    except Exception as exc:
        return _cli_error("compile", exc)

    if args.output:
        out_path = Path(args.output)
    else:
        # Native default is `{stem}.pcfb` in the process cwd, not beside the XML.
        out_path = Path(Path(args.input_file[0]).stem + ".pcfb")
    try:
        out_path.write_bytes(blob)
    except OSError as exc:
        print(f"fzb compile: cannot write {out_path}", file=sys.stderr)
        print(f"fzb compile: {exc}", file=sys.stderr)
        return 1

    model = ""
    top = ""
    try:
        meta = inspect(blob).get("metadata") or {}
        model = str(meta.get("general.model_name") or "")
        top = str(meta.get("fuzzybits.target") or "")
    except Exception:
        pass
    print(
        f"wrote {out_path} ({len(blob)} bytes) model={model} top={top}",
        file=sys.stderr,
    )
    return 0


def cmd_export(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb export",
        usage="fzb export --to nombrilo model.pcfb -o out.safetensors [--strict]",
        description="fzb export options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("--to", help="target format (nombrilo)")
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument("--strict", action="store_true", help="omit fuzzybits.pcfb.* extras")
    parser.add_argument("input_file", nargs="?", help=".pcfb file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb export: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    if args.to not in {"nombrilo", "nombrilo.circuit"}:
        print("fzb export: --to nombrilo is required\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    try:
        blob = pcfb_to_nombrilo(args.input_file, extras=not args.strict)
    except Exception as exc:
        return _cli_error("export", exc)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(Path(args.input_file).stem + ".safetensors")
    try:
        out_path.write_bytes(blob)
    except OSError as exc:
        print(f"fzb export: cannot write {out_path}", file=sys.stderr)
        print(f"fzb export: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out_path} ({len(blob)} bytes)", file=sys.stderr)
    return 0


def _inspect_text(summary: Mapping[str, Any], out: TextIO) -> None:
    schema = int(summary.get("schema_version") or 0)
    out.write(f"schema_version  0x{schema:x}\n")
    out.write(f"tensors         {int(summary.get('tensors') or 0)}\n")
    out.write(f"operators       {int(summary.get('operators') or 0)}\n")
    out.write(f"samples         {int(summary.get('samples') or 0)}\n")
    out.write(f"queries         {int(summary.get('queries') or 0)}\n")
    opcodes = summary.get("opcodes") or {}
    if opcodes:
        out.write("opcodes\n")
        for name, n in opcodes.items():
            out.write(f"  {name}  {n}\n")
    sources = summary.get("sample_source") or {}
    if sources:
        out.write("sample_source\n")
        for name, n in sources.items():
            out.write(f"  {name}  {n}\n")
    names = summary.get("query_names") or []
    if names:
        out.write("query_names\n")
        for name in names:
            out.write(f"  {name}\n")
    metadata = summary.get("metadata") or {}
    if metadata:
        out.write("metadata\n")
        for key, value in metadata.items():
            out.write(f"  {key}  {value}\n")


def _inspect_json(summary: Mapping[str, Any], out: TextIO) -> None:
    payload = {key: summary.get(key) for key in INSPECT_JSON_KEYS}
    json.dump(payload, out, ensure_ascii=False, indent=2)
    out.write("\n")


def cmd_inspect(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb inspect",
        usage="fzb inspect model.pcfb [--validate] [--json] [--exact]",
        description="fzb inspect options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("--validate", action="store_true", help="run pcfb::validate; exit non-zero on diagnostics")
    parser.add_argument("--json", action="store_true", help="print summary as JSON")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="exhaustive exact P(query) for small models (≤ 30 events)",
    )
    parser.add_argument("input_file", nargs="?", help=".pcfb file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb inspect: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    try:
        summary = inspect(args.input_file)
    except Exception as exc:
        return _cli_error("inspect", exc)

    rc = 0
    if args.validate:
        diagnostics = list(summary.get("diagnostics") or [])
        if diagnostics:
            rc = 1
            for diag in diagnostics:
                if isinstance(diag, Mapping):
                    location = diag.get("location", "")
                    message = diag.get("message", diag)
                else:
                    location = getattr(diag, "location", "")
                    message = getattr(diag, "message", diag)
                print(f"validate: {location}: {message}", file=sys.stderr)
        elif not args.json:
            print("validate: ok", file=sys.stderr)

    if args.json:
        _inspect_json(summary, sys.stdout)
    else:
        _inspect_text(summary, sys.stdout)

    if args.exact:
        try:
            p = exact(args.input_file)
        except Exception as exc:
            msg = str(exc)
            print(f"fzb inspect --exact: {msg}", file=sys.stderr)
            return 1
        print(f"exact_probability  {p:.16e}")
    return rc


def cmd_run(argv: Sequence[str]) -> int:
    parser = _Parser(
        prog="fzb run",
        usage="fzb run model.pcfb --trials N [--seed S] [--backend auto|cuda|omp|metal] [--json]",
        description="fzb run options",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help", help="display this help message")
    parser.add_argument("-N", "--trials", help="Bernoulli trials (e.g. 1e9, 1000000)")
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=None,
        help="must be 372 (the scram-cli default); other values are rejected",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="AdaptiveCpp backend: auto, cuda, omp, or metal",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("input_file", nargs="?", help=".pcfb file")
    args = parser.parse_args(list(argv))
    if not args.input_file:
        print("fzb run: missing .pcfb file\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    if args.trials is None:
        print("fzb run: --trials is required\n", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1
    try:
        trials = _parse_trials(args.trials)
    except ValueError as exc:
        print(f"fzb run: {exc}", file=sys.stderr)
        return 1
    if args.seed is not None and args.seed != SCRAM_CLI_DEFAULT_SEED:
        sys.stderr.write(SEED_REJECTED)
        return 1

    backend = str(args.backend or "auto").lower()
    if backend not in {"auto", "cuda", "omp", "metal"}:
        print(
            f"fzb run: unknown backend '{args.backend}' (expected auto, cuda, omp, or metal)",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run(args.input_file, trials=trials, backend=backend)
    except Exception as exc:
        return _cli_error("run", exc)

    if args.json:
        payload = {
            "queries": [
                {
                    "name": row.name,
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "n": row.n,
                    "hits": row.hits,
                }
                for row in rows
            ]
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for row in rows:
            print(
                f"{row.name}  {row.estimate:.8e}  [{row.ci_low:.8e}, {row.ci_high:.8e}]  "
                f"n={row.n}  hits={row.hits}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_root_usage(sys.stderr)
        return 1
    cmd = args[0]
    if cmd in {"-h", "--help", "help"}:
        _print_root_usage(sys.stdout)
        return 0
    if cmd in {"-v", "--version", "version"}:
        print(f"fzb {__version__}")
        return 0
    if cmd == "compile":
        return cmd_compile(args[1:])
    if cmd == "export":
        return cmd_export(args[1:])
    if cmd == "inspect":
        return cmd_inspect(args[1:])
    if cmd == "run":
        return cmd_run(args[1:])
    print(f"fzb: unknown command '{cmd}'", file=sys.stderr)
    _print_root_usage(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
