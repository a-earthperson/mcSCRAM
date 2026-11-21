"""Markdown summary of a versioned results JSON (PR comment body)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .guards import format_foreign_load


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not (number == number):  # NaN
        return "nan"
    if abs(number) != 0.0 and (abs(number) < 1e-4 or abs(number) >= 1e4):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([line, sep, *body])


def shared_device_note(context: Mapping[str, Any] | None) -> str | None:
    """``shared GPU, report-only, foreign load: parakeet 3.0 GiB`` or None."""
    if not context:
        return None
    load = context.get("load_before") if isinstance(context.get("load_before"), Mapping) else {}
    shared = bool(context.get("shared_device") or (load or {}).get("shared_device"))
    report_only = bool(context.get("report_only"))
    bits: list[str] = []
    if shared:
        bits.append("shared GPU")
    if report_only:
        bits.append("report-only")
    if shared:
        text = (load or {}).get("foreign_load") if isinstance(load, Mapping) else None
        if not text:
            text = format_foreign_load(load if isinstance(load, Mapping) else None)
        bits.append(f"foreign load: {text}")
    if not bits:
        return None
    return ", ".join(bits)


def summarize_run(payload: Mapping[str, Any]) -> str:
    context = payload.get("context") or {}
    models = list(payload.get("models") or [])
    lines = [
        f"# Benchmark `{payload.get('kind')}` ({payload.get('dataset')})",
        "",
        f"- verdict: **{payload.get('verdict', 'n/a')}**",
        f"- created: {payload.get('created_utc')}",
        f"- image: `{context.get('image')}`",
        f"- digest: `{context.get('image_digest') or ''}`",
        f"- backend: `{context.get('backend')}` device `{context.get('device')}`",
        f"- runner / context: `{context.get('runner')}` / `{context.get('docker_context')}`",
    ]
    note = shared_device_note(context if isinstance(context, Mapping) else None)
    if note:
        lines.append(f"- {note}")
    lines += [""]
    accuracy_rows = []
    perf_rows = []
    compile_rows = []
    for model in models:
        kind = model.get("mode") or model.get("kind")
        if kind == "accuracy" or model.get("expected_hits") is not None:
            z_cell = "—" if model.get("z") is None else _fmt_float(model.get("z"), 3)
            accuracy_rows.append(
                [
                    model.get("id"),
                    _fmt_float(model.get("p_ref"), 8),
                    _fmt_float(model.get("estimate"), 8),
                    _fmt_float(model.get("expected_hits"), 3),
                    model.get("hits"),
                    z_cell,
                    model.get("verdict"),
                ]
            )
        if kind == "perf":
            profiler = model.get("profiler") or {}
            perf_rows.append(
                [
                    model.get("id"),
                    _fmt_float(model.get("wall_s"), 4),
                    _fmt_float(profiler.get("primary_ms") or profiler.get("basic_event_ms"), 3),
                    _fmt_float(profiler.get("total_gpu_kernel_ms"), 3),
                    model.get("verdict") or model.get("status") or "",
                ]
            )
        if kind == "compile" or (
            model.get("tensors") is not None and kind not in {"accuracy", "perf"}
        ):
            inspect = model.get("inspect") or {}
            compile_rows.append(
                [
                    model.get("id"),
                    _fmt_float(model.get("compile_wall_s"), 3),
                    inspect.get("tensors") or model.get("tensors") or "—",
                    inspect.get("operators") or model.get("operators") or "—",
                    inspect.get("samples") or model.get("samples") or "—",
                    model.get("verdict") or model.get("status") or (
                        model.get("error") or "ok"
                    ),
                ]
            )
    if accuracy_rows:
        lines += ["## Accuracy", "", _md_table(
            ["model", "p_ref", "estimate", "expected hits", "hits", "z", "verdict"],
            accuracy_rows,
        ), ""]
    if perf_rows:
        lines += ["## Performance", "", _md_table(
            ["model", "wall_s", "kernel_ms", "gpu_total_ms", "verdict"], perf_rows
        ), ""]
    if compile_rows:
        lines += ["## Compile / PDAG", "", _md_table(
            ["model", "compile_s", "tensors", "operators", "samples", "status"], compile_rows
        ), ""]
    return "\n".join(lines).rstrip() + "\n"


def summarize_ab(payload: Mapping[str, Any]) -> str:
    context = payload.get("context") or {}
    models = list(payload.get("models") or [])
    lines = [
        f"# A/B `{payload.get('dataset')}`",
        "",
        f"- verdict: **{payload.get('verdict', 'n/a')}**",
        f"- created: {payload.get('created_utc')}",
        f"- base: `{context.get('base_sha')}` image `{context.get('base_image')}`",
        f"- base digest: `{context.get('base_digest') or ''}`",
        f"- head: `{context.get('head_sha')}` image `{context.get('head_image')}`",
        f"- head digest: `{context.get('head_digest') or ''}`",
        f"- target: `{context.get('target')}` backend `{context.get('backend')}`",
    ]
    if context.get("metric") == "sample":
        seconds = context.get("seconds")
        size = f", ~{_fmt_float(seconds, 2)} s per side per round" if seconds else ""
        lines.append(
            f"- rounds: {context.get('rounds')}, base/head order alternating{size} "
            "(in-process `fuzzybits.run` time after a warm-up)"
        )
    else:
        lines.append(f"- rounds: {context.get('rounds')} (first dropped as warm-up)")
    if payload.get("attempts"):
        flagged = payload.get("flagged_ids") or []
        lines.append(
            f"- attempts: {payload.get('attempts')}; flagged: "
            + (", ".join(f"`{i}`" for i in flagged) if flagged else ("yes" if payload.get("flagged") else "none"))
        )
    note = shared_device_note(context if isinstance(context, Mapping) else None)
    if note:
        lines.append(f"- {note}")
    lines += [
        "",
        "## Ratios (head / base)",
        "",
    ]
    rows = []
    for model in models:
        metric = model.get("metric") or "primary"
        ci = model.get("ratio_ci") or [None, None]
        verdict = model.get("verdict")
        if model.get("measured_verdict") and model.get("measured_verdict") != verdict:
            verdict = f"{verdict} (measured {model.get('measured_verdict')})"
        rows.append(
            [
                model.get("id"),
                metric,
                model.get("base_trials") or "—",
                _fmt_float(model.get("ratio"), 4),
                _fmt_float(ci[0] if isinstance(ci, list) else None, 4),
                _fmt_float(ci[1] if isinstance(ci, list) else None, 4),
                _fmt_float(model.get("t"), 2),
                verdict,
            ]
        )
    if rows:
        lines.append(
            _md_table(
                ["model", "metric", "trials", "ratio", "ci_lo", "ci_hi", "t", "verdict"],
                rows,
            )
        )
        lines.append("")
    errors = [m for m in models if m.get("error")]
    for model in errors:
        lines.append(f"- `{model.get('id')}`: {str(model.get('error'))[:300]}")
    if errors:
        lines.append("")
    accuracy = [m for m in models if m.get("accuracy")]
    if accuracy:
        lines += ["## Accuracy (head)", ""]
        acc_rows = []
        for model in accuracy:
            acc = model["accuracy"]
            acc_rows.append(
                [
                    model.get("id"),
                    _fmt_float(acc.get("p_ref"), 8),
                    _fmt_float(acc.get("estimate"), 8),
                    _fmt_float(acc.get("z"), 3),
                    acc.get("verdict"),
                ]
            )
        lines.append(_md_table(["model", "p_ref", "estimate", "z", "verdict"], acc_rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarize(payload: Mapping[str, Any]) -> str:
    kind = payload.get("kind")
    if kind == "ab":
        return summarize_ab(payload)
    return summarize_run(payload)


def summarize_path(path: str | Path, *, markdown_out: str | Path | None = None) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    text = summarize(payload)
    if markdown_out is not None:
        Path(markdown_out).write_text(text, encoding="utf-8")
    return text
