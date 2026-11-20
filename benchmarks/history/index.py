"""Build and publish the static history index consumed by the Benchmarks tab."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..harness.verdict import combine_verdicts

SCHEMA_VERSION = 1
DATASET_DEFAULT = "v1"
INDEX_REL = "bench/v1/index.json"
RESULTS_REL = "bench/v1/results"
DEFAULT_PAGES_INDEX_URL = (
    "https://fuzzybit-org.github.io/fuzzybits/bench/v1/index.json"
)
_ISO_UTC = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z$"
)
_SEMVER_TAG = re.compile(r"^v\d+\.\d+\.\d+")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_name(created_utc: str) -> str:
    """Filesystem-safe UTC stamp. ``2026-11-19T23:01:00Z`` -> ``2026-11-19T23-01-00Z``."""
    raw = (created_utc or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "Z"
    return raw.replace(":", "-")


def channel_for_version(version: str) -> str:
    if _SEMVER_TAG.match(version or ""):
        return "release"
    return "develop"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _perf_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    profiler = row.get("profiler") or {}
    sample_s = row.get("sample_s")
    if sample_s is None:
        sample_s = (row.get("run") or {}).get("sample_s")
    gpu_total = profiler.get("gpu_total_ms")
    if gpu_total is None:
        gpu_total = profiler.get("total_gpu_kernel_ms")
    return {
        "gpu_total_ms": _finite(gpu_total),
        "gpu_kernel_ms": _finite(
            profiler.get("gpu_kernel_ms")
            if profiler.get("gpu_kernel_ms") is not None
            else profiler.get("basic_event_ms")
        ),
        "cpu_sample_s": _finite(sample_s),
        "wall_s": _finite(row.get("wall_s")),
        "verdict": row.get("verdict"),
        "status": row.get("status"),
    }


def _accuracy_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "z": _finite(row.get("z")),
        "verdict": row.get("verdict"),
        "status": row.get("status"),
        "estimate": _finite(row.get("estimate")),
        "n": row.get("n"),
        "hits": row.get("hits"),
        "expected_hits": _finite(row.get("expected_hits")),
        "p_ref": _finite(row.get("p_ref")),
    }


def model_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse accuracy + perf rows for the same model id."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        model_id = str(row.get("id") or "")
        if not model_id:
            continue
        if model_id not in by_id:
            by_id[model_id] = {"id": model_id}
            order.append(model_id)
        slot = by_id[model_id]
        mode = row.get("mode")
        if mode == "accuracy":
            slot["accuracy"] = _accuracy_metrics(row)
        elif mode == "perf":
            slot["perf"] = _perf_metrics(row)
        elif mode == "compile":
            slot["compile"] = {
                "verdict": row.get("verdict"),
                "status": row.get("status"),
                "compile_wall_s": _finite(row.get("compile_wall_s") or row.get("wall_s")),
            }
    return [by_id[key] for key in order]


def merge_runs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    version: str,
    driver: str | None = None,
    runner_name: str | None = None,
) -> dict[str, Any]:
    """Combine accuracy/perf (and optional compile) harness payloads.

    Rows are concatenated. Context is taken from the last payload that has
    one, then overlaid with ``version`` / ``channel`` / driver / runner.
    """
    if not payloads:
        raise ValueError("merge_runs requires at least one payload")
    models: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    dataset = DATASET_DEFAULT
    created = _now()
    kinds: list[str] = []
    for payload in payloads:
        if payload.get("models"):
            models.extend(list(payload["models"]))
        if payload.get("context"):
            context = dict(payload["context"])
        if payload.get("dataset"):
            dataset = str(payload["dataset"])
        if payload.get("created_utc"):
            created = str(payload["created_utc"])
        if payload.get("kind"):
            kinds.append(str(payload["kind"]))
    context = dict(context)
    context["version"] = version
    context["channel"] = channel_for_version(version)
    if driver:
        context["driver"] = driver
    if runner_name:
        context["runner_name"] = runner_name
        context.setdefault("runner", runner_name)
    verdicts = [str(row.get("verdict") or "pass") for row in models]
    if not verdicts:
        overall = str(payloads[-1].get("verdict") or "error")
    else:
        overall = combine_verdicts(verdicts)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run",
        "dataset": dataset,
        "created_utc": created,
        "verdict": overall,
        "context": context,
        "models": models,
        "summary": {"sources": kinds, "n_models": len(models)},
    }


def result_relpath(version: str, target: str, created_utc: str) -> str:
    return f"results/{version}/{target}/{stamp_name(created_utc)}.json"


def _point_from_payload(
    payload: Mapping[str, Any],
    *,
    relpath: str,
) -> dict[str, Any]:
    context = dict(payload.get("context") or {})
    version = str(context.get("version") or "")
    target = str(context.get("target") or "")
    if not version or not target:
        raise ValueError(f"{relpath}: context.version and context.target are required")
    return {
        "version": version,
        "channel": context.get("channel") or channel_for_version(version),
        "target": target,
        "dataset": payload.get("dataset") or DATASET_DEFAULT,
        "created_utc": payload.get("created_utc"),
        "path": relpath,
        "runner": context.get("runner") or context.get("runner_name"),
        "runner_name": context.get("runner_name") or context.get("runner"),
        "device": context.get("device"),
        "driver": context.get("driver"),
        "image": context.get("image"),
        "image_tag": context.get("image_tag"),
        "image_digest": context.get("image_digest"),
        "image_pin_source": context.get("image_pin_source"),
        "recipe": context.get("recipe"),
        "commit": context.get("commit"),
        "backend": context.get("backend"),
        "verdict": payload.get("verdict"),
        "flags": {
            "recipe_changed": False,
            "driver_changed": False,
            "runner_changed": False,
            "image_changed": False,
        },
        "models": model_summaries(list(payload.get("models") or [])),
    }


def _sort_key(point: Mapping[str, Any]) -> tuple:
    return (
        str(point.get("target") or ""),
        str(point.get("created_utc") or ""),
        str(point.get("version") or ""),
    )


def flag_changes(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag recipe / driver / runner / image digest changes vs the previous point on that target."""
    previous: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []
    for point in sorted(points, key=_sort_key):
        target = str(point.get("target") or "")
        prev = previous.get(target)
        flags = {
            "recipe_changed": False,
            "driver_changed": False,
            "runner_changed": False,
            "image_changed": False,
        }
        if prev is not None:
            flags["recipe_changed"] = (point.get("recipe") or "") != (prev.get("recipe") or "")
            flags["driver_changed"] = (point.get("driver") or "") != (prev.get("driver") or "")
            flags["runner_changed"] = (point.get("runner") or "") != (prev.get("runner") or "")
            flags["image_changed"] = (
                (point.get("image_digest") or "") != (prev.get("image_digest") or "")
            )
        out = dict(point)
        out["flags"] = flags
        flagged.append(out)
        previous[target] = point
    return flagged


def iter_result_files(root: Path) -> list[Path]:
    results = root / RESULTS_REL
    if not results.is_dir():
        return []
    return sorted(path for path in results.rglob("*.json") if path.is_file())


def build_index(
    root: Path,
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Walk ``bench/v1/results`` under ``root`` and emit the Pages index."""
    points: list[dict[str, Any]] = []
    base = (root / "bench" / "v1").resolve()
    for path in iter_result_files(root):
        payload = _read_json(path)
        relpath = path.resolve().relative_to(base).as_posix()
        points.append(_point_from_payload(payload, relpath=relpath))
    points = flag_changes(points)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_DEFAULT,
        "generated_utc": generated_utc or _now(),
        "source_url": DEFAULT_PAGES_INDEX_URL,
        "n_points": len(points),
        "points": points,
    }


def unique_result_path(dest: Path, version: str, target: str, created_utc: str) -> Path:
    """Never overwrite. A collision gets ``-2``, ``-3``, … before ``.json``."""
    directory = dest / RESULTS_REL / version / target
    stamp = stamp_name(created_utc)
    candidate = directory / f"{stamp}.json"
    if not candidate.exists():
        return candidate
    for suffix in range(2, 1000):
        extra = directory / f"{stamp}-{suffix}.json"
        if not extra.exists():
            return extra
    raise RuntimeError(f"too many reruns for {version}/{target}/{stamp}")


def publish_payload(
    payload: Mapping[str, Any],
    *,
    dest: Path,
    version: str | None = None,
) -> dict[str, Any]:
    """Write one results JSON (never rewritten) and rebuild the index.

    Returns the written point's relative path and the new index.
    """
    context = dict(payload.get("context") or {})
    version = version or str(context.get("version") or "")
    target = str(context.get("target") or "")
    if not version:
        raise ValueError("publish_payload: version is required")
    if not target:
        raise ValueError("publish_payload: context.target is required")
    created = str(payload.get("created_utc") or _now())
    merged = dict(payload)
    merged_context = dict(context)
    merged_context["version"] = version
    merged_context["channel"] = channel_for_version(version)
    merged["context"] = merged_context
    merged["created_utc"] = created
    dest_file = unique_result_path(dest, version, target, created)
    _write_json(dest_file, merged)
    index = build_index(dest)
    _write_json(dest / INDEX_REL, index)
    relpath = dest_file.resolve().relative_to((dest / "bench" / "v1").resolve()).as_posix()
    return {"path": relpath, "file": str(dest_file), "index": index}


def compare_ratio(head: float | None, base: float | None) -> float | None:
    if head is None or base is None or base == 0:
        return None
    return head / base


def latest_points(
    points: Iterable[Mapping[str, Any]],
    *,
    target: str,
) -> dict[str, Mapping[str, Any] | None]:
    """Newest release and newest develop point for ``target``."""
    release: Mapping[str, Any] | None = None
    develop: Mapping[str, Any] | None = None
    for point in sorted(points, key=lambda p: str(p.get("created_utc") or "")):
        if point.get("target") != target:
            continue
        if point.get("channel") == "release":
            release = point
        elif point.get("channel") == "develop":
            develop = point
    return {"release": release, "develop": develop}
