"""Parse nvprof (sm_70) and nsys (sm_86+) GPU kernel summaries from PR #91's methodology."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_DURATION_RE = re.compile(
    r"^([0-9]*\.?[0-9]+)\s*(ns|us|µs|ms|s)$", re.IGNORECASE
)
_NVPROF_ROW = re.compile(
    r"^\s*(?:GPU activities:\s+)?(?P<pct>[0-9.]+)%\s+"
    r"(?P<time>\S+)\s+(?P<calls>\d+)\s+"
    r"(?P<avg>\S+)\s+(?P<min>\S+)\s+(?P<max>\S+)\s+"
    r"(?P<name>.+?)\s*$"
)
_NSYS_BEGIN = re.compile(r"CUDA GPU Kernel Summary", re.IGNORECASE)


def parse_duration_ms(text: str) -> float:
    raw = text.strip().replace(",", "")
    match = _DURATION_RE.match(raw)
    if not match:
        return float(raw)
    value = float(match.group(1))
    unit = match.group(2).lower().replace("µ", "u")
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[unit]
    return value * scale


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "basic_event" in lowered or "pack_bernoulli" in lowered or "philox" in lowered:
        return "basic_event"
    if "tally" in lowered:
        return "tally"
    for gate in (
        "atleast",
        "nand",
        "nor",
        "xor",
        "xnor",
        "and",
        "or",
        "not",
        "null",
        "kn_",
        "k_n",
    ):
        if gate in lowered:
            return "gate"
    if "memcpy" in lowered or "memset" in lowered or "copy" in lowered:
        return "copy"
    return "other"


def _accumulate(buckets: dict[str, dict[str, float]], kind: str, time_ms: float, calls: int) -> None:
    slot = buckets.setdefault(kind, {"time_ms": 0.0, "calls": 0.0})
    slot["time_ms"] += time_ms
    slot["calls"] += float(calls)


def parse_nvprof(text: str) -> dict[str, Any]:
    """Parse ``nvprof --print-gpu-summary`` stdout+stderr."""
    kernels: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, float]] = {}
    in_gpu = False
    for line in text.splitlines():
        if "GPU activities:" in line:
            in_gpu = True
        elif in_gpu and line.strip().startswith("API calls:"):
            break
        if not in_gpu:
            continue
        match = _NVPROF_ROW.search(line)
        if not match:
            continue
        name = match.group("name").strip()
        time_ms = parse_duration_ms(match.group("time"))
        calls = int(match.group("calls"))
        kind = classify_kernel(name)
        kernels.append(
            {
                "name": name,
                "kind": kind,
                "time_ms": time_ms,
                "calls": calls,
                "avg_ms": parse_duration_ms(match.group("avg")),
            }
        )
        _accumulate(buckets, kind, time_ms, calls)
    return _summary(kernels, buckets, tool="nvprof")


def parse_nsys(text: str) -> dict[str, Any]:
    """Parse ``nsys profile --stats=true`` CUDA GPU Kernel Summary."""
    kernels: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, float]] = {}
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if _NSYS_BEGIN.search(line):
            start = i
            break
    # Header is the first line after the section title that contains "Name".
    body_start = start
    for i in range(start, len(lines)):
        if re.search(r"\bName\b", lines[i]) and i + 1 < len(lines):
            body_start = i + 1
            break
    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped:
            if kernels:
                break
            continue
        if stripped.startswith("==") or stripped.startswith("["):
            break
        if stripped.startswith("Time (%)") or stripped.startswith("---"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        try:
            total_ns = float(parts[1].replace(",", ""))
            instances = int(float(parts[2].replace(",", "")))
        except ValueError:
            continue
        name = " ".join(parts[7:])
        time_ms = total_ns / 1e6
        kind = classify_kernel(name)
        kernels.append(
            {
                "name": name,
                "kind": kind,
                "time_ms": time_ms,
                "calls": instances,
                "avg_ms": time_ms / instances if instances else time_ms,
            }
        )
        _accumulate(buckets, kind, time_ms, instances)
    return _summary(kernels, buckets, tool="nsys")


def _summary(
    kernels: list[dict[str, Any]],
    buckets: Mapping[str, Mapping[str, float]],
    *,
    tool: str,
) -> dict[str, Any]:
    compute = [
        row for row in kernels if row["kind"] not in {"copy"}
    ]
    total_ms = sum(row["time_ms"] for row in compute)
    basic = buckets.get("basic_event", {}).get("time_ms", 0.0)
    gates = buckets.get("gate", {}).get("time_ms", 0.0)
    tally = buckets.get("tally", {}).get("time_ms", 0.0)
    return {
        "tool": tool,
        "kernels": kernels,
        "by_kind_ms": {kind: dict(slot) for kind, slot in buckets.items()},
        "basic_event_ms": basic,
        "gate_ms": gates,
        "tally_ms": tally,
        "total_gpu_kernel_ms": total_ms,
        "gpu_total_ms": total_ms,
        "primary_ms": total_ms,
        "gpu_kernel_ms": basic,
    }


def parse_profiler(text: str, tool: str) -> dict[str, Any]:
    if tool == "nvprof":
        return parse_nvprof(text)
    if tool == "nsys":
        return parse_nsys(text)
    raise ValueError(f"unknown profiler {tool!r}")
