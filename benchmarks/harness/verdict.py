"""A/B and accuracy verdicts (#4 / #121)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .stats import Z_BLOCK, accuracy_pass
from ..loader import INFEASIBLE_HITS, LOW_POWER_HITS, SAMPLER_MIN_P, expected_hits

GPU_BLOCK = 0.03
CPU_BLOCK = 0.05
# Set False if A/A CI half-width cannot get under ~2%; then CPU is warn-only.
CPU_GATE_BLOCKS = False
VERDICT_PASS = "pass"
VERDICT_PASS_LOW_POWER = "pass (low power)"
VERDICT_INFEASIBLE = "infeasible"
VERDICT_KNOWN_FAILURE = "known_failure"
VERDICT_TIMEOUT = "timeout"
VERDICT_WARN = "warn"
VERDICT_BLOCK = "block"
VERDICT_ERROR = "error"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_UNSUPPORTED = "unsupported"
VERDICTS = (
    VERDICT_PASS,
    VERDICT_PASS_LOW_POWER,
    VERDICT_INFEASIBLE,
    VERDICT_KNOWN_FAILURE,
    VERDICT_UNSUPPORTED,
    VERDICT_TIMEOUT,
    VERDICT_WARN,
    VERDICT_BLOCK,
    VERDICT_ERROR,
    VERDICT_INCONCLUSIVE,
)

_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda_error_out_of_memory",
    "cudaerroroutofmemory",
    "std::bad_alloc",
)
_UNSUPPORTED_MARKERS = (
    "no kernel image is available for execution on the device",
    "invalid ptx",
    "unsupported ptx",
    "ptxas fatal",
    "ptxas error",
    "jit compilation failed",
    "sscp compilation",
    "cannot find ptxas",
    "nvptx-arch",
    "fatal error: error in backend",
    "adaptivecpp error",
)


def ratio_verdict(
    ratio: float,
    ci_lo: float,
    ci_hi: float,
    *,
    block_slowdown: float,
) -> str:
    """Classify a head/base ratio.

    * **block** only when the CI lower bound exceeds ``1 + block_slowdown``
      (we are confident the slowdown is above the threshold).
    * **warn** if the CI excludes 1 but the lower bound is still below the
      threshold.
    * **pass** otherwise, including a large median that is not significant.
    """
    if not (ci_lo <= ratio <= ci_hi):
        # Point estimate should sit in the CI; if a bootstrap is degenerate, don't block.
        return "inconclusive"
    if ci_lo > 1.0 + block_slowdown:
        return "block"
    if ci_lo > 1.0:
        return "warn"
    return "pass"


def gpu_kernel_verdict(ratio: float, ci_lo: float, ci_hi: float) -> str:
    return ratio_verdict(ratio, ci_lo, ci_hi, block_slowdown=GPU_BLOCK)


def gpu_total_verdict(ratio: float, ci_lo: float, ci_hi: float) -> str:
    """Same 3% CI-lo rule as the kernel gate; applied to all compute kernels."""
    return gpu_kernel_verdict(ratio, ci_lo, ci_hi)


def cpu_wall_verdict(ratio: float, ci_lo: float, ci_hi: float) -> str:
    verdict = ratio_verdict(ratio, ci_lo, ci_hi, block_slowdown=CPU_BLOCK)
    if verdict == VERDICT_BLOCK and not CPU_GATE_BLOCKS:
        return VERDICT_WARN
    return verdict


def apply_report_only(verdict: str, *, report_only: bool) -> str:
    """Measured ``block`` becomes ``warn`` on report-only targets."""
    if report_only and verdict == VERDICT_BLOCK:
        return VERDICT_WARN
    return verdict


def classify_run_failure(*texts: Any) -> str:
    """Map compile/run stderr to ``error`` (OOM), ``unsupported`` (JIT/PTX), or ``block``."""
    blob = " ".join(str(text or "") for text in texts).lower()
    if any(marker in blob for marker in _OOM_MARKERS):
        return VERDICT_ERROR
    if any(marker in blob for marker in _UNSUPPORTED_MARKERS):
        return VERDICT_UNSUPPORTED
    return VERDICT_BLOCK


def accuracy_verdict(z: float, *, z_max: float = Z_BLOCK) -> str:
    return VERDICT_PASS if accuracy_pass(z, z_max=z_max) else VERDICT_BLOCK


def accuracy_power_verdict(
    *,
    p_ref: float,
    n: int,
    hits: int,
    z: float,
    infeasible_hits: float = INFEASIBLE_HITS,
    low_power_hits: float = LOW_POWER_HITS,
    z_max: float = Z_BLOCK,
) -> dict[str, Any]:
    """Classify one accuracy run. Power is judged before z.

    * ``infeasible`` if p_ref < 2^-32 (#17) or n·p_ref < ``infeasible_hits``.
      Never a pass; z is omitted when hits=0 (no evidence).
    * ``pass (low power)`` if infeasible_hits ≤ n·p_ref < ``low_power_hits``
      and |z| ≤ 4.
    * ``pass`` / ``block`` from |z| otherwise.
    """
    expected = expected_hits(n, p_ref)
    below_sampler = p_ref < SAMPLER_MIN_P
    if below_sampler or expected < infeasible_hits:
        reasons = []
        if below_sampler:
            reasons.append(
                f"p_ref {p_ref:.6g} < 2^-32 sampler threshold (~{SAMPLER_MIN_P:.3g}, #17)"
            )
        if expected < infeasible_hits:
            reasons.append(
                f"expected hits n·p_ref = {expected:.3g} < {infeasible_hits:g}"
            )
        return {
            "verdict": VERDICT_INFEASIBLE,
            "expected_hits": expected,
            "z": None if hits == 0 else z,
            "gated": False,
            "reason": "; ".join(reasons),
        }
    gated_verdict = accuracy_verdict(z, z_max=z_max)
    if expected < low_power_hits and gated_verdict == VERDICT_PASS:
        gated_verdict = VERDICT_PASS_LOW_POWER
    return {
        "verdict": gated_verdict,
        "expected_hits": expected,
        "z": z,
        "gated": True,
        "reason": None,
    }


def ab_timeout_verdict(*, base_timeout: bool, head_timeout: bool) -> str:
    """Head newly timing out (base did not) is a regression."""
    if head_timeout and not base_timeout:
        return VERDICT_BLOCK
    if head_timeout or base_timeout:
        return VERDICT_TIMEOUT
    return VERDICT_PASS


_RANK = {
    VERDICT_PASS: 0,
    VERDICT_PASS_LOW_POWER: 0,
    VERDICT_INFEASIBLE: 1,
    VERDICT_KNOWN_FAILURE: 1,
    VERDICT_UNSUPPORTED: 1,
    VERDICT_TIMEOUT: 2,
    VERDICT_WARN: 3,
    VERDICT_BLOCK: 4,
    VERDICT_ERROR: 5,
    VERDICT_INCONCLUSIVE: 6,
}

_VISIBLE_NOT_GATE = {VERDICT_INFEASIBLE, VERDICT_KNOWN_FAILURE, VERDICT_UNSUPPORTED}


def combine_verdicts(verdicts: Iterable[str]) -> str:
    """Worst of the per-model verdicts.

    ``infeasible`` and ``known_failure`` are visible but not a z-test / A/B
    failure and do not count as a pass. ``error`` is a harness failure (missing
    or zero timings): never pass, never block. ``inconclusive`` dominates so
    host noise is not a regression.
    """
    worst = VERDICT_PASS
    saw_gated_pass = False
    saw_any = False
    for item in verdicts:
        if item not in _RANK:
            raise ValueError(f"unknown verdict {item!r}")
        saw_any = True
        if item in {VERDICT_PASS, VERDICT_PASS_LOW_POWER}:
            saw_gated_pass = True
        if _RANK[item] > _RANK[worst]:
            worst = item
    if worst in _VISIBLE_NOT_GATE and saw_gated_pass:
        return VERDICT_PASS
    if worst in _VISIBLE_NOT_GATE and not saw_gated_pass:
        return worst
    if not saw_any:
        return VERDICT_PASS
    return worst


def finite_positive(value: Any) -> tuple[float | None, str | None]:
    """Return ``(number, None)`` or ``(None, reason)`` for a missing/zero metric.

    A missing or non-positive timing is a harness ``error``, never a crash and
    never a ``block`` / ``pass``.
    """
    if value is None:
        return None, "missing metric"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"non-numeric metric {value!r}"
    if not math.isfinite(number):
        return None, f"non-finite metric {number}"
    if number <= 0.0:
        return None, f"non-positive metric {number}"
    return number, None


def collect_timings(
    runs: Sequence[Mapping[str, Any]],
    *,
    gpu: bool,
    kind: str = "primary",
) -> tuple[list[float] | None, str | None]:
    """Extract A/B samples. ``kind`` is ``primary``, ``total``, or ``kernel``.

    GPU ``primary`` / ``total`` is ``total_gpu_kernel_ms`` (all compute kernels,
    memcpy excluded). GPU ``kernel`` is ``basic_event_ms``. CPU is in-process
    ``sample_s``. Returns ``(None, reason)`` if any sample is missing or ≤ 0.
    """
    if not runs:
        return None, "no timed runs"
    values: list[float] = []
    for index, run in enumerate(runs):
        if gpu:
            profiler = run.get("profiler") or {}
            if kind == "kernel":
                raw = profiler.get("basic_event_ms")
                label = "gpu_kernel_ms"
            else:
                raw = profiler.get("total_gpu_kernel_ms")
                if raw is None:
                    raw = profiler.get("gpu_total_ms")
                if raw is None:
                    raw = profiler.get("primary_ms")
                if raw is None:
                    # No nsys/nvprof: use fzb sample_s (or container wall) in ms.
                    sample = run.get("sample_s")
                    if sample is None:
                        sample = (run.get("run") or {}).get("sample_s")
                    if sample is None:
                        sample = run.get("wall_s")
                    number, _reason = finite_positive(sample)
                    if number is not None:
                        raw = number * 1000.0
                label = "gpu_total_ms"
        else:
            raw = run.get("sample_s")
            if raw is None:
                raw = (run.get("run") or {}).get("sample_s")
            label = "cpu_sample_s"
        number, reason = finite_positive(raw)
        if reason:
            return None, f"{label} run {index}: {reason}"
        assert number is not None
        values.append(number)
    return values, None


def overall_verdict(rows: Iterable[Mapping[str, Any]]) -> str:
    return combine_verdicts(str(row.get("verdict") or "pass") for row in rows)
