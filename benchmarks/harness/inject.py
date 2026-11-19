"""Parse ``--inject-regression`` (head-only slowdown for acceptance tests)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Any

# |round(N*F)/N - F| must be ≤ this fraction of F (e.g. 1.10 ± 1%).
SCALE_REL_TOL = 0.01
_MAX_PASSES_SEARCH = 100_000


@dataclass(frozen=True)
class InjectRegression:
    """Head-only perturbation. Base is never modified except to match pass counts."""

    extra_runs: int = 0
    sleep_ms: int = 0
    scale: float = 1.0
    spec: str | None = None

    @property
    def active(self) -> bool:
        return self.extra_runs > 0 or self.sleep_ms > 0 or self.scale != 1.0


def parse_inject_regression(spec: str | None) -> InjectRegression:
    if spec is None or spec == "":
        return InjectRegression()
    if spec == "repeat":
        return InjectRegression(extra_runs=1, spec=spec)
    if spec.startswith("sleep:"):
        rest = spec.split(":", 1)[1]
        try:
            sleep_ms = int(rest)
        except ValueError as exc:
            raise ValueError(f"inject-regression sleep:MS needs an int, got {spec!r}") from exc
        if sleep_ms < 0:
            raise ValueError("inject-regression sleep:MS must be ≥ 0")
        return InjectRegression(sleep_ms=sleep_ms, spec=spec)
    if spec.startswith("scale:"):
        rest = spec.split(":", 1)[1]
        try:
            scale = float(rest)
        except ValueError as exc:
            raise ValueError(f"inject-regression scale:N needs a float, got {spec!r}") from exc
        if not (scale > 0.0):
            raise ValueError("inject-regression scale:N must be > 0")
        return InjectRegression(scale=scale, spec=spec)
    raise ValueError(
        f"unknown --inject-regression {spec!r}; expected repeat, sleep:MS, or scale:N"
    )


# One uint64 bitpack. A 1-trial ``fzb run --json`` reports n=64; that is not
# the device pass used at perf trial counts (chinese V100S is 32·2^20).
BITPACK_BITS = 64


def _positive_pass_size(raw: Any, *, where: str) -> int | None:
    if raw is None or raw == "":
        return None
    n = int(raw)
    if n <= 0:
        return None
    if n <= BITPACK_BITS:
        raise ValueError(
            f"{where}={n} is a bitpack, not the device pass at the configured "
            "trial count; open Session(num_trials=configured) and read "
            "sample_shape.trials_per_pass"
        )
    return n


def pass_size_from_run(run: Mapping[str, Any], *, requested_trials: int | None = None) -> int:
    """Device pass size from ``sample_shape.trials_per_pass`` or ``fzb run --json``.

    Do not assume 2^20 and do not use a 1-trial ``n`` (that is one bitpack).
    Probe with ``Session(..., num_trials=configured_trials).sample_shape``.
    Fallback: ``n(T+1) - n(T)`` from two ``fzb run --json`` calls.
    """
    parsed = run.get("run") or {}
    shape = parsed.get("sample_shape") or run.get("sample_shape") or {}
    for raw, where in (
        (shape.get("trials_per_pass"), "sample_shape.trials_per_pass"),
        (shape.get("realized_trials_per_pass"), "sample_shape.realized_trials_per_pass"),
        (parsed.get("trials_per_pass"), "trials_per_pass"),
    ):
        n = _positive_pass_size(raw, where=where)
        if n is not None:
            return n
    queries = parsed.get("queries") or []
    if queries:
        n = _positive_pass_size(queries[0].get("n"), where="queries[0].n")
        if n is not None:
            return n
    raise ValueError("cannot read device pass size from sample_shape / fzb run JSON")


def pass_size_from_n_pair(n_at: int, n_next: int) -> int:
    """``n(T+1) - n(T)`` when ``T`` is already a multiple of the device pass."""
    delta = int(n_next) - int(n_at)
    n = _positive_pass_size(delta, where="n(T+1)-n(T)")
    if n is None:
        raise ValueError(f"pass size not visible: n={n_at} n_next={n_next} delta={delta}")
    return n


def choose_pass_counts(factor: float, *, min_passes: int = 1) -> tuple[int, int]:
    """Return ``(n_base, n_head)`` device-pass counts realizing ``factor``.

    ``n_head = round(n_base * factor)`` and ``n_head / n_base`` is within 1% of
    ``factor``. Examples: 1.10 → (10, 11); 1.05 → (20, 21). ``min_passes`` is
    the configured work in passes so the injection is not shorter than a
    normal A/B sample.
    """
    if factor <= 0.0:
        raise ValueError(f"scale factor must be > 0, got {factor}")
    start = max(1, int(min_passes))
    frac = Fraction(factor).limit_denominator(100)
    n0, m0 = frac.denominator, frac.numerator
    if n0 > 0 and abs(m0 / n0 - factor) <= SCALE_REL_TOL * factor:
        k = max(1, ceil(start / n0))
        n_base, n_head = n0 * k, m0 * k
        if n_base >= start and abs(n_head / n_base - factor) <= SCALE_REL_TOL * factor:
            return n_base, n_head
    limit = start + _MAX_PASSES_SEARCH
    for n_base in range(start, limit):
        n_head = max(1, int(round(n_base * float(factor))))
        realized = n_head / n_base
        if abs(realized - factor) <= SCALE_REL_TOL * factor:
            return n_base, n_head
    raise ValueError(
        f"no pass counts for scale {factor} with min_passes={start} "
        f"within {SCALE_REL_TOL:.0%} in {_MAX_PASSES_SEARCH} candidates"
    )


def calibrated_trials(
    pass_size: int, factor: float, *, min_passes: int = 1
) -> tuple[int, int, int, int]:
    """``(base_trials, head_trials, n_base, n_head)`` at the engine pass size."""
    if int(pass_size) <= 0:
        raise ValueError(f"pass_size must be > 0, got {pass_size}")
    n_base, n_head = choose_pass_counts(factor, min_passes=min_passes)
    return n_base * int(pass_size), n_head * int(pass_size), n_base, n_head
