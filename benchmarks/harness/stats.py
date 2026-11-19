"""Accuracy z-scores, medians, IQR, and bootstrap CIs of a head/base ratio."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z95 = 1.959963984540054
Z_BLOCK = 4.0


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sample")
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def iqr(values: Sequence[float]) -> float:
    return percentile(values, 75.0) - percentile(values, 25.0)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sample")
    return sum(float(v) for v in values) / len(values)


def z_score(
    estimate: float,
    n: int,
    p_ref: float,
    p_ref_sd: float = 0.0,
) -> float:
    """Wald z = (p̂ − p_ref) / σ̂, combining reference uncertainty when given.

    σ̂² = p̂(1−p̂)/n + σ_ref². If p̂ is 0 or 1, the Bernoulli variance is 0 and
    the null variance p_ref(1−p_ref)/n is used so a zero-hit run against a
    tiny p_ref does not become inf.
    """
    if n <= 0:
        raise ValueError("z_score requires n > 0")
    if not math.isfinite(estimate) or not math.isfinite(p_ref):
        return float("nan")
    var_hat = max(estimate * (1.0 - estimate), 0.0) / float(n)
    if var_hat == 0.0:
        var_hat = max(p_ref * (1.0 - p_ref), 0.0) / float(n)
    sigma = math.sqrt(var_hat + max(p_ref_sd, 0.0) ** 2)
    if sigma == 0.0:
        return 0.0 if estimate == p_ref else math.copysign(float("inf"), estimate - p_ref)
    return (estimate - p_ref) / sigma


def accuracy_pass(z: float, z_max: float = Z_BLOCK) -> bool:
    return math.isfinite(z) and abs(z) <= z_max


def paired_ratios(base: Sequence[float], head: Sequence[float]) -> list[float]:
    if len(base) != len(head):
        raise ValueError(f"paired_ratios: length mismatch {len(base)} vs {len(head)}")
    out: list[float] = []
    for b, h in zip(base, head, strict=True):
        if b <= 0.0:
            raise ValueError("paired_ratios: base time must be positive")
        out.append(float(h) / float(b))
    return out


def bootstrap_median_ratio_ci(
    base: Sequence[float],
    head: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (median(head)/median(base), ci_lo, ci_hi) from independent resamples.

    The CI is of the ratio itself. Excluding 1 is equivalent to a CI of
    (ratio − 1) excluding 0, which is the significance test used by the gate.
    """
    if len(base) < 2 or len(head) < 2:
        raise ValueError("bootstrap_median_ratio_ci needs ≥ 2 samples per side")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    base_v = [float(v) for v in base]
    head_v = [float(v) for v in head]
    base_med = median(base_v)
    head_med = median(head_v)
    if base_med <= 0.0:
        raise ValueError(
            f"bootstrap_median_ratio_ci: median(base)={base_med} is not positive "
            f"(head median={head_med}); profiler likely missed the GPU process"
        )
    point = head_med / base_med
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(n_boot):
        b_s = [base_v[rng.randrange(len(base_v))] for _ in range(len(base_v))]
        h_s = [head_v[rng.randrange(len(head_v))] for _ in range(len(head_v))]
        b_m = median(b_s)
        if b_m <= 0.0:
            continue
        ratios.append(median(h_s) / b_m)
    if len(ratios) < max(100, n_boot // 2):
        raise ValueError("bootstrap produced too few finite ratios")
    lo = percentile(ratios, 100.0 * (alpha / 2.0))
    hi = percentile(ratios, 100.0 * (1.0 - alpha / 2.0))
    return point, lo, hi
