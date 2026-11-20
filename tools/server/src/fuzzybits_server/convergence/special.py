"""Special functions for host-side Monte-Carlo convergence statistics.

This module is a small, dependency-free (stdlib ``math`` only) port of the
numerical helpers that previously lived in the native C++ engine
(``mc/stats/ci_utils.h``, ``mc/stats/info_gain.h``, ``mc/stats/diagnostics.h``)
and the TypeScript client (``lib/liveConvergenceStats.ts``).  Keeping these in
one audited place lets the policies/metrics layers stay terse and lets us unit
test the numerics in isolation.

All functions are pure and operate on Python floats.  Accuracy targets:

* ``normal_quantile_two_sided`` — Acklam (2003) rational approximation, max abs
  error ~5e-16 over (0, 1); identical coefficients to the C++/TS implementations
  so the three stacks agree to round-off.
"""

from __future__ import annotations

import math

# Guard used throughout to keep p strictly inside (0, 1) and avoid log/÷0 blowups.
DELTA_EPSILON: float = 1.0e-12

LN2: float = math.log(2.0)
LN10: float = math.log(10.0)
SQRT2: float = math.sqrt(2.0)

# Backwards-compatible private aliases.
_LN2 = LN2
_LN10 = LN10
_SQRT2 = SQRT2


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the closed interval ``[low, high]``."""
    return min(high, max(low, value))


# ---------------------------------------------------------------------------
#  Gaussian helpers
# ---------------------------------------------------------------------------
def normal_cdf(x: float) -> float:
    r"""Standard-normal CDF :math:`\Phi(x) = \tfrac12 \operatorname{erfc}(-x/\sqrt2)`."""
    return 0.5 * math.erfc(-x / _SQRT2)


def two_sided_p_value(z: float) -> float:
    r"""Two-sided p-value :math:`P(|Z| \ge |z|) = \operatorname{erfc}(|z|/\sqrt2)`."""
    return math.erfc(abs(z) / _SQRT2)


# Acklam (2003) inverse-normal-CDF rational approximation coefficients.
_ACKLAM_A = (
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
)
_ACKLAM_B = (
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
)


def normal_quantile_two_sided(confidence: float) -> float:
    r"""Two-sided normal quantile :math:`z = \Phi^{-1}(1 - \alpha/2)`.

    Given a two-sided confidence level ``confidence`` in (0, 1) returns the
    ``z`` such that ``P(|Z| <= z) = confidence`` for ``Z ~ N(0, 1)``.
    """
    p = clamp(confidence, DELTA_EPSILON, 1.0 - DELTA_EPSILON)
    alpha = 1.0 - p
    q = 1.0 - alpha / 2.0

    plow = 0.02425
    phigh = 1.0 - plow
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D

    if q < plow:
        u = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / (
            (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        )
    if q > phigh:
        u = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / (
            (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        )
    u = q - 0.5
    t = u * u
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * u / (
        ((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1.0
    )


def confidence_from_alpha(alpha: float) -> float:
    """Convert a two-sided significance ``alpha`` to a confidence level."""
    clamped = clamp(alpha, DELTA_EPSILON, 1.0 - DELTA_EPSILON) if math.isfinite(alpha) else 0.01
    return 1.0 - clamped


# ---------------------------------------------------------------------------
#  Beta / digamma helpers (information gain)
# ---------------------------------------------------------------------------
def digamma(x: float) -> float:
    r"""Digamma function :math:`\psi(x)` for ``x > 0`` (recurrence + asymptotic)."""
    value = x
    result = 0.0
    while value < 8.0:
        result -= 1.0 / value
        value += 1.0
    inv = 1.0 / value
    inv2 = inv * inv
    return (
        result
        + math.log(value)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
    )


def log_beta(a: float, b: float) -> float:
    r"""Natural log of the Beta function :math:`\ln B(a, b)`."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_entropy_bits(alpha: float, beta: float) -> float:
    r"""Shannon entropy of ``Beta(alpha, beta)`` in bits (``alpha, beta > 0``)."""
    nats = (
        log_beta(alpha, beta)
        - (alpha - 1.0) * digamma(alpha)
        - (beta - 1.0) * digamma(beta)
        + (alpha + beta - 2.0) * digamma(alpha + beta)
    )
    return nats / _LN2


def information_gain_bits(
    alpha_prev: float, beta_prev: float, alpha_new: float, beta_new: float
) -> float:
    """Entropy reduction (bits) moving from ``Beta(prev)`` to ``Beta(new)``."""
    return beta_entropy_bits(alpha_prev, beta_prev) - beta_entropy_bits(alpha_new, beta_new)


# ---------------------------------------------------------------------------
#  Bernoulli sampling-theory formulas (shared by policies and metrics)
# ---------------------------------------------------------------------------
def required_trials_two_sided(p: float, epsilon: float, z: float) -> int:
    r"""Wald sample size :math:`N \ge z^2 p(1-p)/\varepsilon^2` (linear scale)."""
    if epsilon <= 0.0:
        return 0
    return math.ceil((z * z * p * (1.0 - p)) / (epsilon * epsilon))


def required_trials_log10_two_sided(p: float, epsilon_log10: float, z: float) -> int:
    r"""Wald sample size for a target half-width on the :math:`\log_{10} p` scale."""
    if p <= 0.0 or epsilon_log10 <= 0.0:
        return 0
    denom = p * epsilon_log10 * epsilon_log10 * _LN10 * _LN10
    return math.ceil(z * z * (1.0 - p) / denom)


def epsilon_from_trials(p: float, n: int, z: float) -> float:
    r"""Achievable half-width :math:`\varepsilon = z\sqrt{p(1-p)/N}` for a fixed budget."""
    if n <= 0:
        return math.inf
    return z * (p * (1.0 - p) / n) ** 0.5


__all__ = [
    "DELTA_EPSILON",
    "clamp",
    "normal_cdf",
    "two_sided_p_value",
    "normal_quantile_two_sided",
    "confidence_from_alpha",
    "digamma",
    "log_beta",
    "beta_entropy_bits",
    "information_gain_bits",
    "required_trials_two_sided",
    "required_trials_log10_two_sided",
    "epsilon_from_trials",
]
