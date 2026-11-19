r"""Huber's Gamma Bernoulli Approximation Scheme (GBAS, 2017).

GBAS estimates a Bernoulli mean :math:`p` with a relative error whose *exact*
distribution is known and **independent of** :math:`p`.  Concretely, fix a target
number of successes :math:`k`; draw Bernoulli(:math:`p`) trials, attaching to each
an independent ``Exp(1)`` clock, until the :math:`k`-th success; let
:math:`R=\sum A_i` be the accumulated clock.  Then

.. math::

    R \sim \mathrm{Gamma}(k, p), \qquad
    \hat p_{\mathrm{GBAS}} = \frac{k-1}{R}, \qquad
    \frac{p}{\hat p_{\mathrm{GBAS}}} \sim \mathrm{Gamma}(k, k-1).

The estimator is unbiased (:math:`\mathbb E[\hat p]=p` for :math:`k\ge 2`) and has
relative standard deviation :math:`1/\sqrt{k-2}` regardless of :math:`p`.  Because
the relative-error law is known exactly, :math:`k` can be sized for an
:math:`(\varepsilon,\delta)`-relative-approximation guarantee
(:math:`\Pr(|\hat p/p-1|>\varepsilon)\le\delta`) by a pure numerical search on the
``Gamma`` CDF -- no normal approximation -- and an *exact* confidence interval is
available at the stopping time.

Reference: M. Huber, "A Bernoulli mean estimate with known relative error
distribution," *Random Structures & Algorithms* 50:173-182, 2017
(arXiv:1309.5413).

Batched device adaptation
-------------------------
The native sampler reports only *cumulative* counters per pass, so the :math:`k`-th
success lands somewhere inside the batch that first pushes the success count to
:math:`\ge k` -- we never observe its exact trial index :math:`N^\*`.  Conditioned
on a batch having :math:`s` successes among :math:`m` trials the success positions
are a uniformly random :math:`s`-subset, so the index of the :math:`j`-th success is
the negative-hypergeometric order statistic, sampled *exactly* via the
Beta-Binomial identity :math:`F\sim\mathrm{BetaBinom}(m-s;\,j,\,s-j+1)` (draw
:math:`\theta\sim\mathrm{Beta}(j,s-j+1)`, then :math:`F\sim\mathrm{Binom}(m-s,
\theta)`).  Given the reconstructed :math:`N^\*`, the per-trial ``Exp(1)`` clocks
are independent of the Bernoulli outcomes, so :math:`R\mid N^\*\sim
\mathrm{Gamma}(N^\*,1)` and drawing a fresh ``Gamma`` is distributionally identical
to having accumulated the clocks online.  Marginalizing over :math:`N^\*` (which is
``NegativeBinomial(k,p)``) recovers :math:`R\sim\mathrm{Gamma}(k,p)` exactly, so the
known relative-error law is preserved despite batching.

This module is intentionally the only place in the engine that depends on
``numpy``/``scipy``; the policy layer imports it lazily so the default
(Wald/Bayes) path stays dependency-light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc, gammaincc, gammaincinv, ndtri

from .special import DELTA_EPSILON

__all__ = [
    "K_MIN",
    "relative_error_tail",
    "solve_k",
    "relative_sd",
    "relative_interval",
    "confidence_interval",
    "GBASEstimate",
    "estimate_at_crossing",
]

# Need k > 2 for a finite relative variance 1/(k-2); k >= 2 for unbiasedness.
K_MIN: int = 3
# Generous cap so a degenerate (epsilon, delta) cannot spin the search forever.
_K_MAX: int = 1 << 42


def _clamp_unit(value: float) -> float:
    return min(max(value, DELTA_EPSILON), 1.0 - DELTA_EPSILON)


def relative_error_tail(k: float, epsilon: float) -> float:
    r"""Two-sided failure probability of GBAS(``k``) at relative tolerance ``epsilon``.

    Returns :math:`\Pr(|\hat p/p - 1| > \varepsilon)`.  With the relative error
    :math:`X=p/\hat p\sim\mathrm{Gamma}(k,k-1)` and :math:`W=(k-1)X\sim
    \mathrm{Gamma}(k,1)`,

    .. math::

        \Pr\!\Big(X<\tfrac1{1+\varepsilon}\Big) + \Pr\!\Big(X>\tfrac1{1-\varepsilon}\Big)
        = P\!\big(k, \tfrac{k-1}{1+\varepsilon}\big) + Q\!\big(k, \tfrac{k-1}{1-\varepsilon}\big),

    where :math:`P,Q` are the regularized lower/upper incomplete gamma functions.
    """
    eps = _clamp_unit(epsilon)
    lower = gammainc(k, (k - 1.0) / (1.0 + eps))   # P(W < (k-1)/(1+eps))
    upper = gammaincc(k, (k - 1.0) / (1.0 - eps))  # P(W > (k-1)/(1-eps))
    return float(lower + upper)


def solve_k(epsilon: float, delta_fail: float) -> int:
    r"""Smallest integer ``k >= K_MIN`` making GBAS an :math:`(\varepsilon,\delta)`-ras.

    The tail :func:`relative_error_tail` is monotone decreasing in ``k``; we seed a
    bracket from the normal approximation :math:`k\approx (z_{\delta/2}/\varepsilon)^2`,
    gallop until the tail drops below ``delta_fail``, then binary-search the exact
    minimal ``k`` (Huber's recommended logarithmic-overhead search).
    """
    eps = _clamp_unit(epsilon)
    df = _clamp_unit(delta_fail)

    if relative_error_tail(K_MIN, eps) <= df:
        return K_MIN

    z = float(ndtri(1.0 - df / 2.0))
    hi = max(K_MIN + 1, int((z / eps) ** 2) + 2)
    while relative_error_tail(hi, eps) > df:
        hi *= 2
        if hi >= _K_MAX:
            return _K_MAX

    lo = K_MIN  # tail(lo) > df (checked above)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if relative_error_tail(mid, eps) <= df:
            hi = mid
        else:
            lo = mid
    return hi


def relative_sd(k: float) -> float:
    r"""Relative standard deviation of the GBAS estimate, :math:`1/\sqrt{k-2}`.

    Defined for ``k > 2``; returns ``+inf`` otherwise (too few successes to bound
    the relative variance).
    """
    return 1.0 / math.sqrt(k - 2.0) if k > 2.0 else math.inf


def confidence_interval(k: int, gamma_r: float, ci_alpha: float) -> tuple[float, float]:
    r"""Exact central ``(1 - ci_alpha)`` confidence interval for :math:`p`.

    Pivots on :math:`p = W/R` with :math:`W\sim\mathrm{Gamma}(k,1)` and the realized
    :math:`R=` ``gamma_r``: :math:`[\,G^{-1}_k(a/2)/R,\ G^{-1}_k(1-a/2)/R\,]`.  Unlike
    :func:`relative_interval` (a data-independent *relative* band scaled by the point
    estimate), this is the actual interval on :math:`p` at the stopping time.
    """
    a = _clamp_unit(ci_alpha)
    lo = float(gammaincinv(k, a / 2.0)) / gamma_r
    hi = float(gammaincinv(k, 1.0 - a / 2.0)) / gamma_r
    return lo, hi


def relative_interval(k: int, ci_alpha: float) -> tuple[float, float]:
    r"""Central ``(1 - ci_alpha)`` interval for the relative error :math:`X=p/\hat p`.

    :math:`X\sim\mathrm{Gamma}(k,k-1)`, i.e. :math:`X=W/(k-1)` with
    :math:`W\sim\mathrm{Gamma}(k,1)`; the interval brackets ``1`` and is independent
    of the data (a property of GBAS).
    """
    a = _clamp_unit(ci_alpha)
    lo = float(gammaincinv(k, a / 2.0)) / (k - 1.0)
    hi = float(gammaincinv(k, 1.0 - a / 2.0)) / (k - 1.0)
    return lo, hi


@dataclass(frozen=True, slots=True)
class GBASEstimate:
    """The GBAS estimate materialized at the stopping time (``successes >= k``)."""

    k: int
    n_star: int       # reconstructed trial index of the k-th success
    gamma_r: float    # R ~ Gamma(n_star, 1) == Gamma(k, p) marginally
    p_hat: float      # (k-1)/R: unbiased, relative error ~ Gamma(k, k-1)
    ci_low: float     # exact lower confidence bound on p
    ci_high: float    # exact upper confidence bound on p
    rel_low: float    # relative-error lower quantile (data-independent)
    rel_high: float   # relative-error upper quantile (data-independent)


def estimate_at_crossing(
    *,
    k: int,
    prev_total: int,
    prev_ones: int,
    cur_total: int,
    cur_ones: int,
    ci_alpha: float,
    rng: np.random.Generator,
) -> GBASEstimate:
    r"""Reconstruct :math:`N^\*` within the crossing batch and draw the GBAS estimate.

    ``prev_*`` are the cumulative counters at the last pre-crossing pass and
    ``cur_*`` the counters at the first pass with ``cur_ones >= k``.  The exact CI
    pivots on :math:`p=W/R` with :math:`W\sim\mathrm{Gamma}(k,1)`:
    :math:`[\,G^{-1}_{k}(a/2)/R,\ G^{-1}_{k}(1-a/2)/R\,]`.
    """
    m = cur_total - prev_total   # trials in the crossing batch
    s = cur_ones - prev_ones     # successes in the crossing batch
    j = k - prev_ones            # the k-th overall success is the j-th of this batch

    if s <= 0 or j < 1 or j > s:
        # Not a well-formed first crossing (e.g. policy attached mid-run); treat the
        # whole accumulation as one batch from a zero baseline.
        n_star = max(k, cur_total)
    else:
        theta = float(rng.beta(j, s - j + 1))         # Beta(j, s-j+1) order-stat fraction
        failures_before = int(rng.binomial(m - s, theta))
        position = j + failures_before                 # 1-indexed position within batch
        n_star = prev_total + position
    n_star = max(int(n_star), k)

    gamma_r = float(rng.standard_gamma(n_star))         # Gamma(n_star, 1) == Gamma(k, p)
    p_hat = (k - 1.0) / gamma_r

    a = _clamp_unit(ci_alpha)
    ci_low, ci_high = confidence_interval(k, gamma_r, a)
    rel_low, rel_high = relative_interval(k, a)

    return GBASEstimate(
        k=k,
        n_star=n_star,
        gamma_r=gamma_r,
        p_hat=p_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        rel_low=rel_low,
        rel_high=rel_high,
    )
