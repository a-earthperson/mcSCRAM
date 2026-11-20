"""Convergence policies: decide *when* a run has reached the requested precision.

A :class:`ConvergencePolicy` maps an immutable ``(NodeState, RunContext)`` to a
:class:`PolicyVerdict` carrying a ``converged`` flag plus the precision
statistics characteristic of that policy (current half-width ``epsilon``, target
``target_epsilon``, projected sample size, and a flat ``metrics`` dict for
emission).

Two policies are provided:

* :class:`WaldLinearPolicy` — classical normal-approximation interval on the
  linear probability scale.  ``epsilon = z * SE``, target ``= delta * p_hat``.
  This is the engine's default *primary* policy: it fills the canonical progress
  fields the web client already consumes.  (Linear only, per requirements; the
  log10 band is still computed and emitted for context but does not gate the
  stop.)
* :class:`GBASPolicy` — Huber's Gamma Bernoulli Approximation Scheme (2017).
  Instead of gating on a moving ``delta * p_hat`` interval width, it fixes a
  target success count ``k`` up front (from the *same* ``delta``/``confidence``
  spec) such that the estimate ``p_hat = (k-1)/R`` has an *exactly known*,
  ``p``-independent relative error ``p/p_hat ~ Gamma(k, k-1)``; it converges the
  instant ``num_one_bits >= k`` and reports an exact confidence interval.  Here
  ``epsilon`` is the current relative standard deviation ``1/sqrt(successes-2)``
  and target ``= 1/sqrt(k-2)``, so ``epsilon <= target`` iff ``successes >= k``.

The runner can attach any number of policies; one is designated the *stopping*
policy.  All are evaluated every pass and emitted under their namespace, so e.g.
Wald and GBAS statistics stream simultaneously.

Most policies are stateless; :class:`GBASPolicy` keeps a little state (the solved
``k``, an RNG, and the estimate materialized at the first crossing) because the
GBAS estimate is a randomized quantity defined *at the stopping time*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from . import special
from .special import DELTA_EPSILON
from .tally import NodeState, RunContext

_Z95 = 1.959963984540054
_Z99 = 2.5758293035489004

# Fixed two-sided significance levels for the 95% / 99% GBAS bands streamed
# alongside the client-requested level (parity with the p05-p95 / p01-p99 display).
_ALPHA95 = 0.05
_ALPHA99 = 0.01


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    """Outcome of evaluating one convergence policy for one node."""

    name: str
    converged: bool
    epsilon: float
    target_epsilon: float
    projected_trials: int
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConvergencePolicy(Protocol):
    """Decides convergence and reports precision statistics for one node."""

    name: str

    def evaluate(self, node: NodeState, ctx: RunContext) -> PolicyVerdict:
        ...


class WaldLinearPolicy:
    """Wald (normal-approximation) convergence on the linear probability scale."""

    name = "wald"

    def evaluate(self, node: NodeState, ctx: RunContext) -> PolicyVerdict:
        z = ctx.z
        n = node.total_bits
        p_hat = node.mean
        p_guard = max(p_hat, DELTA_EPSILON)
        std_err = node.std_err

        epsilon = z * std_err
        target_epsilon = ctx.delta * p_guard
        converged = epsilon > 0.0 and epsilon <= target_epsilon

        # log10-scale band (emitted for context; does not gate the stop).
        epsilon_log10 = (z * std_err) / (p_guard * special.LN10) if n else 0.0
        target_epsilon_log10 = ctx.delta
        log10_converged = epsilon_log10 > 0.0 and epsilon_log10 <= target_epsilon_log10

        projected_trials = special.required_trials_two_sided(p_hat, target_epsilon, z)
        projected_trials_log10 = special.required_trials_log10_two_sided(
            p_guard, target_epsilon_log10, z
        )

        hw95 = _Z95 * std_err
        hw99 = _Z99 * std_err
        metrics: dict[str, Any] = {
            "std_err": std_err,
            "epsilon": epsilon,
            "target_epsilon": target_epsilon,
            "epsilon_converged": converged,
            "epsilon_log10": epsilon_log10,
            "target_epsilon_log10": target_epsilon_log10,
            "epsilon_log10_converged": log10_converged,
            "projected_trials": projected_trials,
            "projected_trials_log10": projected_trials_log10,
            "p01": special.clamp(p_hat - hw99, 0.0, 1.0),
            "p05": special.clamp(p_hat - hw95, 0.0, 1.0),
            "p95": special.clamp(p_hat + hw95, 0.0, 1.0),
            "p99": special.clamp(p_hat + hw99, 0.0, 1.0),
        }
        return PolicyVerdict(
            name=self.name,
            converged=converged,
            epsilon=epsilon,
            target_epsilon=target_epsilon,
            projected_trials=max(projected_trials, projected_trials_log10),
            metrics=metrics,
        )


class GBASPolicy:
    """Huber GBAS (2017): exact relative-error control via a success-count stopping time.

    The relative tolerance and failure probability reuse the engine's existing
    spec rather than introducing a new knob:

    * ``epsilon``    := ``ctx.delta``            (relative tolerance)
    * ``delta_fail`` := ``1 - ctx.confidence``   (failure probability)

    On the first evaluation ``k`` is solved once via :func:`gbas.solve_k`.  Every
    pass reports progress (current/target relative SD, projected trials); once
    ``num_one_bits >= k`` the GBAS estimate ``(k-1)/R`` and its exact confidence
    interval are materialized (and cached, since the estimate is randomized and
    defined at the crossing).  Pass ``seed`` for reproducible randomization.
    """

    name = "gbas"

    def __init__(
        self,
        *,
        epsilon: float | None = None,
        delta_fail: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._epsilon = epsilon
        self._delta_fail = delta_fail
        self._seed = seed
        self._rng: Any = None
        self._k: int | None = None
        self._eps: float = 0.0
        self._delta: float = 0.0
        self._achieved_delta: float | None = None
        self._target_rel_sd: float = math.inf
        self._estimate: Any = None  # gbas.GBASEstimate, cached at first crossing

    def _configure(self, ctx: RunContext) -> None:
        if self._k is not None:
            return
        import numpy as np

        from . import gbas

        self._eps = float(self._epsilon if self._epsilon is not None else ctx.delta)
        self._delta = float(
            self._delta_fail if self._delta_fail is not None else (1.0 - ctx.confidence)
        )
        self._k = gbas.solve_k(self._eps, self._delta)
        self._achieved_delta = gbas.relative_error_tail(self._k, self._eps)
        self._target_rel_sd = gbas.relative_sd(self._k)
        self._rng = np.random.default_rng(self._seed)

    def evaluate(self, node: NodeState, ctx: RunContext) -> PolicyVerdict:
        from . import gbas

        self._configure(ctx)
        k = self._k
        assert k is not None  # set by _configure
        successes = node.num_one_bits
        total = node.total_bits
        converged = successes >= k

        current_rel_sd = gbas.relative_sd(successes)

        if successes > 0:
            projected = max(total, math.ceil(total * k / successes))
        else:
            projected = 0

        # Running GBAS confidence bands: the *exact* relative interval at the current
        # success count, scaled by the running mean.  This is a "precision so far"
        # envelope (independent of p) that shrinks toward the exact interval the
        # estimate settles into at the stopping time.  We emit the client-requested
        # level (``run_ci_*``) plus fixed 95%/99% bands so the GBAS plot has the same
        # nested-CI presentation the client uses elsewhere (p05-p95 / p01-p99).
        def _running_band(ci_alpha: float) -> tuple[float | None, float | None]:
            if successes < gbas.K_MIN:
                return None, None
            rel_lo, rel_hi = gbas.relative_interval(successes, ci_alpha)
            run_mean = node.mean
            return max(0.0, run_mean * rel_lo), min(1.0, run_mean * rel_hi)

        run_ci_low, run_ci_high = _running_band(self._delta)
        run_ci95_low, run_ci95_high = _running_band(_ALPHA95)
        run_ci99_low, run_ci99_high = _running_band(_ALPHA99)

        metrics: dict[str, Any] = {
            "k": k,
            "successes": successes,
            "target_delta": self._delta,
            "achieved_delta": self._achieved_delta,
            "relative_tolerance": self._eps,
            "rel_sd": current_rel_sd,
            "rel_sd_target": self._target_rel_sd,
            "projected_trials": projected,
            "run_ci_low": run_ci_low,
            "run_ci_high": run_ci_high,
            "run_ci95_low": run_ci95_low,
            "run_ci95_high": run_ci95_high,
            "run_ci99_low": run_ci99_low,
            "run_ci99_high": run_ci99_high,
            "mean": None,
            "std_err": None,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci99_low": None,
            "ci99_high": None,
            "rel_low": None,
            "rel_high": None,
            "n_star": None,
            "gamma_r": None,
        }

        if converged:
            if self._estimate is None:
                prev = node.previous_point()
                prev_total = prev.total_bits if prev is not None else 0
                prev_ones = prev.num_one_bits if prev is not None else 0
                if prev_ones >= k:  # defensive: policy attached after the crossing
                    prev_total, prev_ones = 0, 0
                self._estimate = gbas.estimate_at_crossing(
                    k=k,
                    prev_total=prev_total,
                    prev_ones=prev_ones,
                    cur_total=total,
                    cur_ones=successes,
                    ci_alpha=self._delta,
                    rng=self._rng,
                )
            est = self._estimate
            # Posterior-free relative SD of the *estimator* (exact, p-independent).
            std_err = est.p_hat * self._target_rel_sd if math.isfinite(self._target_rel_sd) else None
            # Exact intervals at the stopping time: the client level (cached on the
            # estimate) plus fixed 95%/99% pivots on the same realized R.
            ci95_low, ci95_high = gbas.confidence_interval(k, est.gamma_r, _ALPHA95)
            ci99_low, ci99_high = gbas.confidence_interval(k, est.gamma_r, _ALPHA99)
            metrics.update(
                mean=est.p_hat,
                std_err=std_err,
                ci_low=est.ci_low,
                ci_high=est.ci_high,
                ci95_low=ci95_low,
                ci95_high=ci95_high,
                ci99_low=ci99_low,
                ci99_high=ci99_high,
                rel_low=est.rel_low,
                rel_high=est.rel_high,
                n_star=est.n_star,
                gamma_r=est.gamma_r,
            )

        return PolicyVerdict(
            name=self.name,
            converged=converged,
            epsilon=current_rel_sd,
            target_epsilon=self._target_rel_sd,
            projected_trials=projected,
            metrics=metrics,
        )


# Registry so config strings ("wald", "gbas") resolve to policy factories.
POLICY_REGISTRY: dict[str, type[ConvergencePolicy]] = {
    WaldLinearPolicy.name: WaldLinearPolicy,
    GBASPolicy.name: GBASPolicy,
}


def make_policy(name: str) -> ConvergencePolicy:
    """Instantiate a policy by name (``"wald"`` / ``"gbas"``)."""
    try:
        factory = POLICY_REGISTRY[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown convergence policy {name!r}; known: {sorted(POLICY_REGISTRY)}"
        ) from exc
    return factory()


__all__ = [
    "PolicyVerdict",
    "ConvergencePolicy",
    "WaldLinearPolicy",
    "GBASPolicy",
    "POLICY_REGISTRY",
    "make_policy",
]
