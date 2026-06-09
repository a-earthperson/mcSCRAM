"""Metrics: auxiliary host-side quantities derived from the tally trajectory.

A :class:`Metric` maps an immutable ``(NodeState, RunContext)`` to a flat dict of
named values.  Metrics are *non-stopping*: they describe the run but never decide
when to halt (that is the job of :mod:`.policies`).  The runner attaches any
number of metrics and merges their outputs into each progress frame, so several
families of statistics are produced simultaneously every pass.

Provided metrics:

* :class:`PointEstimate` — ``mean`` (p_hat), ``std_err``, raw counters.
* :class:`GradientMetric` — finite-difference gradient/Hessian of p_hat with
  respect to accumulated trials.  Emits ``neg_grad_p`` (deliverable: negative
  gradient of p_hat) alongside the raw gradient and curvature.
* :class:`InfoGainMetric` — Shannon information gain (bits) of a ``Beta(1/2,1/2)``
  posterior, matching the native ``info_gain`` tracker.
* :class:`OracleDiagnostics` — accuracy + sampling-theory diagnostics versus a
  known ground-truth ``oracle_p`` (only when diagnostics are enabled).

Adding a new metric is a single class with a ``compute`` method; no other layer
changes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol, runtime_checkable

from . import special
from .tally import HistoryPoint, NodeState, RunContext


@runtime_checkable
class Metric(Protocol):
    """A named, stateless producer of derived statistics for one node."""

    name: str

    def compute(self, node: NodeState, ctx: RunContext) -> dict[str, Any]:
        ...


class PointEstimate:
    """Sample proportion and its binomial standard error."""

    name = "point_estimate"

    def compute(self, node: NodeState, ctx: RunContext) -> dict[str, Any]:
        return {
            "mean": node.mean,
            "std_err": node.std_err,
            "num_one_bits": node.num_one_bits,
            "total_bits": node.total_bits,
            "trials": node.total_bits,
        }


class GradientMetric:
    r"""Finite-difference gradient and Hessian of a scalar series vs. trials ``N``.

    By default differentiates the point estimate ``p_hat(N)``.  With uniform pass
    spacing ``h = trials_per_pass``:

    * gradient (backward 1st order): :math:`g = (v_k - v_{k-1}) / (N_k - N_{k-1})`
    * curvature (central 2nd diff): :math:`H = (v_k - 2v_{k-1} + v_{k-2}) / h^2`

    Emits, for ``label='p'``:

    * ``grad_p``      — :math:`dv/dN` (per trial)
    * ``neg_grad_p``  — :math:`-dv/dN` (the requested negative gradient)
    * ``grad_p_per_pass`` — :math:`v_k - v_{k-1}` (per-pass change, unitless)
    * ``hess_p``      — :math:`d^2v/dN^2` (per trial squared), once >=3 points
    """

    def __init__(
        self,
        *,
        label: str = "p",
        value_fn: Callable[[HistoryPoint], float] | None = None,
    ) -> None:
        self.name = f"gradient[{label}]"
        self._label = label
        self._value_fn = value_fn or (lambda point: point.mean)

    def compute(self, node: NodeState, ctx: RunContext) -> dict[str, Any]:
        history = node.history
        out: dict[str, Any] = {
            f"grad_{self._label}": None,
            f"neg_grad_{self._label}": None,
            f"grad_{self._label}_per_pass": None,
            f"hess_{self._label}": None,
        }
        if len(history) < 2:
            return out

        latest = history[-1]
        prev = history[-2]
        v_k = self._value_fn(latest)
        v_km1 = self._value_fn(prev)
        dn = latest.total_bits - prev.total_bits
        if dn > 0:
            grad = (v_k - v_km1) / dn
            out[f"grad_{self._label}"] = grad
            out[f"neg_grad_{self._label}"] = -grad
        out[f"grad_{self._label}_per_pass"] = v_k - v_km1

        if len(history) >= 3:
            prev2 = history[-3]
            v_km2 = self._value_fn(prev2)
            # Uniform spacing h; use the average step to stay robust if a pass was
            # coalesced (dn != h).
            h1 = latest.total_bits - prev.total_bits
            h2 = prev.total_bits - prev2.total_bits
            if h1 > 0 and h2 > 0:
                h = 0.5 * (h1 + h2)
                out[f"hess_{self._label}"] = (v_k - 2.0 * v_km1 + v_km2) / (h * h)
        return out


class InfoGainMetric:
    r"""Shannon information gain (bits) for a ``Beta(alpha0, beta0)`` posterior.

    Mirrors ``mc/stats/info_gain.h``: with a Jeffreys prior ``Beta(1/2, 1/2)`` the
    cumulative information gained from the prior to the current posterior is the
    entropy reduction ``H(prior) - H(posterior)``; the per-pass gain compares the
    previous and current posteriors.
    """

    def __init__(self, *, alpha0: float = 0.5, beta0: float = 0.5) -> None:
        self.name = "info_gain"
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._prior_entropy = special.beta_entropy_bits(alpha0, beta0)

    def _posterior_entropy(self, ones: int, total: int) -> float:
        return special.beta_entropy_bits(self._alpha0 + ones, self._beta0 + (total - ones))

    def compute(self, node: NodeState, ctx: RunContext) -> dict[str, Any]:
        if node.total_bits <= 0:
            return {"info_bits_last": None, "info_bits_total": None}
        current_entropy = self._posterior_entropy(node.num_one_bits, node.total_bits)
        total_gain = self._prior_entropy - current_entropy
        previous = node.previous_point()
        last_gain: float | None = None
        if previous is not None and previous.total_bits > 0:
            previous_entropy = self._posterior_entropy(previous.num_one_bits, previous.total_bits)
            last_gain = previous_entropy - current_entropy
        return {
            "info_bits_last": last_gain if last_gain is None or math.isfinite(last_gain) else None,
            "info_bits_total": total_gain if math.isfinite(total_gain) else None,
        }


class OracleDiagnostics:
    """Accuracy + sampling-theory diagnostics versus a known ground truth.

    Active only when ``ctx.diagnostics_enabled`` (i.e. an ``oracle_p >= 0`` was
    supplied).  Mirrors ``mc/stats/diagnostics.h``.
    """

    name = "oracle_diagnostics"

    def compute(self, node: NodeState, ctx: RunContext) -> dict[str, Any]:
        if not ctx.diagnostics_enabled or ctx.oracle_p is None:
            return {}
        p_true = ctx.oracle_p
        mean = node.mean
        std_err = node.std_err
        z = ctx.z

        bias = mean - p_true
        abs_error = abs(bias)
        rel_error = abs_error / p_true if p_true != 0.0 else math.nan
        mse = bias * bias
        log10_abs_error = math.log10(abs_error) if abs_error > 0.0 else math.nan
        abs_log10_error = (
            abs(math.log10(mean) - math.log10(p_true)) if (mean > 0.0 and p_true > 0.0) else math.nan
        )

        z_score = bias / std_err if std_err > 0.0 else math.nan
        p_value = special.two_sided_p_value(z_score) if std_err > 0.0 else math.nan

        # Wald 95/99 coverage (mirrors populate_point_estimates ci layout).
        hw95 = 1.959963984540054 * std_err
        hw99 = 2.5758293035489004 * std_err
        ci95_covered = (mean - hw95) <= p_true <= (mean + hw95)
        ci99_covered = (mean - hw99) <= p_true <= (mean + hw99)

        eps_target = ctx.delta * max(mean, special.DELTA_EPSILON)
        n_required = (
            special.required_trials_two_sided(p_true, eps_target, z)
            if 0.0 < ctx.confidence < 1.0
            else 0
        )
        n_ratio = node.total_bits / n_required if n_required > 0 else math.nan

        return {
            "oracle_p": p_true,
            "bias": bias,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "mse": mse,
            "log10_abs_error": log10_abs_error,
            "abs_log10_error": abs_log10_error,
            "z_score": z_score,
            "p_value": p_value,
            "ci95_covered": ci95_covered,
            "ci99_covered": ci99_covered,
            "n_required": n_required,
            "n_ratio": n_ratio,
        }


def default_metrics(*, with_gradient: bool = True, with_info_gain: bool = True) -> list[Metric]:
    """Convenience: the standard metric stack used by the live demo."""
    metrics: list[Metric] = [PointEstimate()]
    if with_gradient:
        metrics.append(GradientMetric(label="p"))
    if with_info_gain:
        metrics.append(InfoGainMetric())
    metrics.append(OracleDiagnostics())
    return metrics


__all__ = [
    "Metric",
    "PointEstimate",
    "GradientMetric",
    "InfoGainMetric",
    "OracleDiagnostics",
    "default_metrics",
]
