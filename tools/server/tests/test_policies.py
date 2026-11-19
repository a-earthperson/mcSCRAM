"""Policy-layer checks: Wald-linear and Bayes convergence verdicts."""

from __future__ import annotations

import math

import pytest

from mcscram_server.convergence import BayesPolicy, WaldLinearPolicy
from mcscram_server.convergence import special
from mcscram_server.convergence.tally import NodeCounters, NodeState, RunContext


def _node(ones: int, total: int, index: int = 1) -> NodeState:
    node = NodeState(index=index)
    node.update(NodeCounters(index, ones, total), wall_time=0.0)
    return node


def _ctx(delta: float, confidence: float = 0.99, oracle_p: float | None = None) -> RunContext:
    return RunContext(
        confidence=confidence,
        z=special.normal_quantile_two_sided(confidence),
        delta=delta,
        trials_per_pass=10_000,
        target_index=1,
        oracle_p=oracle_p,
    )


def test_wald_epsilon_equals_z_times_std_err() -> None:
    node = _node(ones=100, total=10_000)
    ctx = _ctx(delta=1.0e-3)
    verdict = WaldLinearPolicy().evaluate(node, ctx)
    assert verdict.epsilon == pytest.approx(ctx.z * node.std_err, rel=1e-12)
    # target = delta * p_hat
    assert verdict.target_epsilon == pytest.approx(ctx.delta * node.mean, rel=1e-12)


def test_wald_convergence_threshold() -> None:
    node = _node(ones=100, total=10_000)  # p_hat = 0.01
    # Tight delta -> not converged.
    assert not WaldLinearPolicy().evaluate(node, _ctx(delta=1.0e-3)).converged
    # Loose delta -> converged (epsilon <= delta * p_hat).
    assert WaldLinearPolicy().evaluate(node, _ctx(delta=0.5)).converged


def test_wald_projected_trials_matches_formula() -> None:
    node = _node(ones=200, total=20_000)  # p_hat = 0.01
    ctx = _ctx(delta=0.05, confidence=0.95)
    verdict = WaldLinearPolicy().evaluate(node, ctx)
    target = ctx.delta * node.mean
    expected = special.required_trials_two_sided(node.mean, target, ctx.z)
    assert verdict.metrics["projected_trials"] == expected


def test_wald_zero_trials_not_converged() -> None:
    node = _node(ones=0, total=0)
    verdict = WaldLinearPolicy().evaluate(node, _ctx(delta=1.0))
    assert verdict.epsilon == 0.0
    assert not verdict.converged


def test_bayes_posterior_mean_and_interval() -> None:
    node = _node(ones=100, total=10_000)
    ctx = _ctx(delta=1.0e-3, confidence=0.95)
    verdict = BayesPolicy().evaluate(node, ctx)
    alpha = 0.5 + 100
    beta = 0.5 + 9_900
    assert verdict.metrics["mean"] == pytest.approx(alpha / (alpha + beta), rel=1e-12)
    lower = verdict.metrics["ci_low"]
    upper = verdict.metrics["ci_high"]
    assert 0.0 < lower < verdict.metrics["mean"] < upper < 1.0
    assert verdict.epsilon == pytest.approx(0.5 * (upper - lower), rel=1e-12)


def test_bayes_converges_with_loose_delta() -> None:
    node = _node(ones=1_000, total=100_000)  # p_hat ~ 0.01, tight posterior
    assert BayesPolicy().evaluate(node, _ctx(delta=0.5)).converged
    assert not BayesPolicy().evaluate(node, _ctx(delta=1.0e-4)).converged


def test_bayes_epsilon_finite_and_positive() -> None:
    node = _node(ones=5, total=1_000)
    verdict = BayesPolicy().evaluate(node, _ctx(delta=1.0e-3))
    assert verdict.epsilon > 0.0 and math.isfinite(verdict.epsilon)
