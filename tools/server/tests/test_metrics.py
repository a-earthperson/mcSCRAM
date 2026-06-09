"""Metric-layer checks: point estimate, gradient/Hessian, information gain."""

from __future__ import annotations

import math

import pytest

from mcscram_server.convergence import (
    GradientMetric,
    InfoGainMetric,
    OracleDiagnostics,
    PointEstimate,
)
from mcscram_server.convergence.tally import NodeCounters, NodeState, RunContext
from mcscram_server.convergence import special


def _ctx(oracle_p: float | None = None) -> RunContext:
    confidence = 0.99
    return RunContext(
        confidence=confidence,
        z=special.normal_quantile_two_sided(confidence),
        delta=1.0e-3,
        trials_per_pass=10_000,
        target_index=1,
        oracle_p=oracle_p,
    )


def _node_from_series(series: list[tuple[int, int]], index: int = 1) -> NodeState:
    """Build a node by replaying cumulative (num_one_bits, total_bits) points."""
    node = NodeState(index=index)
    for t, (ones, total) in enumerate(series):
        node.update(NodeCounters(index, ones, total), wall_time=float(t))
    return node


def test_point_estimate() -> None:
    node = _node_from_series([(100, 10_000)])
    out = PointEstimate().compute(node, _ctx())
    assert out["mean"] == pytest.approx(0.01)
    assert out["std_err"] == pytest.approx(math.sqrt(0.01 * 0.99 / 10_000))
    assert out["trials"] == 10_000


def test_gradient_first_pass_is_none() -> None:
    node = _node_from_series([(100, 10_000)])
    out = GradientMetric(label="p").compute(node, _ctx())
    assert out["grad_p"] is None
    assert out["neg_grad_p"] is None


def test_negative_gradient_sign_and_value() -> None:
    # p_hat decreases 0.01 -> 0.0075 over +10_000 trials.
    node = _node_from_series([(100, 10_000), (150, 20_000)])
    out = GradientMetric(label="p").compute(node, _ctx())
    expected_grad = (0.0075 - 0.01) / 10_000
    assert out["grad_p"] == pytest.approx(expected_grad, rel=1e-9)
    assert out["neg_grad_p"] == pytest.approx(-expected_grad, rel=1e-9)
    assert out["neg_grad_p"] > 0.0  # estimate falling => negative gradient positive
    assert out["grad_p_per_pass"] == pytest.approx(-0.0025, rel=1e-9)


def test_hessian_requires_three_points() -> None:
    node = _node_from_series([(100, 10_000), (150, 20_000)])
    assert GradientMetric(label="p").compute(node, _ctx())["hess_p"] is None
    node.update(NodeCounters(1, 240, 30_000), wall_time=2.0)  # p = 0.008
    hess = GradientMetric(label="p").compute(node, _ctx())["hess_p"]
    assert hess is not None and math.isfinite(hess)


def test_info_gain_positive_and_monotone_total() -> None:
    metric = InfoGainMetric()
    node = _node_from_series([(100, 10_000)])
    first = metric.compute(node, _ctx())
    node.update(NodeCounters(1, 200, 20_000), wall_time=1.0)
    second = metric.compute(node, _ctx())
    assert first["info_bits_total"] is not None and first["info_bits_total"] > 0.0
    # More data -> more cumulative information.
    assert second["info_bits_total"] > first["info_bits_total"]
    assert second["info_bits_last"] is not None and second["info_bits_last"] > 0.0


def test_oracle_diagnostics_disabled_without_oracle() -> None:
    node = _node_from_series([(100, 10_000)])
    assert OracleDiagnostics().compute(node, _ctx(oracle_p=None)) == {}


def test_oracle_diagnostics_values() -> None:
    node = _node_from_series([(100, 10_000)])  # p_hat = 0.01
    out = OracleDiagnostics().compute(node, _ctx(oracle_p=0.012))
    assert out["bias"] == pytest.approx(0.01 - 0.012)
    assert out["abs_error"] == pytest.approx(0.002)
    assert out["oracle_p"] == 0.012
    assert isinstance(out["ci95_covered"], bool)
