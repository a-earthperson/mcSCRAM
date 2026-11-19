"""End-to-end engine checks using the pure-Python BernoulliSampler.

These exercise the full pass -> tally -> update -> emit loop and the primary
deliverables simultaneously:
  (a) Wald-linear convergence statistics (canonical fields),
  (b) Bayesian policy statistics (``bayes_*``).
"""

from __future__ import annotations

import math

import pytest

from mcscram_server.convergence import (
    BernoulliSampler,
    ConvergenceConfig,
    ConvergenceRunner,
    run_calculation,  # noqa: F401  (import smoke; native path not exercised here)
)

CANONICAL_FIELDS = {
    "phase",
    "iterations",
    "trials",
    "mean",
    "std_err",
    "epsilon",
    "target_epsilon",
    "epsilon_converged",
    "p01",
    "p05",
    "p95",
    "p99",
    "projected_trials",
    "remaining_trials",
}


def _collect(config: ConvergenceConfig, sampler: BernoulliSampler):
    frames: list[dict] = []
    runner = ConvergenceRunner(sampler, config, on_progress=frames.append)
    result = runner.run()
    return result, frames


def test_converges_to_true_probability() -> None:
    sampler = BernoulliSampler(true_p=0.2, trials_per_pass=4096, target_index=1, seed=7)
    config = ConvergenceConfig(delta=0.05, confidence=0.95, early_stop=True)
    result, frames = _collect(config, sampler)

    assert result.converged is True
    assert result.stop_reason == "converged"
    assert result.state.target.mean == pytest.approx(0.2, abs=0.02)

    final = result.final_progress
    # (a) Wald-linear canonical stats present and satisfy the stop criterion.
    missing = CANONICAL_FIELDS - final.keys()
    assert not missing, f"missing canonical fields: {missing}"
    assert final["epsilon"] <= final["target_epsilon"]
    assert final["epsilon_converged"] is True


def test_emits_bayes_streams() -> None:
    sampler = BernoulliSampler(true_p=0.2, trials_per_pass=4096, target_index=1, seed=7)
    config = ConvergenceConfig(delta=0.05, confidence=0.95, early_stop=True, burn_in=4096 * 3)
    result, frames = _collect(config, sampler)
    final = result.final_progress

    # Bayesian policy stats streamed alongside Wald.
    for key in ("bayes_mean", "bayes_epsilon", "bayes_target_epsilon", "bayes_converged"):
        assert key in final
    assert final["bayes_mean"] == pytest.approx(0.2, abs=0.02)
    assert math.isfinite(final["bayes_epsilon"])

    # Info gain accumulates.
    assert final["info_bits_total"] is not None and final["info_bits_total"] > 0.0


def test_burn_in_defers_convergence() -> None:
    sampler = BernoulliSampler(true_p=0.5, trials_per_pass=4096, target_index=1, seed=1)
    burn_in = 4096 * 4
    config = ConvergenceConfig(delta=10.0, confidence=0.95, early_stop=True, burn_in=burn_in)
    result, frames = _collect(config, sampler)

    burn_frames = [f for f in frames if f["phase"] == "burn_in"]
    assert burn_frames, "expected burn-in frames"
    assert all(f["converged"] is False for f in burn_frames)
    # The run only stops after burn-in completes.
    assert result.state.target.total_bits >= burn_in
    assert result.final_progress["phase"] == "sampling"


def test_max_trials_stop() -> None:
    sampler = BernoulliSampler(true_p=0.3, trials_per_pass=4096, target_index=1, seed=2)
    config = ConvergenceConfig(max_trials=4096 * 3, early_stop=False)
    result, frames = _collect(config, sampler)
    assert result.stop_reason == "max_trials"
    assert result.state.pass_count == 3
    assert result.state.target.total_bits == 4096 * 3


def test_config_from_options_maps_server_keys() -> None:
    config = ConvergenceConfig.from_options(
        {
            "delta": 1e-9,
            "confidence": 0.999,
            "burn_in": 1024.0,
            "num_trials": 65536.0,
            "early_stop": True,
            "oracle": 0.01,
            "policy": "bayes",
        }
    )
    assert config.delta == 1e-9
    assert config.confidence == 0.999
    assert config.burn_in == 1024
    assert config.max_trials == 65536
    assert config.early_stop is True
    assert config.oracle_p == 0.01
    assert config.stopping_policy == "bayes"
    assert "wald" in config.policies and "bayes" in config.policies


def test_oracle_diagnostics_round_trip() -> None:
    sampler = BernoulliSampler(true_p=0.2, trials_per_pass=8192, target_index=1, seed=3)
    config = ConvergenceConfig(
        delta=0.05, confidence=0.95, early_stop=True, oracle_p=0.2
    )
    result, _ = _collect(config, sampler)
    final = result.final_progress
    assert "z_score" in final and "abs_error" in final
    assert final["oracle_p"] == 0.2
