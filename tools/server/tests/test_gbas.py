"""Huber GBAS (2017): numerics, batched reconstruction, policy, and wiring.

Validated properties:
  * ``solve_k`` reproduces Huber's reference k for (eps=0.1, delta=0.01) -> 672.
  * The estimator is unbiased and its relative error ~ Gamma(k, k-1) is
    independent of p (the USRE property): sample mean ~ 1, sample sd ~ 1/sqrt(k-2)
    across decades of p.
  * The exact confidence interval attains nominal coverage.
  * The batched overshoot correction (Beta-Binomial reconstruction of N*) yields
    an unbiased estimate even with very large device batches -- a naive
    ``R ~ Gamma(total, 1)`` would be off by orders of magnitude here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammaincinv

from fuzzybits_server.convergence import (
    BayesPolicy,
    BernoulliSampler,
    ConvergenceConfig,
    ConvergenceRunner,
    GBASPolicy,
    WaldLinearPolicy,
    gbas,
    special,
)
from fuzzybits_server.convergence.tally import NodeCounters, NodeState, RunContext


# --------------------------------------------------------------------------- #
#  k-solver
# --------------------------------------------------------------------------- #
def test_solve_k_matches_huber_reference() -> None:
    # Huber's slides: (eps=0.1, delta=0.01) -> minimal integer k = 672.
    assert gbas.solve_k(0.1, 0.01) == 672
    # Minimality: 672 satisfies the bound, 671 does not.
    assert gbas.relative_error_tail(672, 0.1) <= 0.01
    assert gbas.relative_error_tail(671, 0.1) > 0.01


def test_solve_k_is_monotone_in_tolerance_and_confidence() -> None:
    base = gbas.solve_k(0.1, 0.01)
    assert gbas.solve_k(0.05, 0.01) > base      # tighter epsilon -> larger k
    assert gbas.solve_k(0.1, 0.001) > base       # smaller failure prob -> larger k
    assert gbas.solve_k(0.5, 0.2) >= gbas.K_MIN  # loose spec still well-formed


def test_relative_sd_and_interval() -> None:
    k = 402
    assert gbas.relative_sd(k) == pytest.approx(1.0 / math.sqrt(k - 2))
    assert gbas.relative_sd(2) == math.inf
    lo, hi = gbas.relative_interval(k, 0.05)
    assert lo < 1.0 < hi                          # brackets the truth (unbiased)
    # Tighter k narrows the relative interval.
    lo2, hi2 = gbas.relative_interval(4 * k, 0.05)
    assert (hi2 - lo2) < (hi - lo)


# --------------------------------------------------------------------------- #
#  Estimator law (ideal per-trial GBAS via N* ~ k + NegBinom(k, p))
# --------------------------------------------------------------------------- #
def _direct_relative_error(true_p: float, k: int, rng: np.random.Generator, reps: int) -> np.ndarray:
    """Vectorized ideal GBAS: N* = k + (#failures before k-th success); R ~ Gamma(N*,1)."""
    n_star = k + rng.negative_binomial(k, true_p, size=reps)
    gamma_r = rng.standard_gamma(n_star)
    p_hat = (k - 1.0) / gamma_r
    return p_hat / true_p


def test_estimator_unbiased_and_relative_error_independent_of_p() -> None:
    rng = np.random.default_rng(20260607)
    k = gbas.solve_k(0.1, 0.01)            # 672 -> theory rel sd ~ 0.0386
    theory_sd = 1.0 / math.sqrt(k - 2)
    sds = []
    for true_p in (0.5, 0.05, 0.005):
        rel = _direct_relative_error(true_p, k, rng, reps=200_000)
        assert rel.mean() == pytest.approx(1.0, abs=0.01)          # unbiased
        assert float(rel.std()) == pytest.approx(theory_sd, abs=0.004)
        sds.append(float(rel.std()))
    # USRE: the relative-error spread is the same across three decades of p.
    assert max(sds) - min(sds) < 0.004


def test_exact_confidence_interval_coverage() -> None:
    rng = np.random.default_rng(99)
    k = gbas.solve_k(0.1, 0.01)
    delta_fail = 0.01
    for true_p in (0.3, 0.02):
        n_star = k + rng.negative_binomial(k, true_p, size=120_000)
        gamma_r = rng.standard_gamma(n_star)
        lo = gammaincinv(k, delta_fail / 2.0) / gamma_r
        hi = gammaincinv(k, 1.0 - delta_fail / 2.0) / gamma_r
        coverage = float(np.mean((lo <= true_p) & (true_p <= hi)))
        assert coverage == pytest.approx(1.0 - delta_fail, abs=0.004)


# --------------------------------------------------------------------------- #
#  Batched overshoot reconstruction (drives the production code path)
# --------------------------------------------------------------------------- #
def test_batched_overshoot_reconstruction_is_unbiased() -> None:
    """Large batches push the k-th success deep inside the crossing pass.

    A naive ``R ~ Gamma(cur_total, 1)`` would estimate ~k/cur_total (orders of
    magnitude too small here); the Beta-Binomial reconstruction must recover an
    unbiased estimate.
    """
    rng = np.random.default_rng(5)
    k = 300
    true_p = 0.3
    trials_per_pass = 8192   # ~2458 successes in pass 1 >> k -> heavy overshoot
    reps = 1500
    estimates = np.empty(reps)
    for i in range(reps):
        prev_total = prev_ones = total = ones = 0
        while ones < k:
            prev_total, prev_ones = total, ones
            ones += int(rng.binomial(trials_per_pass, true_p))
            total += trials_per_pass
        est = gbas.estimate_at_crossing(
            k=k,
            prev_total=prev_total,
            prev_ones=prev_ones,
            cur_total=total,
            cur_ones=ones,
            ci_alpha=0.05,
            rng=rng,
        )
        assert prev_total < est.n_star <= total   # reconstructed index inside batch
        assert est.n_star >= k
        estimates[i] = est.p_hat
    rel = estimates / true_p
    assert rel.mean() == pytest.approx(1.0, abs=0.02)
    assert float(rel.std()) == pytest.approx(1.0 / math.sqrt(k - 2), abs=0.01)


# --------------------------------------------------------------------------- #
#  GBASPolicy
# --------------------------------------------------------------------------- #
def _ctx(delta: float, confidence: float = 0.99) -> RunContext:
    return RunContext(
        confidence=confidence,
        z=special.normal_quantile_two_sided(confidence),
        delta=delta,
        trials_per_pass=8192,
        target_index=1,
    )


def _node(ones: int, total: int, *, prev: tuple[int, int] | None = None) -> NodeState:
    node = NodeState(index=1)
    if prev is not None:
        node.update(NodeCounters(1, prev[1], prev[0]), wall_time=0.0)
    node.update(NodeCounters(1, ones, total), wall_time=1.0)
    return node


def test_gbas_policy_sizes_k_from_ctx_spec() -> None:
    pol = GBASPolicy(seed=0)
    verdict = pol.evaluate(_node(10, 8192), _ctx(delta=0.1, confidence=0.99))
    assert verdict.metrics["k"] == gbas.solve_k(0.1, 0.01)
    assert verdict.metrics["target_delta"] == pytest.approx(0.01)


def test_gbas_policy_not_converged_below_k() -> None:
    pol = GBASPolicy(seed=0)
    ctx = _ctx(delta=0.1, confidence=0.99)
    k = gbas.solve_k(0.1, 0.01)
    verdict = pol.evaluate(_node(k - 1, 50 * k), ctx)
    assert verdict.converged is False
    assert verdict.metrics["mean"] is None
    assert verdict.epsilon > verdict.target_epsilon  # rel sd still above target


def test_gbas_policy_converges_and_reports_exact_estimate() -> None:
    pol = GBASPolicy(seed=0)
    ctx = _ctx(delta=0.3, confidence=0.9)  # loose -> small k, reachable in a test
    k = gbas.solve_k(0.3, 0.1)
    node = _node(k + 40, 4 * (k + 40), prev=(3 * k, k - 1))
    verdict = pol.evaluate(node, ctx)
    assert verdict.converged is True
    assert verdict.epsilon <= verdict.target_epsilon
    assert verdict.metrics["mean"] is not None and verdict.metrics["mean"] > 0.0
    assert verdict.metrics["ci_low"] < verdict.metrics["ci_high"]
    assert verdict.metrics["n_star"] >= k


def test_gbas_policy_streams_nested_running_bands() -> None:
    pol = GBASPolicy(seed=0)
    ctx = _ctx(delta=0.1, confidence=0.99)  # client level = 99% (delta_fail = 0.01)
    k = gbas.solve_k(0.1, 0.01)
    m = pol.evaluate(_node(k // 2, 30 * k), ctx).metrics  # pre-convergence, S >= 3

    # All three running bands are present and ordered 95 ⊆ 99 ⊆ client(99.999...).
    for key in ("run_ci_low", "run_ci_high", "run_ci95_low", "run_ci95_high", "run_ci99_low", "run_ci99_high"):
        assert m[key] is not None and math.isfinite(m[key])
    assert m["run_ci95_low"] >= m["run_ci99_low"] >= m["run_ci_low"]
    assert m["run_ci95_high"] <= m["run_ci99_high"] <= m["run_ci_high"]
    # Client level here is wider than 99% (confidence 0.99 -> delta_fail 0.01 == 99% band),
    # so client band coincides with the 99% band.
    assert m["run_ci_low"] == pytest.approx(m["run_ci99_low"])
    assert m["run_ci_high"] == pytest.approx(m["run_ci99_high"])
    # Exact intervals are still withheld until convergence.
    assert m["ci95_low"] is None and m["ci99_high"] is None


def test_gbas_policy_exact_nested_intervals_at_convergence() -> None:
    pol = GBASPolicy(seed=3)
    ctx = _ctx(delta=0.3, confidence=0.9)  # loose -> small reachable k
    k = gbas.solve_k(0.3, 0.1)
    node = _node(k + 40, 4 * (k + 40), prev=(3 * k, k - 1))
    m = pol.evaluate(node, ctx).metrics
    assert m["ci95_low"] < m["ci95_high"]
    assert m["ci99_low"] < m["ci99_high"]
    # 95% is strictly inside 99% (same realized R; wider confidence -> wider interval).
    assert m["ci99_low"] < m["ci95_low"] < m["ci95_high"] < m["ci99_high"]
    # Exact CIs pivot on the realized R, consistent with the helper.
    assert (m["ci95_low"], m["ci95_high"]) == pytest.approx(
        gbas.confidence_interval(m["k"], m["gamma_r"], 0.05)
    )


def test_gbas_policy_caches_estimate_across_passes() -> None:
    pol = GBASPolicy(seed=1)
    ctx = _ctx(delta=0.3, confidence=0.9)
    k = gbas.solve_k(0.3, 0.1)
    node = _node(k + 40, 4 * (k + 40), prev=(3 * k, k - 1))
    first = pol.evaluate(node, ctx).metrics["mean"]
    # A later pass with more data must not re-randomize the materialized estimate.
    node.update(NodeCounters(1, k + 90, 5 * (k + 40)), wall_time=2.0)
    second = pol.evaluate(node, ctx).metrics["mean"]
    assert first == second


# --------------------------------------------------------------------------- #
#  End-to-end wiring through the runner + config
# --------------------------------------------------------------------------- #
def test_runner_gbas_stopping_uses_exact_estimate() -> None:
    sampler = BernoulliSampler(true_p=0.3, trials_per_pass=8192, target_index=1, seed=7)
    config = ConvergenceConfig(
        delta=0.05, confidence=0.95, early_stop=True, stopping_policy="gbas"
    )
    policies = [WaldLinearPolicy(), BayesPolicy(), GBASPolicy(seed=123)]
    frames: list[dict] = []
    runner = ConvergenceRunner(sampler, config, policies=policies, on_progress=frames.append)
    result = runner.run()

    assert result.converged is True
    assert result.stop_reason == "converged"

    final = result.final_progress
    assert final["gbas_converged"] is True
    assert final["gbas_k"] == gbas.solve_k(0.05, 0.05)
    assert final["gbas_successes"] >= final["gbas_k"]
    assert final["gbas_mean"] is not None

    summary = result.summary
    # The authoritative result is the GBAS estimate, not the raw sample mean.
    assert summary["mean"] == pytest.approx(final["gbas_mean"], rel=1e-12)
    assert summary["mean"] == pytest.approx(0.3, abs=0.03)
    assert summary["sample_mean"] == pytest.approx(result.state.target.mean, rel=1e-12)
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]


def test_native_options_strip_host_only_policy() -> None:
    # The native compiler validates ci-policy against {wald, bayes} and rejects
    # host-only values like "gbas"; the host engine must not forward them.
    from fuzzybits_server.convergence.sampling import _native_options

    native = _native_options(
        {
            "policy": "gbas",
            "early_stop": True,
            "emit_min_interval_s": 0.016,
            "stream_nodes": True,
            "delta": 0.05,
            "confidence": 0.95,
            "num_trials": 1_000_000,
            "oracle": 0.01,
        }
    )
    assert "policy" not in native
    assert "early_stop" not in native
    assert "emit_min_interval_s" not in native
    assert "stream_nodes" not in native
    # Compile-relevant keys are preserved untouched.
    assert native["delta"] == 0.05
    assert native["confidence"] == 0.95
    assert native["num_trials"] == 1_000_000
    assert native["oracle"] == 0.01
    # Still selects gbas host-side from the *unstripped* options.
    assert ConvergenceConfig.from_options({"policy": "gbas"}).stopping_policy == "gbas"


def test_config_and_registry_allow_gbas() -> None:
    config = ConvergenceConfig.from_options(
        {"policy": "gbas", "delta": 0.05, "confidence": 0.95, "early_stop": True}
    )
    assert config.stopping_policy == "gbas"
    assert {"wald", "bayes", "gbas"} <= set(config.policies)

    from fuzzybits_server.convergence.policies import make_policy

    assert isinstance(make_policy("gbas"), GBASPolicy)
