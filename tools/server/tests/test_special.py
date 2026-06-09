"""Numerical checks for the special-function layer."""

from __future__ import annotations

import math

import pytest

from mcscram_server.convergence import special


def test_normal_quantile_known_values() -> None:
    # Acklam's bare rational approximation (no Halley refinement) is accurate to
    # ~1e-9; the C++ (ci_utils.h) and TS (liveConvergenceStats.ts) stacks use the
    # identical coefficients, so all three agree to this level.
    assert special.normal_quantile_two_sided(0.95) == pytest.approx(1.959963984540054, abs=1e-8)
    assert special.normal_quantile_two_sided(0.99) == pytest.approx(2.5758293035489004, abs=1e-8)
    assert special.normal_quantile_two_sided(0.6826894921370859) == pytest.approx(1.0, abs=1e-6)


def test_normal_cdf_and_p_value() -> None:
    assert special.normal_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    # Two-sided p-value at z=1.959964 ~ 0.05.
    assert special.two_sided_p_value(1.959963984540054) == pytest.approx(0.05, abs=1e-6)


def test_regularized_incomplete_beta_matches_uniform() -> None:
    # Beta(1, 1) is Uniform(0, 1): I_x(1, 1) = x.
    for x in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert special.regularized_incomplete_beta(x, 1.0, 1.0) == pytest.approx(x, abs=1e-12)


def test_beta_quantile_inverts_cdf() -> None:
    a, b = 12.5, 88.5
    for q in (0.025, 0.5, 0.975):
        x = special.beta_quantile(q, a, b)
        assert special.regularized_incomplete_beta(x, a, b) == pytest.approx(q, abs=1e-8)


def test_beta_entropy_uniform_is_zero_bits() -> None:
    # Differential entropy of Uniform(0,1) is 0 nats -> 0 bits.
    assert special.beta_entropy_bits(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_required_trials_formula() -> None:
    z = special.normal_quantile_two_sided(0.95)
    p, eps = 0.01, 1.0e-3
    expected = math.ceil((z * z * p * (1.0 - p)) / (eps * eps))
    assert special.required_trials_two_sided(p, eps, z) == expected


def test_epsilon_from_trials_round_trip() -> None:
    z = special.normal_quantile_two_sided(0.99)
    p, n = 0.2, 100_000
    eps = special.epsilon_from_trials(p, n, z)
    assert special.required_trials_two_sided(p, eps, z) == pytest.approx(n, rel=1e-6)
