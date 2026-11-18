"""CPU (OpenMP) tests for the pip-installable fuzzybits package.

These run with small trial counts and do not require a GPU. They expect the
native extension to be importable — either via ``pip install`` of a wheel or
an in-tree editable build.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


def test_version_and_init():
    assert fuzzybits.__version__
    devices = fuzzybits.list_devices()
    assert isinstance(devices, list)
    assert devices, "OpenMP host device should always be present"
    backends = {d.get("backend") for d in devices}
    assert "omp" in backends or any(d.get("type") == "cpu" for d in devices)


def test_mcscram_native_shim():
    import mcscram_native

    assert hasattr(mcscram_native, "run_probability")
    assert hasattr(mcscram_native, "list_devices")
    assert mcscram_native.list_devices()


def test_probability_tiny(tiny_xml: Path):
    # TwoTrains: P = 0.76 * 0.85 = 0.646. A few 10^5 trials is enough for a
    # smoke check that the estimate is in (0, 1) and the Wald interval covers
    # the analytic value.
    result = fuzzybits.probability(tiny_xml, trials=1 << 16, seed=1, pdag_c=2, no_kn=True, policy="wald")
    assert 0.0 < result.estimate < 1.0
    lo, hi = result.ci
    assert lo <= result.estimate <= hi
    assert result.std_err >= 0.0
    analytic = 0.646
    z = 2.576  # 99% — keep the unit test loose; GPU acceptance uses z·σ̂
    assert abs(result.estimate - analytic) <= z * max(result.std_err, 1e-6)


def test_run_pcfb_is_extension_point():
    with pytest.raises(NotImplementedError, match="PCFB"):
        fuzzybits.run("model.pcfb")


def test_native_run_probability_dict(tiny_xml: Path):
    native = fuzzybits.native()
    payload = native.run_probability(
        [str(tiny_xml)],
        {"num_trials": float(1 << 14), "compilation_level": 2, "no_kn": True, "policy": "wald"},
        None,
    )
    assert "probabilities" in payload
    p = payload["probabilities"][0]["p_total"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
