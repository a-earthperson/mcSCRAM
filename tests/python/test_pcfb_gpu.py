"""Short GPU bit-identity check (RTX 3070 / generic wheel). Isolated from the OMP suite."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402
from test_pcfb import K_PASSES, TRIALS_PER_PASS, _identity_stream


@pytest.fixture(scope="module", autouse=True)
def _cuda_backend():
    if fuzzybits._runtime.is_initialized() and fuzzybits.active_backend() != "cuda":
        pytest.skip("runtime already initialised on a non-cuda backend")
    if not fuzzybits.has_cuda_device():
        pytest.skip("no CUDA device")
    fuzzybits.init(backend="cuda")
    yield


@pytest.mark.gpu
def test_bit_identity_chinese_cuda(chinese_xml: Path):
    pcfb, mem = _identity_stream(chinese_xml, k=K_PASSES)
    assert pcfb == mem
    assert int(pcfb[0][2]) >= TRIALS_PER_PASS
