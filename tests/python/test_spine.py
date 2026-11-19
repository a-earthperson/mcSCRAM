"""Hosted-CPU spine: compile → inspect → run chinese within 4·σ̂ of the BDD ref."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

CHINESE_REF = 1.17058e-3
Z_MAX = 4.0
# 2^20 trials: chinese finishes in seconds on a hosted CPU and 4σ is ~11% of p.
SPINE_TRIALS = 1 << 20


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


def test_spine_chinese_omp_within_4sigma():
    repo = Path(__file__).resolve().parents[2]
    xml = repo / "input" / "Aralia" / "chinese.xml"
    if not xml.is_file():
        pytest.skip(f"missing {xml}")

    pcfb = fuzzybits.compile([str(xml)])
    summary = fuzzybits.inspect(pcfb)
    assert summary.get("diagnostics") == []

    rows = fuzzybits.run(pcfb, trials=SPINE_TRIALS, backend="omp")
    assert rows
    row = rows[0]
    assert row.n >= SPINE_TRIALS
    sigma = math.sqrt(max(row.estimate * (1.0 - row.estimate), 0.0) / float(row.n))
    abs_z = abs(row.estimate - CHINESE_REF) / sigma if sigma > 0 else float("inf")
    assert abs_z <= Z_MAX, (
        f"chinese p̂={row.estimate:.6e} p_ref={CHINESE_REF:.6e} "
        f"|z|={abs_z:.3f} n={row.n} bound={Z_MAX}"
    )
