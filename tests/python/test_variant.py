"""Native-module variant selection (no Monte Carlo, CPU-only)."""

from __future__ import annotations

import sys

import pytest

from fuzzybits._runtime import _select_backend, _select_variant


def test_auto_falls_back_to_generic_without_sm70(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_sm70_device", lambda: False)
    assert _select_variant("auto", {"generic": "g", "cuda-sm70": "c"}) == "generic"


def test_auto_selects_cuda_sm70_when_device_present(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_sm70_device", lambda: True)
    assert _select_variant("auto", {"generic": "g", "cuda-sm70": "c"}) == "cuda-sm70"


def test_auto_uses_only_installed_variant(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_sm70_device", lambda: False)
    assert _select_variant("auto", {"cuda-sm70": "c"}) == "cuda-sm70"


def test_explicit_variant_requires_install():
    with pytest.raises(ImportError, match="cuda-sm70"):
        _select_variant("cuda-sm70", {"generic": "g"})


def test_backend_auto_cuda_if_nvidia(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_cuda_device", lambda: True)
    assert _select_backend("auto") == "cuda"
    monkeypatch.setattr(rt, "has_cuda_device", lambda: False)
    if sys.platform == "darwin":
        assert _select_backend("auto") == ""
    else:
        assert _select_backend("auto") == "omp"
    assert _select_backend("omp") == "omp"
    assert _select_backend("cuda") == "cuda"
    assert _select_backend("metal") == "metal"
