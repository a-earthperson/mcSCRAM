"""nombrilo.circuit compile / export (skip if the native bindings are missing)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402


def _nombrilo_dir() -> Path | None:
    env = os.environ.get("FUZZYBITS_NOMBRILO_DIR")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    repo = Path(__file__).resolve().parents[2]
    path = repo / "input" / "aralia-nombrilo"
    return path if path.is_dir() else None


def test_safetensors_ignores_unknown_metadata_keys(tmp_path: Path):
    safetensors = pytest.importorskip("safetensors")
    from safetensors import safe_open

    xml = Path(__file__).resolve().parents[2] / "input" / "SmallTree" / "SmallTree.xml"
    if not xml.is_file():
        pytest.skip("SmallTree.xml not in the source tree")
    pcfb = fuzzybits.compile(xml, pdag_c=2, no_kn=True)
    blob = fuzzybits.pcfb_to_nombrilo(pcfb, extras=True)
    path = tmp_path / "small.safetensors"
    path.write_bytes(blob)
    with safe_open(str(path), framework="np") as f:
        meta = f.metadata() or {}
        keys = set(f.keys())
    assert meta.get("format") == "nombrilo.circuit"
    assert "fuzzybits.pcfb.queries" in meta
    assert "operators" in keys
    assert "offsets" in keys
    assert "children" in keys


def test_compile_detects_nombrilo_header():
    root = _nombrilo_dir()
    if root is None:
        pytest.skip("FUZZYBITS_NOMBRILO_DIR unset and input/aralia-nombrilo absent")
    path = root / "chinese.safetensors"
    if not path.is_file():
        pytest.skip("chinese.safetensors missing from nombrilo dataset")
    blob = fuzzybits.compile(path)
    summary = fuzzybits.inspect(blob)
    assert summary["diagnostics"] == []
    assert summary["metadata"].get("general.source_format") == "nombrilo.circuit-1"
    assert int(summary["samples"]) == 25
    back = fuzzybits.pcfb_to_nombrilo(blob, extras=False)
    assert back[:8]  # non-empty safetensors
    restored = fuzzybits.nombrilo_to_pcfb(path)
    assert fuzzybits.inspect(restored)["diagnostics"] == []
