"""pcfb compile/cache/upload path.

Pure-Python cases always run. Engine tests that need the fuzzybits wheel are
marked ``@pytest.mark.native`` and skipped by ``conftest.py`` when the
extension is absent (hosted ``server.yml``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mcscram_server.config import parse_watch_value, pcfb_cache_dir, server_engine
from mcscram_server.native import (
    InvalidPcfbError,
    compile_option_subset,
    compile_pcfb_to_cache,
    pcfb_cache_digest,
    store_uploaded_pcfb,
    watch_names_from_options,
)

REPO = Path(__file__).resolve().parents[3]
TINY_XML = REPO / "tests" / "python" / "data" / "tiny.xml"


def test_server_engine_defaults_to_pcfb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUZZYBITS_SERVER_ENGINE", raising=False)
    assert server_engine() == "pcfb"
    monkeypatch.setenv("FUZZYBITS_SERVER_ENGINE", "legacy")
    assert server_engine() == "legacy"
    monkeypatch.setenv("FUZZYBITS_SERVER_ENGINE", "PCFB")
    assert server_engine() == "pcfb"


def test_parse_watch_value_splits_mode_and_names() -> None:
    assert parse_watch_value(True) == (True, [])
    assert parse_watch_value("1") == (True, [])
    assert parse_watch_value("false") == (False, [])
    assert parse_watch_value("g1,TrainOne") == (None, ["g1", "TrainOne"])
    assert parse_watch_value(["g1", "  x2  "]) == (None, ["g1", "x2"])
    assert parse_watch_value(7) == (None, ["7"])


def test_watch_names_from_options_ignores_boolean_watch_mode() -> None:
    assert watch_names_from_options({"watch": True}) == []
    assert watch_names_from_options({"watch": "g1"}) == ["g1"]
    assert watch_names_from_options({"watch": ["g1"], "watch_nodes": "TrainOne"}) == ["g1", "TrainOne"]


def test_compile_option_subset_keeps_watch_out_of_host_keys() -> None:
    subset = compile_option_subset(
        {
            "compilation_level": 2,
            "no_kn": True,
            "watch": ["g1"],
            "policy": "gbas",
            "num_trials": 1e6,
        }
    )
    assert subset["compilation_level"] == 2
    assert subset["watch"] == ["g1"]
    assert "policy" not in subset
    assert "num_trials" not in subset


def test_pcfb_cache_digest_changes_with_bytes_and_options(tmp_path: Path) -> None:
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_bytes(b"<a/>")
    b.write_bytes(b"<b/>")
    d1 = pcfb_cache_digest([str(a)], {"compilation_level": 2})
    d2 = pcfb_cache_digest([str(b)], {"compilation_level": 2})
    d3 = pcfb_cache_digest([str(a)], {"compilation_level": 1})
    assert d1 != d2
    assert d1 != d3
    assert len(d1) == 64


@pytest.mark.native
def test_compile_pcfb_to_cache_and_inspect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not TINY_XML.is_file():
        pytest.skip(f"missing {TINY_XML}")
    monkeypatch.setenv("FUZZYBITS_SERVER_CACHE_DIR", str(tmp_path / "cache"))
    blob = compile_pcfb_to_cache([str(TINY_XML)], {"compilation_level": 2, "no_kn": True})
    assert blob[:4]
    digest = pcfb_cache_digest([str(TINY_XML)], {"compilation_level": 2, "no_kn": True})
    cached = pcfb_cache_dir() / f"{digest}.pcfb"
    assert cached.is_file()
    assert cached.read_bytes() == blob
    again = compile_pcfb_to_cache([str(TINY_XML)], {"compilation_level": 2, "no_kn": True})
    assert again == blob

    import fuzzybits

    summary = fuzzybits.inspect(blob)
    assert summary["diagnostics"] == []
    assert summary["queries"] >= 1


@pytest.mark.native
def test_store_uploaded_pcfb_rejects_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not TINY_XML.is_file():
        pytest.skip(f"missing {TINY_XML}")
    monkeypatch.setenv("FUZZYBITS_SERVER_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(InvalidPcfbError):
        store_uploaded_pcfb(b"")
    with pytest.raises(InvalidPcfbError):
        store_uploaded_pcfb(b"not-a-pcfb")

    import fuzzybits

    blob = fuzzybits.compile(TINY_XML, pdag_c=2, no_kn=True)
    digest, path, summary = store_uploaded_pcfb(blob)
    assert digest == hashlib.sha256(blob).hexdigest()
    assert path.is_file()
    assert summary["diagnostics"] == []
    again, path2, _ = store_uploaded_pcfb(blob)
    assert again == digest
    assert path2 == path


@pytest.mark.native
def test_pcfb_sampler_watch_and_legacy_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not TINY_XML.is_file():
        pytest.skip(f"missing {TINY_XML}")
    monkeypatch.setenv("FUZZYBITS_SERVER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FUZZYBITS_SERVER_ENGINE", "pcfb")

    from mcscram_server.convergence.sampling import NativeSampler, make_sampler

    options = {
        "compilation_level": 2,
        "no_kn": True,
        "requested_trials_per_pass": 1 << 14,
        "watch": ["TrainOne"],
    }
    files = [str(TINY_XML)]
    pcfb = make_sampler(files, options)
    monkeypatch.setenv("FUZZYBITS_SERVER_ENGINE", "legacy")
    legacy = NativeSampler(files, options)
    try:
        assert pcfb.shape.trials_per_pass == legacy.shape.trials_per_pass
        pcfb_obs = pcfb.step()
        legacy_obs = legacy.step()
        assert pcfb_obs.target_index in pcfb_obs.nodes
        assert legacy_obs.target_index in legacy_obs.nodes
        p_root = pcfb_obs.nodes[pcfb_obs.target_index]
        l_root = legacy_obs.nodes[legacy_obs.target_index]
        assert p_root.num_one_bits == l_root.num_one_bits
        assert p_root.total_bits == l_root.total_bits
        assert len(pcfb.node_indices) >= 2
        assert len(legacy.node_indices) >= 2
        assert len(pcfb_obs.nodes) >= 2
        assert len(legacy_obs.nodes) >= 2
        extra_pcfb = [c for idx, c in pcfb_obs.nodes.items() if idx != pcfb.target_index]
        extra_legacy = [c for idx, c in legacy_obs.nodes.items() if idx != legacy.target_index]
        assert extra_pcfb and extra_legacy
        extra_pcfb_bits = sorted((c.num_one_bits, c.total_bits) for c in extra_pcfb)
        extra_legacy_bits = sorted((c.num_one_bits, c.total_bits) for c in extra_legacy)
        assert extra_pcfb_bits == extra_legacy_bits
    finally:
        pcfb.close()
        legacy.close()
