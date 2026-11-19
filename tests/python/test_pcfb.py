"""CPU tests for the pcfb Python API (compile / inspect / run / Session).

Bit-identity: the per-pass popcount stream from
``Session(fuzzybits.compile(xml)).passes()`` matches the in-memory
``run_compile`` + ``layer_manager.single_pass_all`` path the bench server uses.
Keep trial counts small so the module finishes well under 60 s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

K_PASSES = 2
TRIALS_PER_PASS = 1 << 14
ONESHOT_TRIALS = 1 << 14


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


def _compile_opts() -> dict:
    return {"compilation_level": 2, "no_kn": True, "requested_trials_per_pass": TRIALS_PER_PASS}


def _identity_stream(xml: Path, k: int = K_PASSES) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Return (pcfb_stream, memory_stream) of (pass_index, num_one_bits, total_bits)."""
    native = fuzzybits.native()
    pdag = native.run_compile([str(xml)], _compile_opts())
    manager = pdag.create_layer_manager()
    mem: list[tuple[int, int, int]] = []
    root = int(manager.root_index)
    for i in range(1, k + 1):
        raw = manager.single_pass_all()
        counter = raw[root]
        mem.append((i, int(counter["num_one_bits"]), int(counter["total_bits"])))

    blob = fuzzybits.compile(xml, pdag_c=2, no_kn=True)
    pcfb: list[tuple[int, int, int]] = []
    with fuzzybits.Session(blob, backend="auto", trials_per_pass=TRIALS_PER_PASS) as session:
        mem_shape = dict(manager.sample_shape)
        assert int(session.sample_shape["trials_per_pass"]) == int(mem_shape["trials_per_pass"])
        for obs in session.passes(max_passes=k):
            tally = obs.nodes[obs.target_index]
            pcfb.append((obs.pass_index, tally.num_one_bits, tally.total_bits))
    return pcfb, mem


def test_compile_inspect_run_tiny(tiny_xml: Path, tmp_path: Path):
    blob = fuzzybits.compile(tiny_xml, pdag_c=2, no_kn=True)
    assert isinstance(blob, (bytes, bytearray))
    assert blob[:4]  # non-empty FlatBuffer

    summary = fuzzybits.inspect(blob)
    assert summary["diagnostics"] == []
    assert summary["queries"] >= 1
    assert summary["samples"] >= 1
    assert "schema_version" in summary
    assert "opcodes" in summary
    assert "metadata" in summary
    meta = summary["metadata"]
    assert meta.get("general.generator") == "fuzzybits"
    assert "compile.options" in meta
    assert "compile.level" in meta
    assert "fuzzybits.origins" in meta
    assert "fuzzybits.tensor_pdag" in meta

    out = tmp_path / "tiny.pcfb"
    written = fuzzybits.compile_file(tiny_xml, out, pdag_c=2, no_kn=True)
    assert written == out
    assert out.is_file()
    assert fuzzybits.inspect(out)["diagnostics"] == []

    results = fuzzybits.run(blob, trials=ONESHOT_TRIALS, backend="omp")
    assert len(results) == 1
    row = results[0]
    assert 0.0 < row.estimate < 1.0
    assert row.ci_low <= row.estimate <= row.ci_high
    assert row.n >= ONESHOT_TRIALS
    assert row.hits <= row.n


def test_session_graph_and_plan(tiny_xml: Path):
    blob = fuzzybits.compile(tiny_xml)
    with fuzzybits.Session(blob, backend="omp", trials_per_pass=TRIALS_PER_PASS) as session:
        shape = session.sample_shape
        assert int(shape["trials_per_pass"]) > 0
        assert int(shape["bits_per_bitpack"]) == 64
        graph = session.graph()
        assert graph["nodes"]
        assert graph["edges"]
        assert graph["top"] is not None
        plan = session.execution_plan()
        assert "partitions" in plan
        assert "sample_shaper" in plan
        assert "semantic_graph" in plan
        first = next(session.passes(max_passes=1))
        assert first.pass_index == 1
        assert first.trials_per_pass == int(shape["trials_per_pass"])
        assert first.target_index in first.nodes
        tally = first.nodes[first.target_index]
        assert tally.total_bits == first.trials_per_pass
        assert 0 <= tally.num_one_bits <= tally.total_bits
        assert tally.one_bits == tally.num_one_bits
        assert tally.popcount == tally.num_one_bits


def test_session_close_and_break(tiny_xml: Path):
    blob = fuzzybits.compile(tiny_xml)
    session = fuzzybits.Session(blob, backend="omp", trials_per_pass=TRIALS_PER_PASS)
    seen = 0
    for _ in session.passes(max_passes=8):
        seen += 1
        break
    assert seen == 1
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(session.passes(max_passes=1))


def test_bit_identity_chinese(chinese_xml: Path):
    pcfb, mem = _identity_stream(chinese_xml)
    assert pcfb == mem
    assert len(pcfb) == K_PASSES


def test_bit_identity_das9201(das9201_xml: Path):
    pcfb, mem = _identity_stream(das9201_xml)
    assert pcfb == mem
    assert len(pcfb) == K_PASSES


def test_compile_seed_rejected(tiny_xml: Path):
    with pytest.raises(NotImplementedError, match="deterministic"):
        fuzzybits.compile(tiny_xml, seed=7)


def test_compile_watch_emits_extra_query(tiny_xml: Path):
    blob = fuzzybits.compile(tiny_xml, pdag_c=2, no_kn=True, watch=["TrainOne"])
    summary = fuzzybits.inspect(blob)
    assert summary["diagnostics"] == []
    assert summary["queries"] >= 2
    names = [str(n) for n in summary["query_names"]]
    assert "TrainOne" in names


def test_session_graph_roundtrips_names(tiny_xml: Path):
    blob = fuzzybits.compile(tiny_xml, pdag_c=2, no_kn=True)
    native = fuzzybits.native()
    live = native.run_compile([str(tiny_xml)], {"compilation_level": 2, "no_kn": True})
    live_graph = live.to_graph()
    with fuzzybits.Session(blob, backend="omp") as session:
        graph = session.graph()
    live_names = {
        node.get("id") or node.get("name")
        for node in live_graph["nodes"]
        if node.get("id") or node.get("name")
    }
    session_names = {
        node.get("id") or node.get("name")
        for node in graph["nodes"]
        if node.get("id") or node.get("name")
    }
    assert live_names
    assert live_names <= session_names
    assert graph["compilation"]["level"] == 2
    assert graph["compilation"]["expand_atleast_gates"] is True
    assert graph["source"]
    assert "TopEvent" in {graph.get("target"), graph.get("target_name")}


def test_session_watch_tallies_named_gate(tiny_xml: Path):
    blob = fuzzybits.compile(tiny_xml, pdag_c=2, no_kn=True)
    with fuzzybits.Session(blob, backend="omp", trials_per_pass=TRIALS_PER_PASS, watch=["TrainOne"]) as session:
        first = session.step()
        assert first.target_index in first.nodes
        named = [
            int(node["index"])
            for node in session.graph()["nodes"]
            if node.get("id") == "TrainOne" or node.get("name") == "TrainOne"
        ]
        assert named
        assert named[0] in first.nodes
        assert first.nodes[named[0]].total_bits == first.trials_per_pass
