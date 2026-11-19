"""Cheap CPU test: dataset v1 schema + checksums. No GPU, no fzb."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.loader import (  # noqa: E402
    NUS9601_REF,
    NUS9601_REF_SOURCE,
    TARGET_HITS,
    accuracy_trials,
    expected_hits,
    load_manifest,
    validate_manifest,
)

SCHEMA = ROOT / "benchmarks" / "schemas" / "dataset-manifest.v1.json"


def test_manifest_schema_file_exists():
    payload = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert payload["properties"]["schema_version"]["const"] == 1
    assert "models" in payload["required"]


def test_loader_validates_checksums_and_references():
    manifest = load_manifest(root=ROOT, check_files=True)
    assert manifest.dataset == "v1"
    assert manifest.schema_version == 1
    ids = {model.id for model in manifest.models}
    assert "chinese" in ids
    assert "das9201" in ids
    assert "nus9601" in ids
    aralia = [m for m in manifest.models if m.source == "aralia"]
    pwr = [m for m in manifest.models if m.source == "pwr"]
    assert len(aralia) == 43
    assert len(pwr) == 95
    accuracy = manifest.tagged("accuracy")
    assert len(accuracy) == 43
    for model in accuracy:
        assert model.reference.probability is not None
        assert 0.0 < model.reference.probability < 1.0
        assert model.trials_accuracy and model.trials_accuracy > 0
    perf = manifest.tagged("perf")
    assert {m.id for m in perf} >= {"das9201", "das9701", "nus9601", "edf9203", "chinese"}
    for model in perf:
        assert model.perf_reason
    validate_manifest(manifest, root=ROOT)


def test_nus9601_nombrillo_reference_and_trial_count():
    manifest = load_manifest(root=ROOT)
    model = manifest.by_id("nus9601")
    assert model.reference.probability == NUS9601_REF
    assert model.reference.source == NUS9601_REF_SOURCE
    assert model.reference.kind == "exact"
    assert model.reference.uncertainty == 0.0
    assert set(model.tags) == {"accuracy", "perf", "hard"}
    n = model.trials_accuracy
    assert n == accuracy_trials(NUS9601_REF)
    assert expected_hits(n, NUS9601_REF) >= TARGET_HITS * 0.99
    assert model.time_budget_s["cpu_accuracy"] >= 180


def test_accuracy_trials_match_target_hits_or_are_hard():
    manifest = load_manifest(root=ROOT)
    for model in manifest.tagged("accuracy"):
        p = float(model.reference.probability)
        n = int(model.trials_accuracy)
        hits = expected_hits(n, p)
        if hits < TARGET_HITS:
            assert model.has_tag("hard"), f"{model.id}: n·p={hits} without hard tag"
        else:
            assert n == accuracy_trials(p)


def test_pwr_known_compile_failures_stay_visible():
    manifest = load_manifest(root=ROOT)
    expected = {
        "EQK-BIN4/Grp-5": ("G3", "G10"),
        "EQK-BIN4/Grp-6": ("G3",),
        "EQK-BIN5/Grp-2": ("F44", "G3"),
    }
    for model_id, missing in expected.items():
        model = manifest.by_id(model_id)
        assert model.known_compile_failure
        assert model.compile_expected == "undefined_element"
        assert tuple(model.compile_missing) == missing
        assert model.compile_reason
        assert "Undefined Element" in model.compile_reason
    assert sum(1 for m in manifest.models if m.known_compile_failure) == 3


def test_rare_events_are_infeasible_not_vacuous_pass():
    manifest = load_manifest(root=ROOT)
    infeasible = {m.id: m for m in manifest.tagged("accuracy") if m.accuracy.check == "infeasible"}
    assert set(infeasible) == {"das9204", "das9205", "das9209", "edf9206"}
    for model in infeasible.values():
        assert model.accuracy.reason
        assert not model.accuracy.gated
    isp = manifest.by_id("isp9607")
    assert isp.accuracy.check == "low_power"
    assert isp.accuracy.gated
    chinese = manifest.by_id("chinese")
    assert chinese.accuracy.check == "feasible"
    nus = manifest.by_id("nus9601")
    assert nus.accuracy.check == "feasible"
