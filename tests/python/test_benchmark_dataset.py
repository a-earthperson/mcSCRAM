"""Cheap CPU test: dataset v1 schema + checksums. No GPU, no fzb."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xml.etree import ElementTree as ET  # noqa: E402

from benchmarks.datasets.compat import (  # noqa: E402
    SHIM_ID,
    SHIM_VERSION,
    matching_jsinp,
    patch_generic_pwr,
)
from benchmarks.loader import (  # noqa: E402
    NUS9601_REF,
    NUS9601_REF_SOURCE,
    TARGET_HITS,
    accuracy_trials,
    expected_hits,
    load_manifest,
    sha256_file,
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


def test_pwr_models_use_generic_pwr_shim():
    manifest = load_manifest(root=ROOT)
    pwr = [m for m in manifest.models if m.source == "pwr"]
    assert len(pwr) == 95
    for model in pwr:
        assert model.compat_shim == SHIM_ID
        assert model.compat_version == SHIM_VERSION
    for model_id in ("EQK-BIN4/Grp-5", "EQK-BIN5/Grp-2"):
        model = manifest.by_id(model_id)
        assert not model.known_compile_failure
        assert model.compile_expected == "ok"
    remaining = manifest.by_id("EQK-BIN4/Grp-6")
    assert remaining.known_compile_failure
    assert remaining.compile_expected == "validity_error"
    assert "FE56" in remaining.compile_missing
    assert "FE57" in remaining.compile_missing
    assert "fork paths" in remaining.compile_reason
    assert sum(1 for m in manifest.models if m.known_compile_failure) == 1


def test_generic_pwr_shim_injects_g10_into_eqk_bin4_grp5(tmp_path):
    xml = ROOT / (
        "input/generic-pwr-model/data/openpsa/"
        "converted_EQK-BIN4_et_Grp-5_24-02-26_15-58-09.xml"
    )
    assert xml.is_file()
    before = sha256_file(xml)
    result = patch_generic_pwr(xml, cache_dir=tmp_path)
    assert xml.is_file()
    assert sha256_file(xml) == before
    assert result.original == xml
    assert result.path != xml
    assert result.path.is_file()
    assert "FT51.G10" in result.injected_gates
    assert any(name.endswith(".G3") for name in result.injected_gates)
    root_el = ET.parse(result.path).getroot()
    ft51 = next(el for el in root_el.findall("define-fault-tree") if el.attrib.get("name") == "FT51")
    gates = {el.attrib["name"]: el for el in ft51.findall("define-gate")}
    assert "G10" in gates
    children = [c.attrib["name"] for c in gates["G10"].find("or").findall("gate")]
    assert children == ["G15", "G19", "G24", "G27", "G39", "G40"]
    # Children already existed; the shim must not drop private clones.
    for name in children:
        assert name in gates
    jsinp = matching_jsinp(xml)
    assert jsinp.name.endswith(".JSInp")
    cached = patch_generic_pwr(xml, cache_dir=tmp_path)
    assert cached.path == result.path


def test_generic_pwr_shim_rewrites_f44_and_clones_g3(tmp_path):
    grp2 = ROOT / (
        "input/generic-pwr-model/data/openpsa/"
        "converted_EQK-BIN5_et_Grp-2_24-02-26_15-58-15.xml"
    )
    result = patch_generic_pwr(grp2, cache_dir=tmp_path)
    assert ("F44", "FE44") in result.rewritten_functional_events
    forks = [
        el.attrib.get("functional-event")
        for el in ET.parse(result.path).getroot().iter("fork")
    ]
    assert "F44" not in forks
    assert "FE44" in forks
    grp6 = ROOT / (
        "input/generic-pwr-model/data/openpsa/"
        "converted_EQK-BIN4_et_Grp-6_24-02-26_15-58-10.xml"
    )
    cloned = patch_generic_pwr(grp6, cache_dir=tmp_path)
    assert any(name.endswith(".G3") and not name.startswith("FT3.") for name in cloned.injected_gates)
    root_el = ET.parse(cloned.path).getroot()
    ft4 = next(el for el in root_el.findall("define-fault-tree") if el.attrib.get("name") == "FT4")
    assert any(el.attrib.get("name") == "G3" for el in ft4.findall("define-gate"))



def test_generic_pwr_shim_leaves_existing_private_clones(tmp_path):
    xml = ROOT / (
        "input/generic-pwr-model/data/openpsa/"
        "converted_EQK-BIN4_et_Grp-3_24-02-26_15-58-03.xml"
    )
    before = ET.parse(xml).getroot()
    n_g43 = sum(1 for el in before.iter("define-gate") if el.attrib.get("name") == "G43")
    assert n_g43 > 1
    result = patch_generic_pwr(xml, cache_dir=tmp_path)
    after = ET.parse(result.path).getroot()
    n_after = sum(1 for el in after.iter("define-gate") if el.attrib.get("name") == "G43")
    assert n_after == n_g43
    extra_g43 = [name for name in result.injected_gates if name.endswith(".G43")]
    assert extra_g43 == []


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
