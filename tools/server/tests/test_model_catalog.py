"""Model catalog: OpenPSA XML, compiled .pcfb, and nombrilo .safetensors circuits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fuzzybits_server.app import create_app
from fuzzybits_server.fault_tree_payload import (
    MODEL_FORMAT_NOMBRILO,
    MODEL_FORMAT_PCFB,
    MODEL_FORMAT_XML,
    is_nombrilo_circuit,
    list_aralia_xml_inputs,
    resolve_model_input,
    resolve_model_xml_path,
)
from fuzzybits_server.native import nombrilo_weights_sibling

REPO = Path(__file__).resolve().parents[3]
TINY_XML = REPO / "tests" / "python" / "data" / "tiny.xml"
_XML = '<?xml version="1.0"?><opsa-mef><define-fault-tree name="t"/></opsa-mef>\n'


def _safetensors(path: Path, metadata: dict[str, str]) -> Path:
    header = json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    return path


def _nombrilo(path: Path) -> Path:
    return _safetensors(path, {"format": "nombrilo.circuit", "version": "1", "base_count": "0"})


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("MCSCRAM_INPUT_FILES", raising=False)
    monkeypatch.delenv("MCSCRAM_ARALIA_INPUT_DIR", raising=False)
    monkeypatch.setenv("FUZZYBITS_SERVER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FUZZYBITS_SERVER_CACHE_DIR", str(tmp_path / ".cache"))
    return tmp_path


def test_catalog_lists_xml_pcfb_and_nombrilo(data_dir: Path) -> None:
    (data_dir / "chinese.xml").write_text(_XML, encoding="utf-8")
    (data_dir / "chinese.pcfb").write_bytes(b"PCF1-placeholder")
    sub = data_dir / "circuits"
    sub.mkdir()
    _nombrilo(sub / "baobab1.safetensors")
    _nombrilo(sub / "baobab1.f3.safetensors")
    _safetensors(sub / "weights-only.safetensors", {"format": "pt"})

    catalog = list_aralia_xml_inputs()
    formats = {row["modelId"]: row["format"] for row in catalog["models"]}
    assert formats["chinese"] == MODEL_FORMAT_XML
    assert formats["chinese.pcfb"] == MODEL_FORMAT_PCFB
    assert formats["circuits/baobab1.safetensors"] == MODEL_FORMAT_NOMBRILO
    assert formats["circuits/baobab1.f3.safetensors"] == MODEL_FORMAT_NOMBRILO
    assert "circuits/weights-only.safetensors" not in formats
    assert catalog["modelIds"] == sorted(catalog["modelIds"])


def test_resolve_model_input_keeps_formats_apart(data_dir: Path) -> None:
    (data_dir / "chinese.xml").write_text(_XML, encoding="utf-8")
    (data_dir / "chinese.pcfb").write_bytes(b"PCF1-placeholder")

    assert resolve_model_input("chinese").format == MODEL_FORMAT_XML
    assert resolve_model_input("chinese.xml").path == data_dir / "chinese.xml"
    pcfb = resolve_model_input("Chinese.PCFB")
    assert pcfb.format == MODEL_FORMAT_PCFB
    assert pcfb.path == data_dir / "chinese.pcfb"
    assert resolve_model_xml_path("chinese") == data_dir / "chinese.xml"
    with pytest.raises(ValueError):
        resolve_model_xml_path("chinese.pcfb")
    with pytest.raises(FileNotFoundError):
        resolve_model_input("missing.pcfb")
    with pytest.raises(ValueError):
        resolve_model_input("../chinese.pcfb")


def test_nombrilo_detection_and_weights_sibling(tmp_path: Path) -> None:
    base = _nombrilo(tmp_path / "baobab1.safetensors")
    variant = _nombrilo(tmp_path / "baobab1.e1.safetensors")
    other = _safetensors(tmp_path / "model.safetensors", {"format": "pt"})
    (tmp_path / "truncated.safetensors").write_bytes(b"\x05\x00")

    assert is_nombrilo_circuit(base)
    assert not is_nombrilo_circuit(other)
    assert not is_nombrilo_circuit(tmp_path / "truncated.safetensors")
    assert nombrilo_weights_sibling(variant) == base
    assert nombrilo_weights_sibling(base) is None


def test_source_views_are_empty_for_compiled_models(data_dir: Path) -> None:
    (data_dir / "chinese.pcfb").write_bytes(b"PCF1-placeholder")
    client = TestClient(create_app())

    catalog = client.get("/api/v1/model/xml-inputs").json()
    assert "chinese.pcfb" in catalog["modelIds"]

    fault_tree = client.get("/api/v1/model/fault-tree", params={"model_id": "chinese.pcfb"})
    assert fault_tree.status_code == 200
    assert fault_tree.json()["nodes"] == []

    missing = client.get("/api/v1/model/fault-tree", params={"model_id": "nope.pcfb"})
    assert missing.status_code == 404


@pytest.fixture
def native_workers():
    from fuzzybits_server.native import shutdown_native_workers

    yield
    shutdown_native_workers()


@pytest.mark.native
@pytest.mark.usefixtures("native_workers")
def test_pcfb_and_nombrilo_models_compile_and_render(data_dir: Path) -> None:
    if not TINY_XML.is_file():
        pytest.skip(f"missing {TINY_XML}")
    import fuzzybits

    blob = fuzzybits.compile(TINY_XML, pdag_c=2, no_kn=True)
    (data_dir / "tiny.pcfb").write_bytes(blob)
    (data_dir / "tiny.safetensors").write_bytes(fuzzybits.pcfb_to_nombrilo(blob))

    # The context manager runs the app's shutdown hook, which stops the
    # isolated compile worker; without it the worker outlives pytest.
    with TestClient(create_app()) as client:
        catalog = client.get("/api/v1/model/xml-inputs").json()
        assert {"tiny.pcfb", "tiny.safetensors"} <= set(catalog["modelIds"])

        for model_id in ("tiny.pcfb", "tiny.safetensors"):
            response = client.get("/api/v1/model/sample-shape", params={"model_id": model_id})
            assert response.status_code == 200, response.text
            dag = client.get("/api/v1/model/dag", params={"model_id": model_id})
            assert dag.status_code == 200, dag.text

    cached = list((data_dir / ".cache").rglob("*.pcfb"))
    assert cached, "nombrilo model should be translated into the pcfb cache"
