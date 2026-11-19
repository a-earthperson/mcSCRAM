"""Data-dir env: absent mount must not crash; present mount is discovered."""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzybits_server.config import input_files_from_env, optional_table_path, server_data_dir
from fuzzybits_server.fault_tree_payload import aralia_model_search_roots, list_aralia_xml_inputs


def test_absent_data_dir_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUZZYBITS_SERVER_DATA_DIR", raising=False)
    monkeypatch.delenv("MCSCRAM_INPUT_FILES", raising=False)
    assert server_data_dir() is None
    assert optional_table_path("compiler.csv") is None
    assert optional_table_path("convergence.csv") is None
    files = input_files_from_env()
    assert files == [] or Path(files[0]).is_file()


def test_data_dir_discovers_csv_and_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCSCRAM_INPUT_FILES", raising=False)
    monkeypatch.delenv("MCSCRAM_ARALIA_INPUT_DIR", raising=False)
    (tmp_path / "compiler.csv").write_text("model,trials\n", encoding="utf-8")
    (tmp_path / "convergence.csv").write_text("model,mean\n", encoding="utf-8")
    xml = (
        '<?xml version="1.0"?><opsa-mef><define-fault-tree name="t"/>'
        "</opsa-mef>\n"
    )
    (tmp_path / "tiny.xml").write_text(xml, encoding="utf-8")
    (tmp_path / "SmallTree.xml").write_text(xml, encoding="utf-8")
    monkeypatch.setenv("FUZZYBITS_SERVER_DATA_DIR", str(tmp_path))

    assert server_data_dir() == tmp_path
    assert optional_table_path("compiler.csv") == tmp_path / "compiler.csv"
    assert optional_table_path("convergence.csv") == tmp_path / "convergence.csv"
    assert input_files_from_env() == [str(tmp_path / "SmallTree.xml")]
    roots = aralia_model_search_roots()
    assert tmp_path in roots
    catalog = list_aralia_xml_inputs()
    assert "tiny" in catalog["modelIds"]
    assert "smalltree" in catalog["modelIds"]


def test_missing_data_dir_path_is_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FUZZYBITS_SERVER_DATA_DIR", str(tmp_path / "does-not-exist"))
    assert server_data_dir() is None
    catalog = list_aralia_xml_inputs()
    assert isinstance(catalog["models"], list)
