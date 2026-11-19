"""Cached / proxied benchmark history JSON."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from fuzzybits_server.app import create_app


def _write_index(root: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "v1",
        "generated_utc": "2026-11-19T00:00:00Z",
        "n_points": 0,
        "points": [],
    }
    (root / "index.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_bench_index_from_cache(tmp_path: Path, monkeypatch) -> None:
    _write_index(tmp_path)
    monkeypatch.setenv("FUZZYBITS_BENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "0")
    client = TestClient(create_app())
    response = client.get("/api/v1/bench/v1/index.json")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "v1"
    assert body["n_points"] == 0
    config = client.get("/api/v1/bench/config")
    assert config.status_code == 200
    assert config.json()["same_origin"] == "/api/v1/bench/v1/index.json"


def test_bench_rejects_path_escape(tmp_path: Path, monkeypatch) -> None:
    _write_index(tmp_path)
    monkeypatch.setenv("FUZZYBITS_BENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "0")
    client = TestClient(create_app())
    response = client.get("/api/v1/bench/v1/../../secret")
    assert response.status_code in {404, 422}
