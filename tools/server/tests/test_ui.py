"""Playground UI at ``/``; ``--no-ui`` / ``FUZZYBITS_SERVER_UI=0`` disables it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcscram_server.app import create_app
from mcscram_server.config import apply_server_cli, server_ui_enabled


def _write_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>fuzzybits · Playground</title></head>"
        '<body><h1>Playground</h1><script type="module" src="/assets/index-test.js"></script></body></html>\n',
        encoding="utf-8",
    )
    (assets / "index-test.js").write_text("console.log('playground');\n", encoding="utf-8")
    return dist


def test_root_serves_index_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = _write_dist(tmp_path)
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "1")
    monkeypatch.setenv("FUZZYBITS_SERVER_UI_DIR", str(dist))
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Playground" in response.text
    assert response.headers.get("cache-control") == "no-cache"

    playground = client.get("/playground")
    assert playground.status_code == 200
    assert "Playground" in playground.text

    asset = client.get("/assets/index-test.js")
    assert asset.status_code == 200
    assert "playground" in asset.text
    assert "immutable" in asset.headers.get("cache-control", "")

    health = client.get("/api/v1/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_no_ui_env_returns_api_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "0")
    monkeypatch.delenv("FUZZYBITS_SERVER_UI_DIR", raising=False)
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api"] == "/api/v1"
    assert payload["ui"] is False
    health = client.get("/api/v1/healthz")
    assert health.status_code == 200


def test_no_ui_cli_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUZZYBITS_SERVER_UI", raising=False)
    apply_server_cli(["--no-ui"])
    assert server_ui_enabled() is False
