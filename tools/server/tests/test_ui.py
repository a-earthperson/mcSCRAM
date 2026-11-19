"""Playground UI at ``/``; ``--no-ui`` / ``FUZZYBITS_SERVER_UI=0`` disables it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fuzzybits_server.app import create_app
from fuzzybits_server.config import apply_server_cli, server_ui_dir, server_ui_enabled


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
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8000")
    apply_server_cli(["--no-ui"])
    assert server_ui_enabled() is False


def test_packaged_ui_served_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wheel install: env unset, importlib.resources copy is used."""
    dist = _write_dist(tmp_path)
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "1")
    monkeypatch.delenv("FUZZYBITS_SERVER_UI_DIR", raising=False)
    monkeypatch.setattr("fuzzybits_server.config.packaged_ui_dir", lambda: dist)
    monkeypatch.setattr("fuzzybits_server.config.source_ui_dir", lambda: None)
    assert server_ui_dir() == dist
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "Playground" in response.text
    asset = client.get("/assets/index-test.js")
    assert asset.status_code == 200
    assert "playground" in asset.text
    health = client.get("/api/v1/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_env_ui_dir_wins_over_packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packaged = _write_dist(tmp_path / "packaged")
    env_dist = _write_dist(tmp_path / "env")
    (env_dist / "index.html").write_text(
        "<!doctype html><html><body>from-env</body></html>\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FUZZYBITS_SERVER_UI", "1")
    monkeypatch.setenv("FUZZYBITS_SERVER_UI_DIR", str(env_dist))
    monkeypatch.setattr("fuzzybits_server.config.packaged_ui_dir", lambda: packaged)
    assert server_ui_dir() == env_dist
    client = TestClient(create_app())
    assert "from-env" in client.get("/").text


def test_mcscram_server_alias_imports() -> None:
    import fuzzybits_server
    import mcscram_server

    assert mcscram_server.create_app is fuzzybits_server.create_app
    from mcscram_server.config import server_ui_enabled as alias_fn
    from fuzzybits_server.config import server_ui_enabled as real_fn

    assert alias_fn is real_fn
