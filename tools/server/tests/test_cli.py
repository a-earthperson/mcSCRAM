"""CLI parsing for ``fuzzybits-server``."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from fuzzybits_server.config import (
    SERVER_CACHE_DIR_ENV,
    SERVER_DATA_DIR_ENV,
    SERVER_ENGINE_ENV,
    SERVER_UI_ENV,
    apply_server_cli,
    parse_server_cli,
    server_engine,
    server_ui_enabled,
)
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.supervisor import CalculationSupervisor
from fuzzybits_server.native import NativeCompileBroker


def test_cli_defaults() -> None:
    args = parse_server_cli([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.data_dir is None
    assert args.cache_dir is None
    assert args.backend == "auto"
    assert args.engine is None
    assert args.no_ui is False
    assert args.open is False


def test_cli_parses_all_flags(tmp_path: Path) -> None:
    data = tmp_path / "models"
    cache = tmp_path / "cache"
    args = parse_server_cli(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--data-dir",
            str(data),
            "--cache-dir",
            str(cache),
            "--backend",
            "omp",
            "--engine",
            "legacy",
            "--no-ui",
            "--open",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.data_dir == str(data)
    assert args.cache_dir == str(cache)
    assert args.backend == "omp"
    assert args.engine == "legacy"
    assert args.no_ui is True
    assert args.open is True


def test_cli_rejects_bad_backend() -> None:
    with pytest.raises(SystemExit):
        parse_server_cli(["--backend", "rocm"])


def test_apply_server_cli_writes_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv(SERVER_DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(SERVER_CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(SERVER_ENGINE_ENV, raising=False)
    monkeypatch.delenv(SERVER_UI_ENV, raising=False)
    monkeypatch.delenv("ACPP_VISIBILITY_MASK", raising=False)
    monkeypatch.delenv("FUZZYBITS_BACKEND", raising=False)
    data = tmp_path / "models"
    data.mkdir()
    cache = tmp_path / "cache"
    apply_server_cli(
        [
            "--host",
            "10.0.0.1",
            "--port",
            "8123",
            "--data-dir",
            str(data),
            "--cache-dir",
            str(cache),
            "--backend",
            "metal",
            "--engine",
            "legacy",
            "--no-ui",
        ]
    )
    assert __import__("os").environ["HOST"] == "10.0.0.1"
    assert __import__("os").environ["PORT"] == "8123"
    assert __import__("os").environ[SERVER_DATA_DIR_ENV] == str(data)
    assert __import__("os").environ[SERVER_CACHE_DIR_ENV] == str(cache)
    assert __import__("os").environ["ACPP_VISIBILITY_MASK"] == "metal"
    assert __import__("os").environ["FUZZYBITS_BACKEND"] == "metal"
    assert server_engine() == "legacy"
    assert server_ui_enabled() is False


def test_apply_backend_auto_does_not_override_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACPP_VISIBILITY_MASK", "omp")
    monkeypatch.delenv("FUZZYBITS_BACKEND", raising=False)
    apply_server_cli(["--backend", "auto"])
    assert __import__("os").environ["ACPP_VISIBILITY_MASK"] == "omp"
    assert "FUZZYBITS_BACKEND" not in __import__("os").environ


def test_supervisor_and_broker_use_spawn() -> None:
    supervisor = CalculationSupervisor(EventHub())
    assert supervisor._ctx.get_start_method() == "spawn"
    broker = NativeCompileBroker()
    assert broker._ctx.get_start_method() == "spawn"
    assert mp.get_context("spawn").get_start_method() == "spawn"
