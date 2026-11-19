from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional data mount from fuzzybits-bench (CSVs, OpenPSA/Aralia XML).
# Historical run tables stay in that private repo; this tree never ships them.
SERVER_DATA_DIR_ENV = "FUZZYBITS_SERVER_DATA_DIR"
SERVER_ENGINE_ENV = "FUZZYBITS_SERVER_ENGINE"
SERVER_CACHE_DIR_ENV = "FUZZYBITS_SERVER_CACHE_DIR"
SERVER_UI_ENV = "FUZZYBITS_SERVER_UI"
SERVER_UI_DIR_ENV = "FUZZYBITS_SERVER_UI_DIR"
_LEGACY_DEFAULT_INPUT = Path("/app/input/SmallTree.xml")
_BOOLEAN_TRUE = {"1", "true", "yes", "on"}
_BOOLEAN_FALSE = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _BOOLEAN_TRUE


def server_engine() -> str:
    """Compile/sample engine: ``pcfb`` (default) or ``legacy`` in-memory PDAG."""
    raw = os.getenv(SERVER_ENGINE_ENV, "pcfb").strip().lower()
    if raw in {"legacy", "memory", "in-memory", "run_compile"}:
        return "legacy"
    return "pcfb"


def parse_watch_value(raw: Any) -> tuple[bool | None, list[str]]:
    """Split ``watch`` into (watch_mode, names).

    ``MCSCRAM_WATCH=1`` is the C++ progress-bar flag. Any other non-boolean
    string is a comma-separated MEF name / PDAG-index list, matching how
    :mod:`mcscram_server.convergence.sampling` registers extra tally nodes.
    """
    if raw is None or raw is False:
        return None, []
    if raw is True:
        return True, []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None, [str(int(raw))]
    if isinstance(raw, (list, tuple, set)):
        return None, [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return None, []
    lowered = text.lower()
    if lowered in _BOOLEAN_TRUE:
        return True, []
    if lowered in _BOOLEAN_FALSE:
        return False, []
    return None, [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def pcfb_cache_dir() -> Path:
    raw = os.getenv(SERVER_CACHE_DIR_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        data = server_data_dir()
        path = (data / ".pcfb-cache") if data is not None else Path.home() / ".cache" / "fuzzybits" / "server-pcfb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def server_data_dir() -> Path | None:
    """Directory of optional bench data (CSVs, model XML).

    Set ``FUZZYBITS_SERVER_DATA_DIR`` to a fuzzybits-bench checkout or mount.
    Returns ``None`` when unset or not a directory — the server still starts;
    model listing is empty and default input files are omitted.
    """
    raw = os.getenv(SERVER_DATA_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def optional_table_path(name: str) -> Path | None:
    """Locate ``compiler.csv`` / ``convergence.csv`` under the data dir.

    These tables are not imported into this public repo. fuzzybits-bench can
    mount its copies via ``FUZZYBITS_SERVER_DATA_DIR``.
    """
    data = server_data_dir()
    if data is None:
        return None
    path = data / name
    return path if path.is_file() else None


def input_files_from_env() -> list[str]:
    raw = os.getenv("MCSCRAM_INPUT_FILES")
    if raw is not None:
        return [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]

    data = server_data_dir()
    if data is not None:
        for candidate in (
            data / "SmallTree.xml",
            data / "input" / "SmallTree.xml",
            data / "input" / "SmallTree" / "SmallTree.xml",
        ):
            if candidate.is_file():
                return [str(candidate)]

    if _LEGACY_DEFAULT_INPUT.is_file():
        return [str(_LEGACY_DEFAULT_INPUT)]
    return []


def options_from_env() -> dict[str, Any]:
    options: dict[str, Any] = {}

    for env_name, key in (
        ("MCSCRAM_NUM_TRIALS", "num_trials"),
        ("MCSCRAM_REQUESTED_TRIALS_PER_PASS", "requested_trials_per_pass"),
        ("MCSCRAM_DELTA", "delta"),
        ("MCSCRAM_BURN_IN", "burn_in"),
        ("MCSCRAM_CONFIDENCE", "confidence"),
        ("MCSCRAM_ORACLE", "oracle"),
        ("MCSCRAM_MISSION_TIME", "mission_time"),
        ("MCSCRAM_TIME_STEP", "time_step"),
        ("MCSCRAM_CUT_OFF", "cut_off"),
    ):
        if raw := os.getenv(env_name):
            options[key] = float(raw)

    for env_name, key in (
        ("MCSCRAM_SEED", "seed"),
        ("MCSCRAM_LIMIT_ORDER", "limit_order"),
        ("MCSCRAM_COMPILATION_PASSES", "compilation_passes"),
    ):
        if raw := os.getenv(env_name):
            options[key] = int(raw)

    for env_name, key in (
        ("MCSCRAM_EARLY_STOP", "early_stop"),
        ("MCSCRAM_NO_KN", "no_kn"),
        ("MCSCRAM_NO_XOR", "no_xor"),
        ("MCSCRAM_RECOGNIZE_ATLEAST_GROUPS", "recognize_atleast_groups"),
        ("MCSCRAM_RECOGNIZE_XOR_GROUPS", "recognize_xor_groups"),
        ("MCSCRAM_PREPROCESSOR", "preprocessor"),
        ("MCSCRAM_PRINT", "print"),
        ("MCSCRAM_IMPORTANCE", "importance"),
        ("MCSCRAM_UNCERTAINTY", "uncertainty"),
        ("MCSCRAM_CCF", "ccf"),
        ("MCSCRAM_SIL", "sil"),
    ):
        if raw := os.getenv(env_name):
            options[key] = raw.strip().lower() in {"1", "true", "yes", "on"}

    if watch_raw := os.getenv("MCSCRAM_WATCH"):
        mode, names = parse_watch_value(watch_raw)
        if mode is not None:
            options["watch_mode"] = mode
        if names:
            options["watch"] = names
    if watch_nodes := os.getenv("MCSCRAM_WATCH_NODES"):
        _, names = parse_watch_value(watch_nodes)
        if names:
            existing = [str(item) for item in options.get("watch", [])] if isinstance(options.get("watch"), list) else []
            options["watch"] = existing + names

    if policy := os.getenv("MCSCRAM_POLICY"):
        options["policy"] = policy

    if overrides := os.getenv("MCSCRAM_OPTIONS_JSON"):
        parsed = json.loads(overrides)
        if not isinstance(parsed, dict):
            raise ValueError("MCSCRAM_OPTIONS_JSON must decode to a JSON object")
        options.update(parsed)

    return options



def apply_server_cli(argv: list[str] | None = None) -> None:
    """Honor ``--no-ui`` by setting ``FUZZYBITS_SERVER_UI=0``."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-ui", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    if args.no_ui:
        os.environ[SERVER_UI_ENV] = "0"


def server_ui_enabled() -> bool:
    return env_bool(SERVER_UI_ENV, True)


def server_ui_dir() -> Path | None:
    """Directory of the CSR playground build (``index.html`` + ``assets/``).

    ``FUZZYBITS_SERVER_UI_DIR`` wins. Otherwise ``tools/server/webui/dist``
    relative to this source tree (repo checkout and the server image). A
    later wheel (#86) can point the env at package data.
    """
    raw = os.getenv(SERVER_UI_DIR_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path(__file__).resolve().parents[2] / "webui" / "dist"
    index = path / "index.html"
    return path if path.is_dir() and index.is_file() else None


@dataclass(frozen=True)
class CalculationConfig:
    input_files: list[str]
    options: dict[str, Any]
    pcfb_path: str | None = None

    @classmethod
    def from_env(cls) -> CalculationConfig:
        return cls(input_files=input_files_from_env(), options=options_from_env())

    def with_options(self, **overrides: Any) -> CalculationConfig:
        if not overrides:
            return self
        options = dict(self.options)
        options.update(overrides)
        return CalculationConfig(
            input_files=self.input_files,
            options=options,
            pcfb_path=self.pcfb_path,
        )
