from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional data mount from fuzzybits-bench (CSVs, OpenPSA/Aralia XML).
# Historical run tables stay in that private repo; this tree never ships them.
SERVER_DATA_DIR_ENV = "FUZZYBITS_SERVER_DATA_DIR"
_LEGACY_DEFAULT_INPUT = Path("/app/input/SmallTree.xml")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        ("MCSCRAM_WATCH", "watch"),
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

    if policy := os.getenv("MCSCRAM_POLICY"):
        options["policy"] = policy

    if overrides := os.getenv("MCSCRAM_OPTIONS_JSON"):
        parsed = json.loads(overrides)
        if not isinstance(parsed, dict):
            raise ValueError("MCSCRAM_OPTIONS_JSON must decode to a JSON object")
        options.update(parsed)

    return options


@dataclass(frozen=True)
class CalculationConfig:
    input_files: list[str]
    options: dict[str, Any]

    @classmethod
    def from_env(cls) -> CalculationConfig:
        return cls(input_files=input_files_from_env(), options=options_from_env())

    def with_options(self, **overrides: Any) -> CalculationConfig:
        if not overrides:
            return self
        options = dict(self.options)
        options.update(overrides)
        return CalculationConfig(input_files=self.input_files, options=options)
