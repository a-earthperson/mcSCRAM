from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


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


def input_files_from_env() -> list[str]:
    raw = os.getenv("MCSCRAM_INPUT_FILES", "/app/input/SmallTree.xml")
    return [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]


def options_from_env() -> dict[str, Any]:
    options: dict[str, Any] = {}

    for env_name, key in (
        ("MCSCRAM_NUM_TRIALS", "num_trials"),
        ("MCSCRAM_OVERHEAD_RATIO", "overhead_ratio"),
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
