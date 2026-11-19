"""Subprocess entry point for mcSCRAM probability calculations.

By default the convergence loop and all statistics run **on the host in Python**
(:mod:`fuzzybits_server.convergence`); the native engine is used only to produce
raw per-pass popcounts.  Set ``MCSCRAM_CONVERGENCE_ENGINE=native`` to fall back
to the legacy in-engine C++ convergence loop (``mcscram_native.run_probability``).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any

from fuzzybits_server.config import CalculationConfig
from fuzzybits_server.convergence import run_calculation
from fuzzybits_server.event_log import emit
from fuzzybits_server.native import format_native_exception, run_probability_call


def _convergence_engine() -> str:
    return os.getenv("MCSCRAM_CONVERGENCE_ENGINE", "python").strip().lower()


def calculation_main(config: CalculationConfig, events: mp.Queue) -> None:
    sequence = 0
    engine = _convergence_engine()
    emit(
        "calculation_started",
        input_files=config.input_files,
        options=config.options,
        engine=engine,
    )

    def on_progress(update: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        message = {
            "sequence": sequence,
            "timestamp": time.time(),
            "update": update,
        }
        emit("progress", **message)
        events.put({"type": "progress", **message})

    try:
        if engine == "native":
            result = run_probability_call(config.input_files, config.options, on_progress)
        else:
            result = run_calculation(
                config.input_files,
                config.options,
                on_progress,
                pcfb_path=config.pcfb_path,
            )
    except BaseException as exc:
        error = format_native_exception(exc)
        emit("calculation_failed", error=error)
        events.put({"type": "failed", "error": error})
        return

    emit("calculation_completed", result=result)
    events.put({"type": "completed", "result": result})
