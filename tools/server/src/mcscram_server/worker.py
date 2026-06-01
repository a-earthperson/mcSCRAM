"""Subprocess entry point for native mcSCRAM probability calculations."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from typing import Any

from mcscram_server.config import CalculationConfig
from mcscram_server.event_log import emit


def calculation_main(config: CalculationConfig, events: mp.Queue) -> None:
    import mcscram_native

    sequence = 0
    emit("calculation_started", input_files=config.input_files, options=config.options)

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
        result = mcscram_native.run_probability(config.input_files, config.options, on_progress)
    except BaseException as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        emit("calculation_failed", error=error)
        events.put({"type": "failed", "error": error})
        raise

    emit("calculation_completed", result=result)
    events.put({"type": "completed", "result": result})
