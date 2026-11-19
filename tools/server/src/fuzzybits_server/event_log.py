from __future__ import annotations

import json
from typing import Any

from fuzzybits_server.config import env_bool


def log_events_to_stdout() -> bool:
    return env_bool("MCSCRAM_LOG_EVENTS", False)


def emit(event: str, **payload: Any) -> None:
    if not log_events_to_stdout():
        return
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)
