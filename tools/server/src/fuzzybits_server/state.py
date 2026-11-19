from __future__ import annotations

import threading
import time
from typing import Any


class CalculationState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "not_started"
        self.pid: int | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.update_count = 0
        self.last_update: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None

    def snapshot(self, alive: bool | None = None) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "pid": self.pid,
                "alive": alive,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "update_count": self.update_count,
                "last_update": self.last_update,
                "last_result": self.last_result,
                "last_error": self.last_error,
            }

    def mark_started(self, pid: int) -> None:
        with self._lock:
            self.status = "running"
            self.pid = pid
            self.started_at = time.time()
            self.finished_at = None
            self.update_count = 0
            self.last_update = None
            self.last_result = None
            self.last_error = None

    def record_progress(self, update: dict[str, Any]) -> None:
        with self._lock:
            self.update_count += 1
            self.last_update = update

    def mark_complete(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.status = "completed"
            self.finished_at = time.time()
            self.last_result = result

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.finished_at = time.time()
            self.last_error = error

    def mark_stopped(self) -> None:
        with self._lock:
            self.status = "stopped"
            self.finished_at = time.time()
            self.pid = None
            self.last_error = None
