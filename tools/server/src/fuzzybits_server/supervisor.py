from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from typing import Any

from fuzzybits_server.config import CalculationConfig
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.event_log import emit
from fuzzybits_server.fault_tree_payload import build_fault_tree_payload
from fuzzybits_server.state import CalculationState
from fuzzybits_server.worker import calculation_main

_FINISHED_EXIT_GRACE_S = 5.0


class CalculationSupervisor:
    def __init__(self, event_hub: EventHub) -> None:
        self._ctx = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._drainer: threading.Thread | None = None
        self._generation = 0
        self._restart_lock = threading.Lock()
        self.state = CalculationState()
        self._event_hub = event_hub

    def start(self, config: CalculationConfig) -> None:
        with self._restart_lock:
            self._start_locked(config)

    def _start_locked(self, config: CalculationConfig) -> None:
        if self._process and self._process.is_alive():
            return

        events: mp.Queue = self._ctx.Queue()
        self._generation += 1
        generation = self._generation
        self._process = self._ctx.Process(
            target=calculation_main,
            args=(config, events),
            name="fuzzybits-calculation",
        )
        self._process.start()
        process = self._process
        self.state.mark_started(self._process.pid or -1)
        emit("server_started_calculation", pid=self._process.pid, input_files=config.input_files)
        self._event_hub.publish_from_thread({
            "event": "server_started_calculation",
            "pid": self._process.pid,
            "input_files": config.input_files,
            "options": config.options,
        })
        self._event_hub.publish_from_thread({
            "event": "model_loaded",
            "model": build_fault_tree_payload(config.input_files),
        })

        self._drainer = threading.Thread(
            target=self._drain_events,
            args=(generation, events, process),
            daemon=True,
        )
        self._drainer.start()

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)

    def restart(self, config: CalculationConfig) -> dict[str, Any]:
        if not self._restart_lock.acquire(blocking=False):
            return {
                "accepted": False,
                "reason": "restart already in progress",
                "status": self.state.snapshot(alive=self.alive()),
            }
        try:
            self._generation += 1
            self.stop()
            self._event_hub.clear_replay()
            self._event_hub.publish_from_thread({"event": "calculation_reset"})
            self._start_locked(config)
            return {
                "accepted": True,
                "status": self.state.snapshot(alive=self.alive()),
            }
        finally:
            self._restart_lock.release()

    def start_sampling(self, config: CalculationConfig) -> dict[str, Any]:
        if not self._restart_lock.acquire(blocking=False):
            return {
                "accepted": False,
                "reason": "control operation already in progress",
                "status": self.state.snapshot(alive=self.alive()),
            }
        try:
            if self._process and self._process.is_alive():
                if self.state.snapshot()["status"] == "running":
                    return {
                        "accepted": False,
                        "reason": "calculation already running",
                        "status": self.state.snapshot(alive=True),
                    }
                # The previous calculation reported its result but its process
                # is still exiting (worker and runtime teardown).
                self._process.join(timeout=_FINISHED_EXIT_GRACE_S)
                self.stop()
            self._generation += 1
            self._event_hub.clear_replay()
            self._event_hub.publish_from_thread({"event": "calculation_reset"})
            self._start_locked(config)
            return {
                "accepted": True,
                "status": self.state.snapshot(alive=self.alive()),
            }
        finally:
            self._restart_lock.release()

    def stop_sampling(self) -> dict[str, Any]:
        if not self._restart_lock.acquire(blocking=False):
            return {
                "accepted": False,
                "reason": "control operation already in progress",
                "status": self.state.snapshot(alive=self.alive()),
            }
        try:
            if not (self._process and self._process.is_alive()):
                self.state.mark_stopped()
                return {
                    "accepted": False,
                    "reason": "calculation is not running",
                    "status": self.state.snapshot(alive=False),
                }
            self._generation += 1
            self.stop()
            self.state.mark_stopped()
            status = self.state.snapshot(alive=False)
            self._event_hub.publish_from_thread({
                "event": "calculation_stopped",
                "status": status,
            })
            return {
                "accepted": True,
                "status": status,
            }
        finally:
            self._restart_lock.release()

    def alive(self) -> bool | None:
        return self._process.is_alive() if self._process else None

    def _drain_events(self, generation: int, events: mp.Queue, process: mp.Process) -> None:
        while True:
            if generation != self._generation:
                return

            if not process.is_alive() and events.empty():
                if process.exitcode not in (0, None) and self.state.snapshot()["status"] == "running":
                    error = f"calculation process exited with code {process.exitcode}"
                    self.state.mark_failed(error)
                    self._event_hub.publish_from_thread({
                        "event": "calculation_failed",
                        "error": error,
                    })
                return

            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                continue

            if generation != self._generation:
                return

            match event.get("type"):
                case "progress":
                    self.state.record_progress(event["update"])
                    self._event_hub.publish_from_thread({
                        "event": "progress",
                        "sequence": event["sequence"],
                        "timestamp": event["timestamp"],
                        "update": event["update"],
                    })
                case "completed":
                    self.state.mark_complete(event["result"])
                    self._event_hub.publish_from_thread({
                        "event": "calculation_completed",
                        "result": event["result"],
                    })
                case "failed":
                    self.state.mark_failed(event["error"])
                    self._event_hub.publish_from_thread({
                        "event": "calculation_failed",
                        "error": event["error"],
                    })
