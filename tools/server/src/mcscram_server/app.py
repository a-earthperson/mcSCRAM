from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from mcscram_server.fault_tree_payload import build_fault_tree_payload, resolve_model_xml_path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _input_files_from_env() -> list[str]:
    raw = os.getenv("MCSCRAM_INPUT_FILES", "/app/input/SmallTree.xml")
    return [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]


def _options_from_env() -> dict[str, Any]:
    options: dict[str, Any] = {
        # num_trials=0 removes the fixed iteration limit; early_stop=false keeps
        # sampling after convergence, so the calculation is intentionally long-lived.
        "num_trials": _env_float("MCSCRAM_NUM_TRIALS", 0.0),
        "early_stop": _env_bool("MCSCRAM_EARLY_STOP", False),
        "burn_in": _env_float("MCSCRAM_BURN_IN", 0.0),
        "delta": _env_float("MCSCRAM_DELTA", 0.05),
        "policy": os.getenv("MCSCRAM_POLICY", "bayes"),
    }

    if oracle := os.getenv("MCSCRAM_ORACLE"):
        options["oracle"] = float(oracle)

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
    def from_env(cls) -> "CalculationConfig":
        return cls(input_files=_input_files_from_env(), options=_options_from_env())


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


class EventHub:
    def __init__(self, max_replay_events: int = 10_000) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_replay_events)
        self._recent_lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket, initial_status: dict[str, Any], replay: bool = False) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json({"event": "snapshot", "status": initial_status})
        if replay:
            with self._recent_lock:
                recent = list(self._recent)
            for event in recent:
                await websocket.send_json(event)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    def publish_from_thread(self, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish(event)))

    async def _publish(self, event: dict[str, Any]) -> None:
        with self._recent_lock:
            self._recent.append(event)
        stale: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(event)
            except RuntimeError:
                stale.append(client)
        for client in stale:
            self.disconnect(client)

    def clear_replay(self) -> None:
        with self._recent_lock:
            self._recent.clear()


def _log_events_to_stdout() -> bool:
    return _env_bool("MCSCRAM_LOG_EVENTS", False)


def _emit(event: str, **payload: Any) -> None:
    if not _log_events_to_stdout():
        return
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def _calculation_main(config: CalculationConfig, events: mp.Queue) -> None:
    import mcscram_native

    sequence = 0
    _emit("calculation_started", input_files=config.input_files, options=config.options)

    def on_progress(update: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        message = {
            "sequence": sequence,
            "timestamp": time.time(),
            "update": update,
        }
        _emit("progress", **message)
        events.put({"type": "progress", **message})

    try:
        result = mcscram_native.run_probability(config.input_files, config.options, on_progress)
    except BaseException as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _emit("calculation_failed", error=error)
        events.put({"type": "failed", "error": error})
        raise

    _emit("calculation_completed", result=result)
    events.put({"type": "completed", "result": result})


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
            target=_calculation_main,
            args=(config, events),
            name="mcscram-calculation",
        )
        self._process.start()
        process = self._process
        self.state.mark_started(self._process.pid or -1)
        _emit("server_started_calculation", pid=self._process.pid, input_files=config.input_files)
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

        self._drainer = threading.Thread(target=self._drain_events, args=(generation, events, process), daemon=True)
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
                return {
                    "accepted": False,
                    "reason": "calculation already running",
                    "status": self.state.snapshot(alive=True),
                }
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


app = FastAPI(title="mcSCRAM Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
event_hub = EventHub()
supervisor = CalculationSupervisor(event_hub)


@app.on_event("startup")
async def start_calculation() -> None:
    event_hub.bind_loop(asyncio.get_running_loop())
    if _env_bool("MCSCRAM_AUTOSTART", True):
        supervisor.start(CalculationConfig.from_env())


@app.on_event("shutdown")
def stop_calculation() -> None:
    supervisor.stop()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Any]:
    return supervisor.state.snapshot(alive=supervisor.alive())


@app.get("/model/fault-tree")
def fault_tree_model(model_id: str | None = Query(default=None)) -> dict[str, Any]:
    if model_id:
        try:
            path = resolve_model_xml_path(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return build_fault_tree_payload([str(path)])
    return build_fault_tree_payload(CalculationConfig.from_env().input_files)


@app.post("/reset")
def reset_calculation() -> dict[str, Any]:
    return supervisor.restart(CalculationConfig.from_env())


@app.post("/start")
def start_sampling() -> dict[str, Any]:
    return supervisor.start_sampling(CalculationConfig.from_env())


@app.post("/stop")
def stop_sampling() -> dict[str, Any]:
    return supervisor.stop_sampling()


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    replay = websocket.query_params.get("replay", "").lower() in {"1", "true", "yes", "on"}
    await event_hub.connect(websocket, supervisor.state.snapshot(alive=supervisor.alive()), replay=replay)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(websocket)


def main() -> None:
    uvicorn.run(
        "mcscram_server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
