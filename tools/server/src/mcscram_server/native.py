"""Isolated wrappers around mcscram_native calls.

AdaptiveCpp and CUDA failures can abort the interpreter via ``std::terminate``,
which Python exception handlers cannot catch. Compilation therefore runs in a
single long-lived child worker so the API server survives device errors and
concurrent requests do not stampede the GPU allocator.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class CompileProduct(str, Enum):
    GRAPH = "graph"
    KERNEL_GRAPH = "kernel_graph"
    EXECUTION_PLAN = "execution_plan"
    SAMPLE_SHAPE = "sample_shape"
    BUNDLE = "bundle"


@dataclass(frozen=True)
class NativeResult:
    ok: bool
    payload: Any = None
    error: str | None = None


_COMPILE_CACHE_MAX = 8


def format_native_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def import_native() -> Any:
    import mcscram_native  # type: ignore[import-not-found]

    return mcscram_native


def native_import_error() -> str | None:
    if importlib.util.find_spec("mcscram_native") is None:
        return "mcscram_native extension is not installed"
    return None


def run_probability_call(
    input_files: list[str],
    options: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Run Monte Carlo probability analysis in the current process.

    Intended for the dedicated calculation subprocess only. Device failures may
    still abort this process; the API server stays isolated via
    :mod:`mcscram_server.worker`.
    """
    native = import_native()
    return native.run_probability(input_files, options, on_progress)


def _native_timeout_seconds() -> float:
    raw = os.getenv("MCSCRAM_NATIVE_TIMEOUT_SEC", "600")
    try:
        return float(raw)
    except ValueError:
        return 600.0


def _compile_cache_key(input_files: list[str], options: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(input_files), tuple(sorted(options.items())))


def _silence_native_stderr() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        pass


def _payload_from_pdag(pdag: Any, product: str) -> Any:
    match product:
        case CompileProduct.GRAPH.value:
            return pdag.to_graph()
        case CompileProduct.KERNEL_GRAPH.value:
            return pdag.to_kernel_graph()
        case CompileProduct.EXECUTION_PLAN.value:
            return dict(pdag.to_execution_plan())
        case CompileProduct.SAMPLE_SHAPE.value:
            layer_manager = pdag.create_layer_manager()
            return {"sample_shape": dict(layer_manager.sample_shape)}
        case CompileProduct.BUNDLE.value:
            return {
                "compiled": pdag.to_graph(),
                "kernel": pdag.to_kernel_graph(),
            }
        case _:
            raise ValueError(f"unknown compile product: {product}")


def _persistent_compile_worker(request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    _silence_native_stderr()
    compile_cache: dict[tuple[Any, ...], Any] = {}
    cache_order: list[tuple[Any, ...]] = []

    while True:
        request = request_queue.get()
        if request is None:
            return

        request_id: int = request["request_id"]
        input_files: list[str] = request["input_files"]
        options: dict[str, Any] = request["options"]
        product: str = request["product"]
        cache_key = _compile_cache_key(input_files, options)

        try:
            if cache_key not in compile_cache:
                native = import_native()
                compile_cache[cache_key] = native.run_compile(input_files, options)
                cache_order.append(cache_key)
                while len(cache_order) > _COMPILE_CACHE_MAX:
                    evicted = cache_order.pop(0)
                    compile_cache.pop(evicted, None)
            payload = _payload_from_pdag(compile_cache[cache_key], product)
            response_queue.put({"request_id": request_id, "ok": True, "payload": payload})
        except BaseException as exc:
            compile_cache.pop(cache_key, None)
            if cache_key in cache_order:
                cache_order.remove(cache_key)
            response_queue.put({
                "request_id": request_id,
                "ok": False,
                "error": format_native_exception(exc),
            })


def _result_from_queue_message(message: dict[str, Any] | None) -> NativeResult:
    if message is None:
        return NativeResult(ok=False, error="native compile worker produced no result")
    if not message.get("ok", False):
        return NativeResult(ok=False, error=str(message.get("error", "native compile failed")))
    return NativeResult(ok=True, payload=message.get("payload"))


class NativeCompileBroker:
    """Owns one compile worker process and serializes all native compile calls."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._request_queue: mp.Queue | None = None
        self._response_queue: mp.Queue | None = None
        self._worker: mp.Process | None = None
        self._next_request_id = 0

    def stop(self) -> None:
        with self._lock:
            self._shutdown_worker()

    def run(
        self,
        input_files: list[str],
        options: dict[str, Any] | None,
        product: CompileProduct,
        *,
        timeout: float | None = None,
    ) -> NativeResult:
        timeout_seconds = _native_timeout_seconds() if timeout is None else timeout
        with self._lock:
            if not self._ensure_worker():
                return NativeResult(ok=False, error="native compile worker failed to start")

            assert self._request_queue is not None
            assert self._response_queue is not None
            self._next_request_id += 1
            request_id = self._next_request_id
            self._request_queue.put({
                "request_id": request_id,
                "input_files": input_files,
                "options": options or {},
                "product": product.value,
            })

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                worker = self._worker
                if worker is None or not worker.is_alive():
                    self._shutdown_worker()
                    return NativeResult(
                        ok=False,
                        error="native compile worker crashed (device or native runtime error)",
                    )
                try:
                    message = self._response_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if message.get("request_id") != request_id:
                    continue
                return _result_from_queue_message(message)

            self._shutdown_worker()
            return NativeResult(
                ok=False,
                error=f"native compile timed out after {timeout_seconds:g}s",
            )

    def _ensure_worker(self) -> bool:
        if self._worker is not None and self._worker.is_alive():
            return True
        self._shutdown_worker()
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._worker = self._ctx.Process(
            target=_persistent_compile_worker,
            args=(self._request_queue, self._response_queue),
            name="mcscram-native-compile",
        )
        self._worker.start()
        return self._worker.is_alive()

    def _shutdown_worker(self) -> None:
        worker = self._worker
        request_queue = self._request_queue
        self._worker = None
        self._request_queue = None
        self._response_queue = None

        if request_queue is not None:
            try:
                request_queue.put(None)
            except (OSError, ValueError):
                pass

        if worker is None:
            return
        if worker.is_alive():
            worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)


_broker = NativeCompileBroker()


def run_compile_isolated(
    input_files: list[str],
    options: dict[str, Any] | None,
    product: CompileProduct,
    *,
    timeout: float | None = None,
) -> NativeResult:
    """Compile MEF input in the native worker and return a serializable product."""
    return _broker.run(input_files, options, product, timeout=timeout)


def shutdown_native_workers() -> None:
    """Stop background native workers during application shutdown."""
    _broker.stop()
