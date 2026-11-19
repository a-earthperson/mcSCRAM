"""Isolated wrappers around mcscram_native calls.

AdaptiveCpp and CUDA failures can abort the interpreter via ``std::terminate``,
which Python exception handlers cannot catch. Compilation therefore runs in a
single long-lived child worker so the API server survives device errors and
concurrent requests do not stampede the GPU allocator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from fuzzybits_server.config import parse_watch_value, pcfb_cache_dir, server_engine


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


class InvalidPcfbError(ValueError):
    """Uploaded / loaded `.pcfb` failed `fuzzybits.inspect` / `pcfb::validate`."""

    def __init__(self, message: str, diagnostics: list[Any] | None = None, inspect: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []
        self.inspect = inspect or {}


_COMPILE_CACHE_MAX = 8

# Host-side keys that run_compile's settings_from_options must not see.
# `watch` as a name list is not watch_mode; booleans stay as watch_mode.
_HOST_ONLY_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "policy",
        "P",
        "early_stop",
        "max_passes",
        "emit_min_interval_s",
        "stream_nodes",
        "convergence_event_ids",
        "pcfb_path",
        "pcfb_id",
    }
)

# Options that change the lowered circuit (cache key + compile_pcfb).
_COMPILE_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "compilation_level",
        "compilation_passes",
        "pdag_c",
        "c",
        "no_kn",
        "expand_atleast_gates",
        "expand_kn",
        "no_xor",
        "expand_xor_gates",
        "recognize_atleast_groups",
        "recognize_xor_groups",
        "keep_null_gates",
        "ccf",
        "allow_extern",
        "target",
        "target_gate",
        "top",
        "top_event",
        "watch",
        "watch_nodes",
    }
)


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
    :mod:`fuzzybits_server.worker`.
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


def compile_option_subset(options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return {}
    out: dict[str, Any] = {}
    for key, value in options.items():
        if key in _COMPILE_OPTION_KEYS:
            out[key] = value
    return out


def watch_names_from_options(options: dict[str, Any] | None) -> list[str]:
    if not options:
        return []
    _, names = parse_watch_value(options.get("watch"))
    extra = options.get("watch_nodes")
    if extra:
        _, more = parse_watch_value(extra)
        names = names + more
    return names


def pcfb_cache_digest(input_files: list[str], options: dict[str, Any] | None) -> str:
    hasher = hashlib.sha256()
    for path in input_files:
        payload = Path(path).read_bytes()
        hasher.update(path.encode())
        hasher.update(b"\0")
        hasher.update(payload)
        hasher.update(b"\0")
    canon = json.dumps(compile_option_subset(options), sort_keys=True, default=str)
    hasher.update(canon.encode())
    return hasher.hexdigest()


def cached_pcfb_path(digest: str) -> Path:
    return pcfb_cache_dir() / f"{digest}.pcfb"


def compile_pcfb_to_cache(input_files: list[str], options: dict[str, Any] | None) -> bytes:
    """Compile MEF to `.pcfb`, keyed by source SHA-256s + compile options."""
    if not input_files:
        raise ValueError("compile requires at least one input file")
    digest = pcfb_cache_digest(input_files, options)
    path = cached_pcfb_path(digest)
    if path.is_file():
        return path.read_bytes()
    native = import_native()
    blob = bytes(native.compile_pcfb(input_files, compile_option_subset(options) or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pcfb.tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return blob


def inspect_pcfb_bytes(data: bytes) -> dict[str, Any]:
    try:
        import fuzzybits

        return dict(fuzzybits.inspect(data))
    except ImportError:
        native = import_native()
        return dict(native.inspect_pcfb_bytes(data))


def uploaded_pcfb_dir() -> Path:
    path = pcfb_cache_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_uploaded_pcfb(pcfb_id: str) -> Path:
    digest = pcfb_id.strip().lower().removeprefix("pcfb:")
    if not digest or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"invalid pcfb id: {pcfb_id}")
    path = uploaded_pcfb_dir() / f"{digest}.pcfb"
    if not path.is_file():
        path = cached_pcfb_path(digest)
    if not path.is_file():
        raise FileNotFoundError(f"no uploaded .pcfb with id {digest}")
    return path


def store_uploaded_pcfb(data: bytes) -> tuple[str, Path, dict[str, Any]]:
    if not data:
        raise InvalidPcfbError("empty pcfb upload")
    try:
        summary = inspect_pcfb_bytes(data)
    except BaseException as exc:
        raise InvalidPcfbError(f"invalid pcfb: {exc}") from exc
    diagnostics = list(summary.get("diagnostics") or [])
    if diagnostics:
        raise InvalidPcfbError("pcfb validation failed", diagnostics=diagnostics, inspect=summary)
    digest = hashlib.sha256(data).hexdigest()
    path = uploaded_pcfb_dir() / f"{digest}.pcfb"
    if not path.is_file():
        tmp = path.with_suffix(".pcfb.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return digest, path, summary


def _trials_per_pass(options: dict[str, Any] | None) -> int | None:
    if not options:
        return None
    raw = options.get("requested_trials_per_pass", options.get("trials_per_pass"))
    if raw is None:
        return None
    value = int(float(raw))
    return value if value > 0 else None


def _num_trials(options: dict[str, Any] | None) -> int | None:
    if not options:
        return None
    raw = options.get("num_trials")
    if raw is None:
        return None
    value = int(float(raw))
    return value if value > 0 else None


def _session_from_pcfb(native: Any, blob: bytes, options: dict[str, Any] | None) -> Any:
    return native.PcfbSession(
        blob,
        "",
        _trials_per_pass(options),
        watch_names_from_options(options),
        _num_trials(options),
    )


def _payload_from_pcfb_session(session: Any, product: str) -> Any:
    match product:
        case CompileProduct.GRAPH.value:
            return dict(session.graph())
        case CompileProduct.KERNEL_GRAPH.value:
            return dict(session.kernel_graph())
        case CompileProduct.EXECUTION_PLAN.value:
            return dict(session.execution_plan())
        case CompileProduct.SAMPLE_SHAPE.value:
            return {"sample_shape": dict(session.sample_shape)}
        case CompileProduct.BUNDLE.value:
            return {
                "compiled": dict(session.graph()),
                "kernel": dict(session.kernel_graph()),
            }
        case "pcfb":
            return None
        case _:
            raise ValueError(f"unknown compile product: {product}")


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
    pcfb_cache: dict[str, bytes] = {}

    while True:
        request = request_queue.get()
        if request is None:
            return

        request_id: int = request["request_id"]
        input_files: list[str] = request["input_files"]
        options: dict[str, Any] = request["options"]
        product: str = request["product"]
        engine: str = request.get("engine") or server_engine()
        pcfb_path: str | None = request.get("pcfb_path")
        cache_key = _compile_cache_key(input_files, options)

        try:
            if engine == "legacy" and not pcfb_path:
                if cache_key not in compile_cache:
                    native = import_native()
                    compile_opts = {
                        key: value
                        for key, value in options.items()
                        if key not in _HOST_ONLY_OPTION_KEYS and key not in {"watch", "watch_nodes"}
                    }
                    mode, _ = parse_watch_value(options.get("watch_mode", options.get("watch")))
                    if mode is not None:
                        compile_opts["watch_mode"] = mode
                    compile_cache[cache_key] = native.run_compile(input_files, compile_opts)
                    cache_order.append(cache_key)
                    while len(cache_order) > _COMPILE_CACHE_MAX:
                        evicted = cache_order.pop(0)
                        compile_cache.pop(evicted, None)
                payload = _payload_from_pdag(compile_cache[cache_key], product)
            else:
                native = import_native()
                if pcfb_path:
                    blob = Path(pcfb_path).read_bytes()
                else:
                    digest = pcfb_cache_digest(input_files, options)
                    blob = pcfb_cache.get(digest)
                    if blob is None:
                        blob = compile_pcfb_to_cache(input_files, options)
                        pcfb_cache[digest] = blob
                        while len(pcfb_cache) > _COMPILE_CACHE_MAX:
                            pcfb_cache.pop(next(iter(pcfb_cache)))
                session = _session_from_pcfb(native, blob, options)
                try:
                    payload = _payload_from_pcfb_session(session, product)
                    if product == "pcfb":
                        payload = blob
                finally:
                    close = getattr(session, "close", None)
                    if close is not None:
                        close()
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
        engine: str | None = None,
        pcfb_path: str | None = None,
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
                "engine": engine or server_engine(),
                "pcfb_path": pcfb_path,
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
            name="fuzzybits-native-compile",
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
    engine: str | None = None,
    pcfb_path: str | None = None,
) -> NativeResult:
    """Compile MEF input in the native worker and return a serializable product."""
    return _broker.run(
        input_files,
        options,
        product,
        timeout=timeout,
        engine=engine,
        pcfb_path=pcfb_path,
    )


def shutdown_native_workers() -> None:
    """Stop background native workers during application shutdown."""
    _broker.stop()
