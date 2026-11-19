"""Stable Python API wrapping the native ``run_probability`` / device bindings."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import _runtime

Z95 = 1.959963984540054


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Point estimate and Wald interval from a Monte Carlo run."""

    estimate: float
    ci: tuple[float, float]
    std_err: float
    trials: int | None
    analysis_time: float
    warnings: str
    ci99: tuple[float, float] | None = None
    raw: Mapping[str, Any] | None = None

    def __float__(self) -> float:
        return float(self.estimate)


def list_devices() -> list[dict[str, Any]]:
    """Return AdaptiveCpp/SYCL devices visible under the current backend mask.

    Triggers :func:`fuzzybits.init` with defaults if the runtime is not yet
    initialised.
    """
    native = _runtime.native()
    devices = native.list_devices()
    return [dict(d) for d in devices]


def probability(
    model_xml: str | Path | Sequence[str | Path],
    *,
    trials: int | float | None = None,
    seed: int | None = None,
    pdag_c: int = 2,
    no_kn: bool = True,
    policy: str = "wald",
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    **options: Any,
) -> ProbabilityEstimate:
    """Run a Monte Carlo probability analysis on OpenPSA MEF XML.

    Wraps ``mcscram_native.run_probability`` (the existing engine). Extra
    keyword arguments are forwarded as CLI-like options (``delta``,
    ``early_stop``, ``compilation_level``, …). Results are deterministic for
    a given model and trial count (Philox counters); ``seed`` is not
    implemented.

    Parameters
    ----------
    model_xml:
        Path to a MEF XML file, or a sequence of input files.
    trials:
        Bernoulli trial count (``num_trials``). ``None`` uses the engine
        default (auto / early-stop).
    seed:
        Unused. Passing a value raises :class:`NotImplementedError`.
    pdag_c:
        PDAG compilation level (``-c`` / ``compilation_level``).
    no_kn:
        Match the CLI ``--no-kn`` flag (expand k-out-of-n to AND/OR).
    policy:
        Convergence interval policy (``"wald"`` or ``"bayes"``).
    """
    if seed is not None or "seed" in options:
        raise NotImplementedError(
            "fuzzybits.probability() does not accept seed: Monte Carlo results "
            "are deterministic for a given model and trial count (Philox "
            "counters). Seeded runs are tracked with fzb run --seed (#44)."
        )
    native = _runtime.native()
    if isinstance(model_xml, (str, Path)):
        input_files = [str(model_xml)]
    else:
        input_files = [str(p) for p in model_xml]
    if not input_files:
        raise ValueError("probability() requires at least one MEF XML path")

    opts: dict[str, Any] = {
        "compilation_level": pdag_c,
        "no_kn": no_kn,
        "policy": policy,
    }
    if trials is not None:
        opts["num_trials"] = float(trials)
        opts.setdefault("early_stop", False)
    opts.update(options)

    last_progress: dict[str, Any] = {}

    def _capture(payload: dict[str, Any]) -> None:
        last_progress.clear()
        last_progress.update(payload)
        if on_progress is not None:
            on_progress(payload)

    result = native.run_probability(input_files, opts, _capture)
    probs = list(result.get("probabilities") or [])
    if not probs:
        raise RuntimeError("native run_probability returned no probability results")
    p_total = float(probs[0]["p_total"])
    analysis_time = float(result.get("analysis_time") or probs[0].get("analysis_time") or 0.0)
    warnings = str(result.get("warnings") or "")

    mean = last_progress.get("mean", p_total)
    std_err = last_progress.get("std_err")
    trials_done = last_progress.get("trials")
    p05 = last_progress.get("p05")
    p95 = last_progress.get("p95")
    p01 = last_progress.get("p01")
    p99 = last_progress.get("p99")

    if std_err is None:
        n = float(trials_done or trials or 0.0)
        p = float(mean)
        std_err = (p * (1.0 - p) / n) ** 0.5 if n > 0 else float("nan")
    if p05 is None or p95 is None:
        hw = float(std_err) * Z95
        p05 = max(0.0, float(mean) - hw)
        p95 = min(1.0, float(mean) + hw)
    ci99: tuple[float, float] | None = None
    if p01 is not None and p99 is not None:
        ci99 = (float(p01), float(p99))

    return ProbabilityEstimate(
        estimate=float(mean),
        ci=(float(p05), float(p95)),
        std_err=float(std_err),
        trials=int(trials_done) if trials_done is not None else (int(trials) if trials else None),
        analysis_time=analysis_time,
        warnings=warnings,
        ci99=ci99,
        raw=dict(result),
    )


@dataclass(frozen=True)
class QueryResult:
    """One query from :func:`run` / ``fzb run``."""

    name: str
    estimate: float
    ci_low: float
    ci_high: float
    n: int
    hits: int


@dataclass(frozen=True)
class NodeTally:
    """Cumulative tally counters for one PDAG node after a device pass."""

    index: int
    num_one_bits: int
    total_bits: int

    @property
    def one_bits(self) -> int:
        return self.num_one_bits

    @property
    def popcount(self) -> int:
        return self.num_one_bits


@dataclass(frozen=True)
class PassResult:
    """One device pass from :class:`Session.passes`.

    ``nodes`` is keyed by PDAG index. Each value has the same ``num_one_bits`` /
    ``total_bits`` fields the bench server's ``NativeSampler`` reads from
    ``single_pass_all()``. Counters are cumulative across passes.
    """

    pass_index: int
    trials_per_pass: int
    target_index: int
    nodes: Mapping[int, NodeTally]


def _reject_seed(seed: int | None, options: Mapping[str, Any]) -> None:
    if seed is not None or "seed" in options:
        raise NotImplementedError(
            "seed is not accepted: Monte Carlo results are deterministic for a "
            "given model and trial count (Philox counters). Seeded runs are "
            "tracked with fzb run --seed (#65)."
        )


def _as_input_files(model_xml: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[str]:
    if isinstance(model_xml, (str, os.PathLike)):
        items: Sequence[str | os.PathLike[str]] = [model_xml]
    else:
        items = model_xml
    files: list[str] = []
    for item in items:
        text = os.fspath(item) if not isinstance(item, str) else item
        if not text:
            raise ValueError("compile/probability input path is empty")
        stripped = text.lstrip()
        if stripped.startswith("<") or stripped.startswith("<?xml"):
            raise ValueError(
                "compile() takes MEF XML file paths, not XML text; write the "
                "model to a file first"
            )
        files.append(text)
    if not files:
        raise ValueError("at least one MEF XML path is required")
    return files


def _pcfb_bytes(pcfb: bytes | str | os.PathLike[str]) -> bytes:
    if isinstance(pcfb, (bytes, bytearray, memoryview)):
        return bytes(pcfb)
    path = Path(pcfb)
    return path.read_bytes()


def _as_watch(value: Any) -> list[str]:
    """Coerce ``watch`` into MEF names / PDAG indices. Booleans are watch_mode, not a list."""
    if value is None or value is False or value is True:
        return []
    if isinstance(value, (bytes, bytearray)):
        raise TypeError("watch must be a name, index, or sequence of names")
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (int, float)):
        return [str(int(value))]
    return [str(item).strip() for item in value if str(item).strip()]


def _mask_backend(backend: str | None) -> str:
    """Translate Python ``backend=`` to the C++ ``apply_backend_mask`` argument."""
    chosen = "auto" if backend is None else str(backend).lower()
    if chosen in {"", "auto"}:
        _runtime.native()
        return ""
    if chosen not in {"cuda", "omp", "metal"}:
        raise ValueError(f"backend must be 'auto', 'cuda', 'omp', or 'metal' (got {backend!r})")
    if not _runtime.is_initialized():
        _runtime.init(backend=chosen)  # type: ignore[arg-type]
        return chosen
    active = _runtime.active_backend()
    if active not in {None, "", chosen} and chosen != active:
        raise RuntimeError(
            "fuzzybits.init() already ran; AdaptiveCpp cannot switch "
            f"backend after import (active backend={active!r}). "
            f"Start a new interpreter to use backend={chosen!r}."
        )
    return chosen


def compile(  # noqa: A001 - public API name
    model_xml: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    pdag_c: int = 2,
    no_kn: bool = True,
    watch: Any = None,
    **options: Any,
) -> bytes:
    """Compile OpenPSA/MEF XML to a pcfb 1.1 buffer.

    Parameters
    ----------
    model_xml:
        Path to a MEF XML file, or a sequence of input files.
    pdag_c:
        PDAG compilation level (``-c`` / ``compilation_level``).
    no_kn:
        Match the CLI ``--no-kn`` flag (expand k-out-of-n to AND/OR).
    watch:
        Extra MEF gate / event names (or PDAG indices) emitted as additional
        ``PROBABILITY`` queries so :class:`Session` tallies them. Boolean
        ``True``/``False`` is ignored (that is ``watch_mode``).
    """
    _reject_seed(options.pop("seed", None) if "seed" in options else None, options)
    native = _runtime.native()
    opts: dict[str, Any] = {
        "compilation_level": pdag_c,
        "no_kn": no_kn,
    }
    opts.update(options)
    names = _as_watch(watch)
    if not names:
        names = _as_watch(opts.pop("watch", None))
    else:
        opts.pop("watch", None)
    if names:
        opts["watch"] = names
    return bytes(native.compile_pcfb(_as_input_files(model_xml), opts))


def compile_file(
    model_xml: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    out_path: str | os.PathLike[str],
    *,
    pdag_c: int = 2,
    no_kn: bool = True,
    **options: Any,
) -> Path:
    """Compile MEF XML to ``out_path`` and return that path."""
    data = compile(model_xml, pdag_c=pdag_c, no_kn=no_kn, **options)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def inspect(pcfb: bytes | str | os.PathLike[str]) -> dict[str, Any]:
    """Summarise a ``.pcfb`` (bytes or path).

    Keys match ``fzb inspect --json``, plus ``diagnostics`` from
    ``pcfb::validate`` (empty list iff the circuit is core-1.1 conforming).
    """
    native = _runtime.native()
    if isinstance(pcfb, (bytes, bytearray, memoryview)):
        return dict(native.inspect_pcfb_bytes(bytes(pcfb)))
    return dict(native.inspect_pcfb_path(os.fspath(pcfb)))


def run(
    pcfb: bytes | str | os.PathLike[str],
    *,
    trials: int | float,
    backend: str = "auto",
    seed: int | None = None,
    **options: Any,
) -> list[QueryResult]:
    """One-shot Monte Carlo of a compiled ``.pcfb``. Same numbers as ``fzb run``.

    ``seed`` is rejected (#65). ``backend`` is ``"auto"``, ``"cuda"``,
    ``"omp"``, or ``"metal"``; ``"auto"`` keeps the mask :func:`fuzzybits.init`
    already applied. Metal SSCP currently rejects ``double`` until #17/#30.
    """
    _reject_seed(seed, options)
    if options:
        unknown = ", ".join(sorted(options))
        raise TypeError(f"run() got unexpected keyword argument(s): {unknown}")
    n = int(trials)
    if n <= 0:
        raise ValueError("run() requires a positive trial count")
    native = _runtime.native()
    rows = native.run_pcfb(_pcfb_bytes(pcfb), n, _mask_backend(backend))
    return [
        QueryResult(
            name=str(row["name"]),
            estimate=float(row["estimate"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            n=int(row["n"]),
            hits=int(row["hits"]),
        )
        for row in rows
    ]


def _pass_from_native(payload: Mapping[str, Any]) -> PassResult:
    nodes: dict[int, NodeTally] = {}
    raw_nodes = payload.get("nodes") or {}
    for index, counter in raw_nodes.items():
        idx = int(index)
        nodes[idx] = NodeTally(
            index=int(counter.get("index", idx)),
            num_one_bits=int(counter["num_one_bits"]),
            total_bits=int(counter["total_bits"]),
        )
    return PassResult(
        pass_index=int(payload["pass_index"]),
        trials_per_pass=int(payload["trials_per_pass"]),
        target_index=int(payload["target_index"]),
        nodes=nodes,
    )


class Session:
    """Streaming Monte Carlo over a compiled ``.pcfb``.

    Reconstructs the PDAG, builds one DirectEval layer manager, and yields the
    same per-pass popcount fields the bench server consumes today
    (``num_one_bits`` / ``total_bits``, pass index, trials per pass). The GIL is
    released during kernel execution. ``break`` out of :meth:`passes` or call
    :meth:`close` to stop early; prefer the context manager so device USM is
    released.

    ``graph()`` / ``execution_plan()`` / ``sample_shape`` use the reconstructed
    PDAG and have the same JSON-able keys as ``run_compile`` + ``Pdag.to_graph``
    / ``to_execution_plan`` / ``LayerManager.sample_shape``.
    """

    def __init__(
        self,
        pcfb: bytes | str | os.PathLike[str],
        *,
        backend: str = "auto",
        trials_per_pass: int | None = None,
        seed: int | None = None,
        watch: Any = None,
        num_trials: int | float | None = None,
    ) -> None:
        _reject_seed(seed, {})
        native = _runtime.native()
        tpp = None if trials_per_pass is None else int(trials_per_pass)
        n_trials = None if num_trials is None else int(num_trials)
        self._impl = native.PcfbSession(
            _pcfb_bytes(pcfb),
            _mask_backend(backend),
            tpp,
            _as_watch(watch),
            n_trials,
        )
        self._closed = False

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._impl, "close", None)
        if close is not None:
            close()
        self._impl = None

    @property
    def sample_shape(self) -> dict[str, Any]:
        self._require_open()
        return dict(self._impl.sample_shape)

    def graph(self) -> dict[str, Any]:
        """JSON-able reconstructed PDAG (``nodes``, ``edges``, ``top``, …)."""
        self._require_open()
        return dict(self._impl.graph())

    def kernel_graph(self) -> dict[str, Any]:
        """JSON-able kernel DAG (same keys as ``Pdag.to_kernel_graph``)."""
        self._require_open()
        return dict(self._impl.kernel_graph())

    def execution_plan(self) -> dict[str, Any]:
        """JSON-able DirectEval execution plan (same shape as ``Pdag.to_execution_plan``)."""
        self._require_open()
        return dict(self._impl.execution_plan())

    def step(self) -> PassResult:
        """Run exactly one device pass (same payload as one :meth:`passes` yield)."""
        self._require_open()
        return _pass_from_native(self._impl.step())

    def passes(self, max_passes: int | None = None) -> Iterator[PassResult]:
        """Yield one :class:`PassResult` per device pass until ``max_passes`` or :meth:`close`."""
        n = 0
        while max_passes is None or n < max_passes:
            self._require_open()
            yield _pass_from_native(self._impl.step())
            n += 1

    def _require_open(self) -> None:
        if self._closed or self._impl is None:
            raise RuntimeError("Session is closed")


__all__ = [
    "NodeTally",
    "PassResult",
    "ProbabilityEstimate",
    "QueryResult",
    "Session",
    "compile",
    "compile_file",
    "inspect",
    "list_devices",
    "probability",
    "run",
]
