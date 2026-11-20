"""Samplers: the boundary between the host engine and the bitstream producer.

A :class:`Sampler` is the *only* thing the convergence engine needs from the
outside world.  Each :meth:`Sampler.step` runs exactly one device pass and
returns a :class:`PassObservation` of raw cumulative counters.  Two
implementations are provided:

* :class:`NativeSampler` — wraps the ``fuzzybits`` native extension
  (``run_compile`` -> ``Pdag.create_layer_manager`` -> ``single_pass``).  It
  prefers the multi-node ``single_pass_all`` binding when present and otherwise
  falls back to the single-event ``single_pass`` (root only), so the engine
  works against existing native builds.
* :class:`BernoulliSampler` — a pure-Python binomial simulator used to unit-test
  the entire engine (policies, metrics, runner) without a GPU/SYCL build.

Because the engine programs only against the protocol, swapping samplers (real
device, replay, simulation) requires no changes anywhere else.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fuzzybits_server.config import server_engine
from fuzzybits_server.native import compile_pcfb_to_cache, watch_names_from_options

from .tally import NodeCounters, PassObservation


@dataclass(frozen=True, slots=True)
class SampleShape:
    """Device sampling geometry for a run."""

    batch_size: int
    bitpacks_per_batch: int
    bits_per_bitpack: int
    trials_per_pass: int
    total_iterations: int

    @classmethod
    def from_native(cls, shape: dict[str, Any]) -> "SampleShape":
        bits_per_bitpack = int(shape.get("bits_per_bitpack", 64))
        batch_size = int(shape.get("batch_size", 0))
        bitpacks_per_batch = int(shape.get("bitpacks_per_batch", 0))
        trials_per_pass = int(
            shape.get("trials_per_pass", batch_size * bitpacks_per_batch * bits_per_bitpack)
        )
        return cls(
            batch_size=batch_size,
            bitpacks_per_batch=bitpacks_per_batch,
            bits_per_bitpack=bits_per_bitpack,
            trials_per_pass=trials_per_pass,
            total_iterations=int(shape.get("total_iterations", 0)),
        )


# Convergence is decided host-side (see :mod:`.runner`/:mod:`.policies`), so these
# keys are consumed by the Python engine and must NOT be forwarded to the native
# compiler -- which validates them against its own (wald|bayes) ci-policy set and
# would reject host-only values such as "gbas".
_HOST_ONLY_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "policy",
        "P",
        "early_stop",
        "max_passes",
        "emit_min_interval_s",
        "stream_nodes",
        "convergence_event_ids",
        "watch",
        "watch_nodes",
        "watch_mode",
        "pcfb_path",
        "pcfb_id",
    }
)


def _native_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Strip host-only convergence keys before handing options to ``run_compile``."""
    if not options:
        return {}
    return {key: value for key, value in options.items() if key not in _HOST_ONLY_OPTION_KEYS}


@runtime_checkable
class Sampler(Protocol):
    """Produces raw cumulative tally counters, one step (one or more device passes) at a time."""

    @property
    def shape(self) -> SampleShape: ...

    @property
    def target_index(self) -> int: ...

    @property
    def node_indices(self) -> list[int]: ...

    def step(self, passes: int = 1) -> PassObservation:
        """Run ``passes`` passes, waiting once, and return cumulative counters for tracked nodes."""
        ...

    def close(self) -> None:
        """Release any underlying native resources."""
        ...


def _watch_event_ids(pdag: Any, options: dict[str, Any] | None, event_ids: list[int] | None) -> list[int]:
    ids = list(event_ids or [])
    names = watch_names_from_options(options)
    if not names:
        return ids
    graph = pdag.to_graph()
    for want in names:
        found: int | None = None
        for node in graph.get("nodes") or []:
            index = int(node["index"])
            if want == str(index) or want in {
                str(node.get("id") or ""),
                str(node.get("name") or ""),
                str(node.get("full_path") or ""),
            }:
                found = index
                break
        if found is None:
            raise ValueError(f"watched node not in compiled PDAG: {want}")
        if found not in ids:
            ids.append(found)
    return ids


class NativeSampler:
    """Adapts the ``fuzzybits`` layer manager to the :class:`Sampler` protocol."""

    def __init__(
        self,
        input_files: list[str],
        options: dict[str, Any] | None = None,
        *,
        event_ids: list[int] | None = None,
    ) -> None:
        import fuzzybits  # imported lazily so the engine imports without the ext

        self._pdag = fuzzybits.native().run_compile(input_files, _native_options(options))
        requested = list(event_ids or [])
        extra = _watch_event_ids(self._pdag, options, [])
        if requested or extra:
            root = int(self._pdag.to_graph()["top"])
            base = requested if requested else [root]
            tally_ids: list[int] | None = list(dict.fromkeys(base + extra))
        else:
            tally_ids = None
        self._manager = self._pdag.create_layer_manager(tally_ids)
        self._shape = SampleShape.from_native(dict(self._manager.sample_shape))
        self._target_index = int(requested[0]) if requested else int(self._manager.root_index)
        self._node_indices = sorted(int(i) for i in self._manager.tally_indices)
        # Detect the multi-node fast path once.
        self._has_pass_all = hasattr(self._manager, "single_pass_all")
        self._pass_index = 0

    @property
    def shape(self) -> SampleShape:
        return self._shape

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def node_indices(self) -> list[int]:
        return list(self._node_indices)

    def step(self, passes: int = 1) -> PassObservation:
        self._pass_index += passes
        if self._has_pass_all:
            raw = self._manager.single_pass_all(passes)
            nodes = {
                int(index): NodeCounters(
                    index=int(index),
                    num_one_bits=int(counter["num_one_bits"]),
                    total_bits=int(counter["total_bits"]),
                )
                for index, counter in raw.items()
            }
        else:
            for _ in range(passes):
                counter = self._manager.single_pass(self._target_index)
            nodes = {
                self._target_index: NodeCounters(
                    index=self._target_index,
                    num_one_bits=int(counter["num_one_bits"]),
                    total_bits=int(counter["total_bits"]),
                )
            }
        return PassObservation(
            pass_index=self._pass_index,
            wall_time=time.monotonic(),
            trials_per_pass=self._shape.trials_per_pass,
            target_index=self._target_index,
            nodes=nodes,
        )

    def close(self) -> None:
        # The native layer manager keeps the compiled PDAG alive via keep_alive;
        # dropping our references lets the extension free device memory.
        self._manager = None
        self._pdag = None


class BernoulliSampler:
    """Pure-Python binomial sampler for testing the engine without a device.

    Draws ``trials_per_pass`` Bernoulli(p) outcomes per pass per node and
    accumulates them, mimicking the native engine's cumulative counters.  Use a
    fixed ``seed`` for deterministic tests.
    """

    def __init__(
        self,
        *,
        true_p: float | dict[int, float] = 0.1,
        trials_per_pass: int = 1 << 16,
        target_index: int = 1,
        seed: int | None = 1234,
        pass_delay_s: float = 0.0,
    ) -> None:
        if isinstance(true_p, dict):
            self._true_p = dict(true_p)
            if target_index not in self._true_p:
                self._true_p[target_index] = next(iter(true_p.values()))
        else:
            self._true_p = {target_index: float(true_p)}
        self._target_index = target_index
        self._trials_per_pass = int(trials_per_pass)
        self._rng = random.Random(seed)
        self._pass_delay_s = pass_delay_s
        self._pass_index = 0
        self._counts: dict[int, int] = {idx: 0 for idx in self._true_p}
        self._totals: dict[int, int] = {idx: 0 for idx in self._true_p}
        self._shape = SampleShape(
            batch_size=1,
            bitpacks_per_batch=max(1, self._trials_per_pass // 64),
            bits_per_bitpack=64,
            trials_per_pass=self._trials_per_pass,
            total_iterations=0,
        )

    @property
    def shape(self) -> SampleShape:
        return self._shape

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def node_indices(self) -> list[int]:
        return sorted(self._true_p)

    def step(self, passes: int = 1) -> PassObservation:
        self._pass_index += passes
        trials = self._trials_per_pass * passes
        nodes: dict[int, NodeCounters] = {}
        for index, p in self._true_p.items():
            successes = self._binomial(trials, p)
            self._counts[index] += successes
            self._totals[index] += trials
            nodes[index] = NodeCounters(
                index=index,
                num_one_bits=self._counts[index],
                total_bits=self._totals[index],
            )
        if self._pass_delay_s:
            time.sleep(self._pass_delay_s)
        return PassObservation(
            pass_index=self._pass_index,
            wall_time=time.monotonic(),
            trials_per_pass=self._trials_per_pass,
            target_index=self._target_index,
            nodes=nodes,
        )

    def _binomial(self, n: int, p: float) -> int:
        if p <= 0.0:
            return 0
        if p >= 1.0:
            return n
        # Exact for the trial counts used in tests; inverse-transform per trial is
        # too slow for 65k trials, so use the normal approximation with a small-n
        # exact fallback.  Tests assert on aggregate convergence, not exact draws.
        if n <= 2048:
            return sum(1 for _ in range(n) if self._rng.random() < p)
        mean = n * p
        std = (n * p * (1.0 - p)) ** 0.5
        draw = round(self._rng.gauss(mean, std))
        return max(0, min(n, draw))

    def close(self) -> None:  # noqa: D401 - protocol completeness
        return None


class PcfbSampler:
    """Adapts ``fuzzybits.Session`` (``.pcfb`` runtime) to the :class:`Sampler` protocol."""

    def __init__(
        self,
        pcfb: bytes,
        options: dict[str, Any] | None = None,
        *,
        event_ids: list[int] | None = None,
    ) -> None:
        import fuzzybits

        names = watch_names_from_options(options)
        if event_ids:
            names = names + [str(int(i)) for i in event_ids]
        tpp = options.get("requested_trials_per_pass") if options else None
        trials = options.get("num_trials") if options else None
        self._session = fuzzybits.Session(
            pcfb,
            backend="auto",
            trials_per_pass=int(float(tpp)) if tpp else None,
            watch=names,
            num_trials=int(float(trials)) if trials else None,
        )
        self._shape = SampleShape.from_native(self._session.sample_shape)
        graph = self._session.graph()
        queries = list(graph.get("query_nodes") or [])
        self._target_index = int(queries[0]["index"]) if queries else int(graph.get("top") or 0)
        if event_ids:
            self._target_index = int(event_ids[0])
        indices: set[int] = {int(q["index"]) for q in queries if "index" in q}
        if event_ids:
            indices.update(int(i) for i in event_ids)
        for want in names:
            for node in graph.get("nodes") or []:
                index = int(node["index"])
                if want == str(index) or want in {
                    str(node.get("id") or ""),
                    str(node.get("name") or ""),
                    str(node.get("full_path") or ""),
                }:
                    indices.add(index)
                    break
        self._node_indices = sorted(indices)
        self._pass_index = 0

    @property
    def shape(self) -> SampleShape:
        return self._shape

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def node_indices(self) -> list[int]:
        return list(self._node_indices)

    def step(self, passes: int = 1) -> PassObservation:
        obs = self._session.step(passes)
        self._pass_index = int(obs.pass_index)
        if not self._node_indices:
            self._node_indices = sorted(int(i) for i in obs.nodes)
        nodes = {
            int(index): NodeCounters(
                index=int(index),
                num_one_bits=int(tally.num_one_bits),
                total_bits=int(tally.total_bits),
            )
            for index, tally in obs.nodes.items()
        }
        return PassObservation(
            pass_index=self._pass_index,
            wall_time=time.monotonic(),
            trials_per_pass=self._shape.trials_per_pass,
            target_index=self._target_index,
            nodes=nodes,
        )

    def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            session.close()


def make_sampler(
    input_files: list[str],
    options: dict[str, Any] | None = None,
    *,
    event_ids: list[int] | None = None,
    pcfb: bytes | None = None,
    pcfb_path: str | None = None,
) -> Sampler:
    """Construct the engine sampler. Default is the ``.pcfb`` Session path."""
    if pcfb is None and pcfb_path:
        pcfb = Path(pcfb_path).read_bytes()
    engine = server_engine()
    if engine == "legacy" and pcfb is None:
        return NativeSampler(input_files, options, event_ids=event_ids)
    if pcfb is None:
        pcfb = compile_pcfb_to_cache(input_files, options)
    return PcfbSampler(pcfb, options, event_ids=event_ids)


__all__ = ["SampleShape", "Sampler", "NativeSampler", "BernoulliSampler", "PcfbSampler", "make_sampler"]
