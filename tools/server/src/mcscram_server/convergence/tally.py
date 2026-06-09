"""Host-side tally state for the Python convergence engine.

The native sampler only ever produces *raw counters* per node per pass:
``num_one_bits`` (Bernoulli successes) and ``total_bits`` (cumulative trials).
Everything statistical is derived from these on the host.  This module owns the
state model:

* :class:`NodeCounters` — the immutable raw counter pair (the binding's
  ``tally_counter_dict``) for a single node at a single point in time.
* :class:`PassObservation` — the full output of one sampler pass: counters for
  every tracked node plus pass metadata (index, wall-clock, trials/pass).
* :class:`NodeState` — accumulates a single node's counters across passes and
  keeps a bounded trajectory ``history`` so trajectory metrics (gradient,
  information gain, throughput) can be computed without re-deriving from scratch.
* :class:`RunState` — the collection of :class:`NodeState` for a run.
* :class:`RunContext` — immutable per-run configuration shared with every
  policy/metric (confidence ``z``, relative margin ``delta``, oracle, geometry).

Estimators and policies consume ``(NodeState, RunContext)`` and never touch the
sampler, so they are trivially unit-testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Mapping

from .special import DELTA_EPSILON

# Default trajectory window.  Trajectory metrics only need the last few points;
# a few thousand is plenty to support smoothed/windowed differentiation while
# bounding memory for unlimited (num_trials=0) live runs.
DEFAULT_HISTORY: int = 4096


@dataclass(frozen=True, slots=True)
class NodeCounters:
    """Raw cumulative Bernoulli counters for one node (the ``tally_counter_dict``)."""

    index: int
    num_one_bits: int
    total_bits: int

    @property
    def mean(self) -> float:
        """Sample proportion ``p_hat = num_one_bits / total_bits`` (0 if no trials)."""
        return self.num_one_bits / self.total_bits if self.total_bits else 0.0

    @property
    def num_zero_bits(self) -> int:
        return self.total_bits - self.num_one_bits


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """One point on a node's cumulative trajectory."""

    total_bits: int
    num_one_bits: int
    wall_time: float

    @property
    def mean(self) -> float:
        return self.num_one_bits / self.total_bits if self.total_bits else 0.0


@dataclass(frozen=True, slots=True)
class PassObservation:
    """Immutable result of a single sampler pass over all tracked nodes."""

    pass_index: int  # 1-based count of passes executed so far this run
    wall_time: float  # time.monotonic() captured right after the device drained
    trials_per_pass: int
    target_index: int
    nodes: Mapping[int, NodeCounters]

    @property
    def target(self) -> NodeCounters:
        return self.nodes[self.target_index]


@dataclass(slots=True)
class NodeState:
    """Accumulated state for a single node, with a bounded trajectory buffer."""

    index: int
    num_one_bits: int = 0
    total_bits: int = 0
    first_wall_time: float | None = None
    last_wall_time: float | None = None
    history: Deque[HistoryPoint] = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY))

    def update(self, counters: NodeCounters, wall_time: float) -> None:
        """Fold a new (cumulative) counter reading into this node's state."""
        self.num_one_bits = counters.num_one_bits
        self.total_bits = counters.total_bits
        if self.first_wall_time is None:
            self.first_wall_time = wall_time
        self.last_wall_time = wall_time
        self.history.append(HistoryPoint(counters.total_bits, counters.num_one_bits, wall_time))

    # -- derived point estimates ------------------------------------------
    @property
    def counters(self) -> NodeCounters:
        return NodeCounters(self.index, self.num_one_bits, self.total_bits)

    @property
    def mean(self) -> float:
        return self.num_one_bits / self.total_bits if self.total_bits else 0.0

    @property
    def mean_guarded(self) -> float:
        """``p_hat`` floored at :data:`DELTA_EPSILON` (for ratios / log scales)."""
        return max(self.mean, DELTA_EPSILON)

    @property
    def std_err(self) -> float:
        r"""Binomial standard error :math:`\sqrt{\hat p (1-\hat p)/N}`."""
        n = self.total_bits
        if n == 0:
            return 0.0
        p = self.mean
        return (p * (1.0 - p) / n) ** 0.5

    def previous_point(self) -> HistoryPoint | None:
        """The trajectory point immediately before the latest one, if any."""
        return self.history[-2] if len(self.history) >= 2 else None


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable per-run configuration handed to every policy and metric."""

    confidence: float
    z: float
    delta: float
    trials_per_pass: int
    target_index: int
    oracle_p: float | None = None

    @property
    def diagnostics_enabled(self) -> bool:
        return self.oracle_p is not None and self.oracle_p >= 0.0


@dataclass(slots=True)
class RunState:
    """All node states for a run plus aggregate pass bookkeeping."""

    target_index: int
    pass_count: int = 0
    nodes: dict[int, NodeState] = field(default_factory=dict)

    def update(self, observation: PassObservation) -> None:
        self.pass_count = observation.pass_index
        for index, counters in observation.nodes.items():
            node = self.nodes.get(index)
            if node is None:
                node = NodeState(index=index)
                self.nodes[index] = node
            node.update(counters, observation.wall_time)

    @property
    def target(self) -> NodeState:
        return self.nodes[self.target_index]

    def iter_nodes(self) -> Iterable[NodeState]:
        return self.nodes.values()


__all__ = [
    "DEFAULT_HISTORY",
    "NodeCounters",
    "HistoryPoint",
    "PassObservation",
    "NodeState",
    "RunContext",
    "RunState",
]
