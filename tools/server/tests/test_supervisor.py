"""Supervisor start/stop rules for the calculation process."""

from __future__ import annotations

from fuzzybits_server.config import CalculationConfig
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.supervisor import CalculationSupervisor


class _LingeringProcess:
    """A calculation process that stays alive until it is joined or terminated."""

    def __init__(self, exits_on_join: bool) -> None:
        self._alive = True
        self._exits_on_join = exits_on_join
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        if self._exits_on_join:
            self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self._alive = False


def _supervisor_with(process: _LingeringProcess) -> tuple[CalculationSupervisor, list[CalculationConfig]]:
    supervisor = CalculationSupervisor(EventHub())
    supervisor._process = process  # type: ignore[assignment]
    started: list[CalculationConfig] = []
    supervisor._start_locked = started.append  # type: ignore[method-assign]
    return supervisor, started


def test_start_after_completion_reaps_exiting_process() -> None:
    process = _LingeringProcess(exits_on_join=False)
    supervisor, started = _supervisor_with(process)
    supervisor.state.mark_started(42)
    supervisor.state.mark_complete({"mean": 0.001})

    config = CalculationConfig(input_files=[], options={})
    result = supervisor.start_sampling(config)

    assert result["accepted"] is True
    assert process.terminated
    assert started == [config]


def test_start_after_completion_waits_for_clean_exit() -> None:
    process = _LingeringProcess(exits_on_join=True)
    supervisor, started = _supervisor_with(process)
    supervisor.state.mark_started(42)
    supervisor.state.mark_complete({"mean": 0.001})

    result = supervisor.start_sampling(CalculationConfig(input_files=[], options={}))

    assert result["accepted"] is True
    assert not process.terminated
    assert len(started) == 1


def test_start_while_running_is_rejected() -> None:
    process = _LingeringProcess(exits_on_join=False)
    supervisor, started = _supervisor_with(process)
    supervisor.state.mark_started(42)

    result = supervisor.start_sampling(CalculationConfig(input_files=[], options={}))

    assert result["accepted"] is False
    assert result["reason"] == "calculation already running"
    assert not process.terminated
    assert started == []
