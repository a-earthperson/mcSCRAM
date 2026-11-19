"""The pass -> tally -> update -> emit loop and its configuration.

:class:`ConvergenceRunner` ties the layers together:

1. pull one :class:`PassObservation` from a :class:`Sampler`,
2. fold it into :class:`RunState`,
3. evaluate every attached :class:`ConvergencePolicy` and :class:`Metric`,
4. assemble a flat progress frame, and
5. decide whether to stop (burn-in, early-stop on the chosen policy, or a
   trial/pass budget).

The progress frame keeps the field names the existing web client consumes
(``mean``, ``epsilon``, ``target_epsilon``, ``epsilon_converged``,
``p01/05/95/99``, ``info_bits_*`` …) filled from the *primary* policy (GBAS
by default), and adds every policy/metric under its own namespace (``wald_*``,
``bayes_*``, ``info_bits_*`` …) so additional streams are available without
breaking the UI.

:class:`ConvergenceConfig` carries the host-side knobs and is built from the
existing server options dict via :meth:`ConvergenceConfig.from_options`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .metrics import Metric, default_metrics
from .policies import ConvergencePolicy, PolicyVerdict, make_policy
from .sampling import Sampler, make_sampler
from .special import normal_quantile_two_sided
from .tally import RunContext, RunState

ProgressCallback = Callable[[dict[str, Any]], None]

# Default primary policy: fills the canonical (un-namespaced) frontend fields.
PRIMARY_POLICY = "gbas"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return int(round(f)) if f is not None else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sanitize(value: Any) -> Any:
    """Make a progress value safe for strict JSON (browsers reject NaN/Infinity)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


@dataclass(frozen=True)
class ConvergenceConfig:
    """Host-side convergence configuration (independent of the native settings).

    Note: intentionally *not* ``slots=True`` -- :meth:`from_options` reads field
    defaults via ``cls.<field>`` (which slots would shadow with descriptors).
    """

    delta: float = 1.0e-3
    confidence: float = 0.99
    burn_in: int = 0
    max_trials: int = 0  # 0 -> unlimited
    max_passes: int = 0  # 0 -> derive from max_trials, else unbounded
    early_stop: bool = False
    oracle_p: float | None = None
    policies: tuple[str, ...] = ("wald", "bayes")
    stopping_policy: str = "wald"
    event_ids: tuple[int, ...] = ()
    emit_min_interval_s: float = 1.0 / 60.0
    stream_nodes: bool = False

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> "ConvergenceConfig":
        """Build a config from the server's CLI-like options mapping."""
        options = options or {}
        delta = _as_float(options.get("delta"))
        confidence = _as_float(options.get("confidence"))
        burn_in = _as_int(options.get("burn_in"))
        max_trials = _as_int(options.get("num_trials"))
        oracle = _as_float(options.get("oracle"))
        policy = options.get("policy")
        stopping = str(policy).strip().lower() if policy else cls.stopping_policy
        # Always evaluate both policies so Wald and Bayes stats stream together,
        # regardless of which one gates the stop.
        policies: tuple[str, ...] = ("wald", "bayes")
        if stopping not in policies:
            policies = (*policies, stopping)

        emit_interval = _as_float(options.get("emit_min_interval_s"))
        event_ids = tuple(int(i) for i in options.get("convergence_event_ids", ()) or ())

        return cls(
            delta=delta if delta is not None and delta > 0 else cls.delta,
            confidence=confidence if confidence is not None else cls.confidence,
            burn_in=max(0, burn_in or 0),
            max_trials=max(0, max_trials or 0),
            max_passes=max(0, _as_int(options.get("max_passes")) or 0),
            early_stop=_as_bool(options.get("early_stop", False)),
            oracle_p=oracle if (oracle is not None and oracle >= 0.0) else None,
            policies=policies,
            stopping_policy=stopping,
            event_ids=event_ids,
            emit_min_interval_s=emit_interval if emit_interval is not None else cls.emit_min_interval_s,
            stream_nodes=_as_bool(options.get("stream_nodes", False)),
        )


@dataclass(slots=True)
class RunResult:
    """Outcome of a completed (or stopped) convergence run."""

    state: RunState
    final_progress: dict[str, Any]
    stop_reason: str | None
    converged: bool
    elapsed_s: float

    @property
    def summary(self) -> dict[str, Any]:
        """A native-``run_probability``-compatible result dict for the worker."""
        target = self.state.target
        mean = target.mean

        # If the stopping policy publishes its own point estimate (e.g. GBAS's
        # Gamma-corrected ``(k-1)/R``, whose relative error is exactly known),
        # prefer it as the authoritative result over the raw sample proportion.
        stopping = self.final_progress.get("stopping_policy")
        if stopping and self.converged:
            pol_mean = self.final_progress.get(f"{stopping}_mean")
            if isinstance(pol_mean, (int, float)) and math.isfinite(float(pol_mean)):
                mean = float(pol_mean)

        result: dict[str, Any] = {
            "analysis_time": self.elapsed_s,
            "warnings": "",
            "probabilities": [{"p_total": mean, "analysis_time": self.elapsed_s}],
            "engine": "python",
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "mean": mean,
            "sample_mean": target.mean,
            "trials": target.total_bits,
            "passes": self.state.pass_count,
            "stopping_policy": stopping,
        }

        # Carry the stopping policy's confidence interval when it provides one.
        ci_low = self.final_progress.get(f"{stopping}_ci_low") if stopping else None
        ci_high = self.final_progress.get(f"{stopping}_ci_high") if stopping else None
        if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
            result["ci_low"] = float(ci_low)
            result["ci_high"] = float(ci_high)
        return result


class ConvergenceRunner:
    """Drives the host-side convergence loop for a single sampler."""

    def __init__(
        self,
        sampler: Sampler,
        config: ConvergenceConfig,
        *,
        policies: Sequence[ConvergencePolicy] | None = None,
        metrics: Sequence[Metric] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.sampler = sampler
        self.config = config
        self.on_progress = on_progress
        self.policies: list[ConvergencePolicy] = (
            list(policies) if policies is not None else [make_policy(name) for name in config.policies]
        )
        self.metrics: list[Metric] = (
            list(metrics) if metrics is not None else default_metrics()
        )
        self.state = RunState(target_index=sampler.target_index)

        z = normal_quantile_two_sided(config.confidence)
        self.ctx = RunContext(
            confidence=config.confidence,
            z=z,
            delta=config.delta,
            trials_per_pass=sampler.shape.trials_per_pass,
            target_index=sampler.target_index,
            oracle_p=config.oracle_p,
        )

        by_name = {policy.name: policy for policy in self.policies}
        self._primary_name = PRIMARY_POLICY if PRIMARY_POLICY in by_name else self.policies[0].name
        self._stopping_name = (
            config.stopping_policy if config.stopping_policy in by_name else self._primary_name
        )

    # -- public API --------------------------------------------------------
    def run(self) -> RunResult:
        """Run until a stop condition fires; returns the final :class:`RunResult`."""
        cfg = self.config
        tpp = max(1, self.sampler.shape.trials_per_pass)
        max_passes = cfg.max_passes
        if not max_passes and cfg.max_trials:
            max_passes = math.ceil(cfg.max_trials / tpp)

        start = time.monotonic()
        last_emit = -math.inf
        last_phase: str | None = None
        final_progress: dict[str, Any] = {}
        stop_reason: str | None = None
        converged = False

        while True:
            observation = self.sampler.step()
            self.state.update(observation)
            target = self.state.target

            phase = "burn_in" if target.total_bits < cfg.burn_in else "sampling"
            verdicts = {policy.name: policy.evaluate(target, self.ctx) for policy in self.policies}

            stopping_converged = verdicts[self._stopping_name].converged and phase == "sampling"
            stop_reason = self._stop_reason(
                phase=phase,
                stopping_converged=stopping_converged,
                total_bits=target.total_bits,
                pass_count=self.state.pass_count,
                max_passes=max_passes,
            )
            converged = stopping_converged
            stopping = stop_reason is not None

            progress = self._build_progress(phase, verdicts)
            progress["converged"] = stopping_converged
            if stopping:
                progress["stop_reason"] = stop_reason
            final_progress = progress

            force = stopping or last_phase != phase or self.state.pass_count == 1
            if self.on_progress is not None and (
                force or (observation.wall_time - last_emit) >= cfg.emit_min_interval_s
            ):
                self.on_progress(progress)
                last_emit = observation.wall_time
            last_phase = phase

            if stopping:
                break

        return RunResult(
            state=self.state,
            final_progress=final_progress,
            stop_reason=stop_reason,
            converged=converged,
            elapsed_s=time.monotonic() - start,
        )

    # -- internals ---------------------------------------------------------
    def _stop_reason(
        self,
        *,
        phase: str,
        stopping_converged: bool,
        total_bits: int,
        pass_count: int,
        max_passes: int,
    ) -> str | None:
        if self.config.max_trials and total_bits >= self.config.max_trials:
            return "max_trials"
        if max_passes and pass_count >= max_passes:
            return "max_passes"
        if phase == "sampling" and self.config.early_stop and stopping_converged:
            return "converged"
        return None

    def _build_progress(
        self, phase: str, verdicts: dict[str, PolicyVerdict]
    ) -> dict[str, Any]:
        target = self.state.target
        tpp = max(1, self.sampler.shape.trials_per_pass)
        total_bits = target.total_bits

        progress: dict[str, Any] = {
            "phase": phase,
            "pass_index": self.state.pass_count,
            "iterations": self.state.pass_count,
            "trials": total_bits,
            "trials_per_pass": tpp,
        }

        # Auxiliary metrics (point estimate, info gain, diagnostics).
        for metric in self.metrics:
            progress.update(metric.compute(target, self.ctx))

        # Primary policy fills the canonical (un-namespaced) fields.
        primary = verdicts[self._primary_name]
        for key, value in primary.metrics.items():
            if value is not None:
                progress[key] = value
        progress["epsilon"] = primary.epsilon
        progress["target_epsilon"] = primary.target_epsilon
        progress["epsilon_converged"] = primary.converged
        # GBAS does not emit Wald percentile bands; keep them on the canonical
        # frame from Wald when both policies are attached.
        if self._primary_name != "wald" and "wald" in verdicts:
            wald_metrics = verdicts["wald"].metrics
            for key in ("std_err", "p01", "p05", "p95", "p99"):
                if progress.get(key) is None:
                    value = wald_metrics.get(key)
                    if value is not None:
                        progress[key] = value
        projected = primary.projected_trials
        progress["projected_trials"] = projected
        progress["projected_iterations"] = math.ceil(projected / tpp) if projected else 0
        remaining = max(0, projected - total_bits)
        progress["remaining_trials"] = remaining
        progress["remaining_iterations"] = math.ceil(remaining / tpp) if remaining else 0

        # Every policy is emitted under its namespace so multiple convergence
        # criteria stream simultaneously (wald_*, bayes_*, ...).  Core verdict
        # fields are written last so they win over any duplicated metric keys.
        for name, verdict in verdicts.items():
            for key, value in verdict.metrics.items():
                progress[f"{name}_{key}"] = value
            progress[f"{name}_converged"] = verdict.converged
            progress[f"{name}_epsilon"] = verdict.epsilon
            progress[f"{name}_target_epsilon"] = verdict.target_epsilon
            progress[f"{name}_projected_trials"] = verdict.projected_trials

        progress["primary_policy"] = self._primary_name
        progress["stopping_policy"] = self._stopping_name

        if self.config.stream_nodes and len(self.state.nodes) > 1:
            progress["nodes"] = {
                str(index): {"mean": node.mean, "trials": node.total_bits}
                for index, node in self.state.nodes.items()
            }

        return {key: _sanitize(value) for key, value in progress.items()}


def run_calculation(
    input_files: list[str],
    options: dict[str, Any] | None,
    on_progress: ProgressCallback | None = None,
    *,
    pcfb: bytes | None = None,
    pcfb_path: str | None = None,
) -> dict[str, Any]:
    """Compile, sample, and converge entirely on the host; return a result dict.

    Default engine is ``.pcfb`` / ``fuzzybits.Session``. Set
    ``FUZZYBITS_SERVER_ENGINE=legacy`` for the in-memory ``run_compile`` path.
    The native engine is used *only* to produce raw per-pass popcounts; all
    convergence and statistics are computed here.
    """
    config = ConvergenceConfig.from_options(options)
    sampler = make_sampler(
        input_files,
        options,
        event_ids=list(config.event_ids) or None,
        pcfb=pcfb,
        pcfb_path=pcfb_path,
    )
    try:
        runner = ConvergenceRunner(sampler, config, on_progress=on_progress)
        result = runner.run()
    finally:
        sampler.close()
    return result.summary


__all__ = [
    "ConvergenceConfig",
    "ConvergenceRunner",
    "RunResult",
    "ProgressCallback",
    "run_calculation",
]
