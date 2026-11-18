"""Stable Python API wrapping the native ``run_probability`` / device bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    ``early_stop``, ``compilation_level``, …).

    Parameters
    ----------
    model_xml:
        Path to a MEF XML file, or a sequence of input files.
    trials:
        Bernoulli trial count (``num_trials``). ``None`` uses the engine
        default (auto / early-stop).
    seed:
        PRNG seed.
    pdag_c:
        PDAG compilation level (``-c`` / ``compilation_level``).
    no_kn:
        Match the CLI ``--no-kn`` flag (expand k-out-of-n to AND/OR).
    policy:
        Convergence interval policy (``"wald"`` or ``"bayes"``).
    """
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
    if seed is not None:
        opts["seed"] = int(seed)
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


def run(model: str | Path, **_kwargs: Any) -> Any:
    """Reserved for compiled PCFB models.

    See fuzzybit-org/fuzzybits#44. MEF XML belongs in :func:`probability`.
    """
    raise NotImplementedError(
        "fuzzybits.run() will accept compiled PCFB models "
        "(https://github.com/fuzzybit-org/fuzzybits/issues/44). "
        f"For OpenPSA MEF XML use fuzzybits.probability({model!r}, ...)."
    )
