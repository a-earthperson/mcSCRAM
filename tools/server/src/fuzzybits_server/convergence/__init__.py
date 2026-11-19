"""Host-side Monte-Carlo convergence and statistics engine.

The native ``fuzzybits`` engine is responsible only for the massively
parallel, device-side bitstream sampling.  Everything statistical -- point
estimates, confidence/credible intervals, convergence policies, information gain,
diagnostics -- is computed here, on the host, in pure Python.

Layered design (each layer is independently testable):

* :mod:`.special`   -- dependency-free special functions / sampling formulas.
* :mod:`.tally`     -- raw counter + trajectory state (``NodeState``, ``RunState``).
* :mod:`.sampling`  -- the ``Sampler`` boundary (native + simulated).
* :mod:`.metrics`   -- non-stopping derived metrics (point estimate, information
  gain, oracle diagnostics).
* :mod:`.policies`  -- convergence policies (Wald linear, Bayes credible, GBAS).
* :mod:`.gbas`      -- Huber's GBAS numerics (numpy/scipy; imported lazily).
* :mod:`.runner`    -- the pass->tally->update->emit loop and configuration.

Typical use::

    from fuzzybits_server.convergence import run_calculation
    result = run_calculation(input_files, options, on_progress)

or, for full control / testing::

    from fuzzybits_server.convergence import (
        BernoulliSampler, ConvergenceConfig, ConvergenceRunner,
        WaldLinearPolicy, BayesPolicy,
    )
"""

from __future__ import annotations

from .metrics import (
    InfoGainMetric,
    Metric,
    OracleDiagnostics,
    PointEstimate,
    default_metrics,
)
from .policies import (
    BayesPolicy,
    ConvergencePolicy,
    GBASPolicy,
    PolicyVerdict,
    WaldLinearPolicy,
    make_policy,
)
from .runner import (
    ConvergenceConfig,
    ConvergenceRunner,
    ProgressCallback,
    RunResult,
    run_calculation,
)
from .sampling import BernoulliSampler, NativeSampler, PcfbSampler, SampleShape, Sampler, make_sampler
from .tally import (
    HistoryPoint,
    NodeCounters,
    NodeState,
    PassObservation,
    RunContext,
    RunState,
)

__all__ = [
    # entry points
    "run_calculation",
    "ConvergenceRunner",
    "ConvergenceConfig",
    "RunResult",
    "ProgressCallback",
    # sampling
    "Sampler",
    "NativeSampler",
    "PcfbSampler",
    "BernoulliSampler",
    "make_sampler",
    "SampleShape",
    # state
    "NodeCounters",
    "HistoryPoint",
    "PassObservation",
    "NodeState",
    "RunState",
    "RunContext",
    # metrics
    "Metric",
    "PointEstimate",
    "InfoGainMetric",
    "OracleDiagnostics",
    "default_metrics",
    # policies
    "ConvergencePolicy",
    "PolicyVerdict",
    "WaldLinearPolicy",
    "BayesPolicy",
    "GBASPolicy",
    "make_policy",
]
