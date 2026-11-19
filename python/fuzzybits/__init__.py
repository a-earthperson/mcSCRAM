"""fuzzybits — Monte Carlo probability engine for OpenPSA MEF fault trees.

Public surface (this wave):

- :data:`__version__`
- :func:`init`
- :func:`list_devices`
- :func:`probability`
- :func:`run` (extension point for PCFB, #44 — not implemented)

The native SYCL extension is selected at import: ``_native_cuda_sm70`` if that
wheel is installed *and* an sm_70 device is present, otherwise
``_native_generic``. ``import mcscram_native`` remains a compatibility shim
that re-exports the active extension.
"""

from __future__ import annotations

from . import _runtime
from ._api import ProbabilityEstimate, list_devices, probability, run
from ._runtime import (
    active_backend,
    active_device,
    active_variant,
    has_cuda_device,
    has_sm70_device,
    init,
    native,
)

__version__ = "0.17.0"

__all__ = [
    "ProbabilityEstimate",
    "__version__",
    "active_backend",
    "active_device",
    "active_variant",
    "has_cuda_device",
    "has_sm70_device",
    "init",
    "list_devices",
    "native",
    "probability",
    "run",
]


def __getattr__(name: str):
    """Forward unknown names to the native module after lazy init.

    Lets ``fuzzybits.run_probability`` / ``fuzzybits.run_compile`` keep working
    for callers that previously used ``mcscram_native`` directly.
    """
    native_mod = _runtime.native()
    try:
        return getattr(native_mod, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'fuzzybits' has no attribute {name!r}") from exc
