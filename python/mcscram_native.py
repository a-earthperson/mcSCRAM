"""Compatibility shim: ``import mcscram_native`` re-exports the active native module.

The fuzzybits-bench server (and older scripts) import ``mcscram_native``
directly. The pip package ships this top-level module so those callers keep
working without ``PYTHONPATH`` or ``LD_PRELOAD``. AdaptiveCpp visibility is
applied by :func:`fuzzybits.init` before the extension is loaded.
"""

from __future__ import annotations

from fuzzybits._runtime import native as _native

_mod = _native()
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")})
# Preserve dunder names the C++ module actually defines.
for _name in ("__doc__",):
    if hasattr(_mod, _name):
        globals()[_name] = getattr(_mod, _name)

__all__ = [k for k in globals() if not k.startswith("_")]
