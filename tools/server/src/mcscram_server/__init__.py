"""Deprecated import path; use ``fuzzybits_server``.

This alias will be removed after one release.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

warnings.warn(
    "The mcscram_server package is renamed to fuzzybits_server; "
    "this alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

import fuzzybits_server as _pkg  # noqa: E402  (warn before importing the real package)


def _alias(real: str, alias: str) -> None:
    module = importlib.import_module(real)
    sys.modules[alias] = module
    path = getattr(module, "__path__", None)
    if path is None:
        return
    for info in pkgutil.iter_modules(path, prefix=real + "."):
        _alias(info.name, alias + info.name[len(real) :])


_alias("fuzzybits_server", __name__)
