"""Frozen benchmark dataset and harness (issues #120, #121).

Legacy mcSCRAM-era scripts in this directory (``bench_scram.py``,
``benchmark_compiler.py``, …) are kept for history and are not used by the
harness.
"""

from __future__ import annotations

from .loader import Manifest, Model, load_manifest, repo_root, validate_manifest

__all__ = [
    "Manifest",
    "Model",
    "load_manifest",
    "repo_root",
    "validate_manifest",
]
