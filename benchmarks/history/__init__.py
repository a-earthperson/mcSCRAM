"""Published single-build history for the frozen dataset (#123).

Merge accuracy and perf harness runs into one JSON per
(version x target x dataset), never rewrite older files, and build
``bench/v1/index.json`` for the web UI and GitHub Pages.
"""

from __future__ import annotations

from .index import (
    DEFAULT_PAGES_INDEX_URL,
    INDEX_REL,
    RESULTS_REL,
    SCHEMA_VERSION,
    build_index,
    channel_for_version,
    merge_runs,
    model_summaries,
    publish_payload,
    stamp_name,
)

__all__ = [
    "DEFAULT_PAGES_INDEX_URL",
    "INDEX_REL",
    "RESULTS_REL",
    "SCHEMA_VERSION",
    "build_index",
    "channel_for_version",
    "merge_runs",
    "model_summaries",
    "publish_payload",
    "stamp_name",
]
