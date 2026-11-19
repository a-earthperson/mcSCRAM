"""Skip ``@pytest.mark.native`` cases when the fuzzybits wheel is absent."""

from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if importlib.util.find_spec("fuzzybits") is not None:
        try:
            import fuzzybits

            fuzzybits.native()
        except Exception:
            pass
        else:
            return
    skip_native = pytest.mark.skip(reason="fuzzybits wheel / native extension not installed")
    for item in items:
        if item.get_closest_marker("native") is not None:
            item.add_marker(skip_native)
