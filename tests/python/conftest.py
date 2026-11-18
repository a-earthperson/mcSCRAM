from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def tiny_xml() -> Path:
    path = DATA / "tiny.xml"
    if not path.is_file():
        pytest.skip(f"missing fixture {path}")
    return path


@pytest.fixture(scope="session")
def das9201_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    for candidate in (
        repo / "input" / "Aralia" / "das9201.xml",
        repo / "docker" / "inputs" / "das9201.xml",
    ):
        if candidate.is_file():
            return candidate
    pytest.skip("das9201.xml not found in the source tree")


@pytest.fixture(scope="session")
def chinese_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    for candidate in (
        repo / "input" / "Aralia" / "chinese.xml",
        repo / "docker" / "inputs" / "chinese.xml",
    ):
        if candidate.is_file():
            return candidate
    pytest.skip("chinese.xml not found in the source tree")
