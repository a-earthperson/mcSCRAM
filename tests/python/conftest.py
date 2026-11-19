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


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="session")
def das9201_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = _first_existing(
        DATA / "das9201.xml",
        repo / "input" / "Aralia" / "das9201.xml",
        repo / "docker" / "inputs" / "das9201.xml",
    )
    if path is None:
        pytest.skip("das9201.xml not found in the source tree")
    return path


@pytest.fixture(scope="session")
def smalltree_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = _first_existing(
        DATA / "SmallTree.xml",
        repo / "input" / "SmallTree" / "SmallTree.xml",
    )
    if path is None:
        pytest.skip("SmallTree.xml not found in the source tree")
    return path


@pytest.fixture(scope="session")
def bscu_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = _first_existing(
        DATA / "BSCU.xml",
        repo / "input" / "BSCU" / "BSCU.xml",
    )
    if path is None:
        pytest.skip("BSCU.xml not found in the source tree")
    return path


@pytest.fixture(scope="session")
def chinese_xml() -> Path:
    repo = Path(__file__).resolve().parents[2]
    path = _first_existing(
        DATA / "chinese.xml",
        repo / "input" / "Aralia" / "chinese.xml",
        repo / "docker" / "inputs" / "chinese.xml",
    )
    if path is None:
        pytest.skip("chinese.xml not found in the source tree")
    return path
