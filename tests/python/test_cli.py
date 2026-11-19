"""Installed ``fzb`` console script (subprocess, not the function)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

CHINESE_REF = 1.17058e-3
Z_MAX = 4.0
INSPECT_JSON_KEYS = {
    "schema_version",
    "tensors",
    "operators",
    "samples",
    "queries",
    "opcodes",
    "sample_source",
    "query_names",
    "metadata",
}
RUN_JSON_QUERY_KEYS = {"name", "estimate", "ci_low", "ci_high", "n", "hits"}


def _fzb() -> Path:
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    # Do not Path.resolve() sys.executable: venv python is often a symlink
    # into /usr/bin, which would miss the console script beside the venv.
    candidates = [
        Path(sys.executable).parent / name,
        Path(sys.executable).resolve().parent / name,
    ]
    for exe in candidates:
        if exe.is_file():
            return exe
    pytest.skip(f"fzb console script not installed next to {sys.executable}")


def _run_fzb(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("ACPP_VISIBILITY_MASK", "omp")
    env.setdefault("FUZZYBITS_BACKEND", "omp")
    return subprocess.run(
        [str(_fzb()), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


def test_fzb_help_and_version():
    module_help = subprocess.run(
        [sys.executable, "-m", "fuzzybits._cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert module_help.returncode == 0, module_help.stderr
    assert "Usage: fzb" in module_help.stdout

    help_proc = _run_fzb("--help")
    assert help_proc.returncode == 0, help_proc.stderr
    assert "Usage: fzb" in help_proc.stdout
    assert "compile" in help_proc.stdout
    assert "inspect" in help_proc.stdout
    assert "run" in help_proc.stdout

    version_proc = _run_fzb("--version")
    assert version_proc.returncode == 0, version_proc.stderr
    assert version_proc.stdout.strip() == f"fzb {fuzzybits.__version__}"

    empty = _run_fzb()
    assert empty.returncode == 1
    assert "Usage: fzb" in empty.stderr

    unknown = _run_fzb("not-a-command")
    assert unknown.returncode == 1
    assert "unknown command" in unknown.stderr


def test_fzb_compile_inspect_run_chinese(chinese_xml: Path, tmp_path: Path):
    pcfb = tmp_path / "c.pcfb"
    compiled = _run_fzb("compile", str(chinese_xml), "-o", str(pcfb), cwd=tmp_path)
    assert compiled.returncode == 0, compiled.stderr
    assert pcfb.is_file()
    assert "wrote" in compiled.stderr

    inspected = _run_fzb("inspect", str(pcfb), "--json")
    assert inspected.returncode == 0, inspected.stderr
    payload = json.loads(inspected.stdout)
    assert INSPECT_JSON_KEYS <= set(payload)
    assert "diagnostics" not in payload
    assert int(payload["queries"]) >= 1

    ran = _run_fzb(
        "run",
        str(pcfb),
        "--trials",
        "1e6",
        "--backend",
        "omp",
        "--json",
    )
    assert ran.returncode == 0, ran.stderr
    result = json.loads(ran.stdout)
    queries = result["queries"]
    assert queries
    row = queries[0]
    assert RUN_JSON_QUERY_KEYS <= set(row)
    p_hat = float(row["estimate"])
    n = float(row["n"])
    sigma = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n) if n else float("nan")
    z = (p_hat - CHINESE_REF) / sigma if sigma > 0 else float("inf")
    assert abs(z) <= Z_MAX, (
        f"chinese p̂={p_hat:.6e} ref={CHINESE_REF:.6e} |z|={abs(z):.3f} n={n:.0f}"
    )


def test_fzb_run_seed_rejected(tiny_xml: Path, tmp_path: Path):
    pcfb = tmp_path / "tiny.pcfb"
    compiled = _run_fzb("compile", str(tiny_xml), "-o", str(pcfb), cwd=tmp_path)
    assert compiled.returncode == 0, compiled.stderr
    rejected = _run_fzb("run", str(pcfb), "--trials", "1024", "--seed", "1", "--backend", "omp")
    assert rejected.returncode == 1
    assert "--seed is not supported yet" in rejected.stderr

    accepted = _run_fzb(
        "run",
        str(pcfb),
        "--trials",
        "1024",
        "--seed",
        "372",
        "--backend",
        "omp",
        "--json",
    )
    assert accepted.returncode == 0, accepted.stderr
    payload = json.loads(accepted.stdout)
    assert payload["queries"]
