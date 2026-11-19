"""Unit tests for the PR bench-gate helpers. No docker, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "ci") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "ci"))

from bench_comment import (  # noqa: E402
    MARKER,
    overall_verdict,
    render,
    write_skip,
)
from bench_filter import classify  # noqa: E402
from benchmarks.harness.cli import _runtime_host  # noqa: E402
from benchmarks.harness.guards import acquire_lock, release_lock  # noqa: E402


def test_classify_docs_only():
    result = classify(["docs/benchmarks.md", "README.md"])
    assert result["docs_only"] is True
    assert result["run_benches"] is False
    assert result["product"] is False


def test_classify_product_rebuild():
    result = classify(["src/mc/engine.cpp", "docs/benchmarks.md"])
    assert result["product"] is True
    assert result["run_benches"] is True
    assert result["docs_only"] is False


def test_classify_harness_reuses_image():
    result = classify(["benchmarks/harness/cli.py", ".github/workflows/bench.yml"])
    assert result["product"] is False
    assert result["bench"] is True
    assert result["run_benches"] is True
    assert classify(["scripts/ci/bench_comment.py"])["bench"] is True


def test_runtime_host_is_local_for_default_docker_context():
    target = {
        "docker_context": "default",
        "lock_host": "xerxes",
        "lock_path": "/tmp/gpu-lock-xerxes-0",
    }
    assert _runtime_host(target) is None
    target["docker_context"] = "xerxes"
    assert _runtime_host(target) == "xerxes"


def test_local_gpu_lock_roundtrip(tmp_path: Path):
    lock = tmp_path / "gpu-lock-xerxes-0"
    acquire_lock(None, str(lock), retries=0)
    assert lock.is_dir()
    try:
        acquire_lock(None, str(lock), retries=0)
        assert False, "second acquire must fail"
    except Exception as exc:
        assert "already held" in str(exc)
    release_lock(None, str(lock))
    assert not lock.exists()
    acquire_lock(None, str(lock), retries=0)
    release_lock(None, str(lock))


def test_comment_cpu_block_is_report_only(tmp_path: Path):
    cpu = tmp_path / "cpu.json"
    cpu.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ab",
                "verdict": "block",
                "dataset": "v1",
                "context": {"target": "cpu", "base_digest": "sha256:" + "a" * 64},
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    acc = tmp_path / "acc.json"
    acc.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "run",
                "verdict": "pass",
                "dataset": "v1",
                "context": {},
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    jobs = {
        "accuracy": acc,
        "v100s": None,
        "rtx3070": None,
        "cpu": cpu,
    }
    loaded = {
        "accuracy": json.loads(acc.read_text(encoding="utf-8")),
        "v100s": None,
        "rtx3070": None,
        "cpu": json.loads(cpu.read_text(encoding="utf-8")),
    }
    assert overall_verdict(loaded) == "warn"
    text, overall = render(
        jobs=jobs,
        run_url="https://example.test/run",
        notices=[],
        trusted=True,
        run_benches=True,
    )
    assert MARKER in text
    assert "report-only" in text
    assert overall == "warn"


def test_comment_gpu_block_fails(tmp_path: Path):
    v100s = tmp_path / "v100s.json"
    v100s.write_text(
        json.dumps({"schema_version": 1, "kind": "ab", "verdict": "block", "models": []}),
        encoding="utf-8",
    )
    loaded = {"accuracy": {"verdict": "pass"}, "v100s": {"verdict": "block"}, "rtx3070": {"verdict": "pass"}, "cpu": {"verdict": "pass"}}
    assert overall_verdict(loaded) == "block"


def test_write_skip_json(tmp_path: Path):
    path = tmp_path / "skip.json"
    write_skip(path, job="bench-perf-v100s", reason="fork PR")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "skip"
    assert payload["verdict"] == "pass"
