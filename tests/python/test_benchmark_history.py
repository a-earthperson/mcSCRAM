"""History index builder. CPU only, no docker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.history.index import (  # noqa: E402
    build_index,
    channel_for_version,
    compare_ratio,
    flag_changes,
    merge_runs,
    model_summaries,
    publish_payload,
    stamp_name,
)


def _run(*, target: str, version: str, digest: str, created: str, z: float, gpu_total: float | None, sample_s: float | None) -> dict:
    return {
        "schema_version": 1,
        "kind": "run",
        "dataset": "v1",
        "created_utc": created,
        "verdict": "pass",
        "context": {
            "target": target,
            "version": version,
            "channel": channel_for_version(version),
            "image_digest": digest,
            "recipe": "docker/runtime.Dockerfile --target runtime",
            "driver": "580.95.05",
            "runner": "fzb-xerxes-5" if target != "cpu" else "fzb-bench-gtw-1",
            "device": "Tesla V100S-PCIE-32GB" if target == "v100s" else "cpu",
        },
        "models": [
            {
                "id": "chinese",
                "mode": "accuracy",
                "z": z,
                "verdict": "pass",
                "status": "pass",
                "estimate": 1.17e-3,
                "n": 1048576,
                "hits": 1226,
                "expected_hits": 1226.0,
                "p_ref": 1.17e-3,
            },
            {
                "id": "chinese",
                "mode": "perf",
                "verdict": "pass",
                "status": "ok",
                "wall_s": 1.4,
                "sample_s": sample_s,
                "profiler": {
                    "gpu_total_ms": gpu_total,
                    "gpu_kernel_ms": None if gpu_total is None else gpu_total * 0.8,
                    "total_gpu_kernel_ms": gpu_total,
                    "basic_event_ms": None if gpu_total is None else gpu_total * 0.8,
                },
            },
        ],
    }


def test_channel_and_stamp():
    assert channel_for_version("v0.17.1") == "release"
    assert channel_for_version("develop") == "develop"
    assert stamp_name("2026-11-19T23:01:00Z") == "2026-11-19T23-01-00Z"


def test_merge_accuracy_and_perf():
    acc = _run(target="v100s", version="v0.17.1", digest="sha256:aaa", created="2026-11-19T01:00:00Z", z=0.2, gpu_total=None, sample_s=None)
    acc["models"] = [acc["models"][0]]
    perf = _run(target="v100s", version="v0.17.1", digest="sha256:aaa", created="2026-11-19T01:05:00Z", z=0.2, gpu_total=12.5, sample_s=None)
    perf["models"] = [perf["models"][1]]
    merged = merge_runs([acc, perf], version="v0.17.1", driver="580.95.05", runner_name="fzb-xerxes-5")
    assert merged["context"]["version"] == "v0.17.1"
    assert merged["context"]["channel"] == "release"
    assert merged["context"]["driver"] == "580.95.05"
    assert len(merged["models"]) == 2
    summaries = model_summaries(merged["models"])
    assert summaries[0]["id"] == "chinese"
    assert summaries[0]["accuracy"]["z"] == 0.2
    assert summaries[0]["perf"]["gpu_total_ms"] == 12.5


def test_publish_never_rewrites(tmp_path: Path):
    first = _run(target="cpu", version="v0.17.0", digest="sha256:bbb", created="2026-11-19T02:00:00Z", z=-0.1, gpu_total=None, sample_s=2.9)
    written = publish_payload(first, dest=tmp_path, version="v0.17.0")
    path = tmp_path / "bench/v1/results/v0.17.0/cpu/2026-11-19T02-00-00Z.json"
    assert path.is_file()
    assert written["path"] == "results/v0.17.0/cpu/2026-11-19T02-00-00Z.json"
    original = path.read_text(encoding="utf-8")
    again = publish_payload(first, dest=tmp_path, version="v0.17.0")
    assert again["path"] == "results/v0.17.0/cpu/2026-11-19T02-00-00Z-2.json"
    assert path.read_text(encoding="utf-8") == original
    index = json.loads((tmp_path / "bench/v1/index.json").read_text(encoding="utf-8"))
    assert index["n_points"] == 2


def test_flag_image_change(tmp_path: Path):
    a = _run(target="v100s", version="v0.17.0", digest="sha256:old", created="2026-11-01T00:00:00Z", z=0.0, gpu_total=10.0, sample_s=None)
    b = _run(target="v100s", version="v0.17.1", digest="sha256:new", created="2026-11-10T00:00:00Z", z=0.1, gpu_total=10.2, sample_s=None)
    publish_payload(a, dest=tmp_path, version="v0.17.0")
    publish_payload(b, dest=tmp_path, version="v0.17.1")
    index = build_index(tmp_path)
    points = {p["version"]: p for p in index["points"]}
    assert points["v0.17.0"]["flags"]["image_changed"] is False
    assert points["v0.17.1"]["flags"]["image_changed"] is True
    assert points["v0.17.1"]["flags"]["recipe_changed"] is False


def test_compare_ratio():
    assert compare_ratio(10.3, 10.0) == 1.03
    assert compare_ratio(None, 10.0) is None
    assert compare_ratio(10.0, 0.0) is None


def test_local_gpu_lock(tmp_path: Path):
    from benchmarks.harness.guards import GuardError, acquire_lock, release_lock

    lock = tmp_path / "gpu-lock-test"
    handle = acquire_lock(None, str(lock), retries=0)
    try:
        acquire_lock(None, str(lock), retries=0)
        raise AssertionError("second acquire should fail")
    except GuardError:
        pass
    release_lock(handle)
    release_lock(acquire_lock(None, str(lock), retries=0))


def test_flag_changes_isolated_targets():
    points = flag_changes(
        [
            {"target": "cpu", "created_utc": "2026-01-01T00:00:00Z", "recipe": "a", "driver": "1", "runner": "r", "image_digest": "x"},
            {"target": "v100s", "created_utc": "2026-01-02T00:00:00Z", "recipe": "a", "driver": "1", "runner": "g", "image_digest": "y"},
        ]
    )
    assert all(not any(p["flags"].values()) for p in points)
