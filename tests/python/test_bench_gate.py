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
from benchmarks.harness.cli import (  # noqa: E402
    TARGETS,
    _flagged_ids,
    _gpu_profiler_with_fallback,
    _payload_flagged,
    _runtime_host,
)
from benchmarks.harness.stats import paired_log_ratio_ci, t975  # noqa: E402
from benchmarks.harness.guards import (  # noqa: E402
    GuardError,
    _display_name,
    acquire_lock,
    format_foreign_load,
    release_lock,
    wait_for_idle,
)
from benchmarks.harness.verdict import (  # noqa: E402
    apply_report_only,
    classify_run_failure,
    collect_timings,
    combine_verdicts,
)
from benchmarks.harness.summarize import shared_device_note, summarize  # noqa: E402


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


def test_xerxes_gpu_indices_match_nvml_harness_table():
    """nvidia-smi / NVML ordinals, not CUDA FASTEST_FIRST (reverse on xerxes)."""
    assert TARGETS["v100s"]["gpus"] == "device=0"
    assert TARGETS["v100s"]["lock_path"] == "/tmp/gpu-lock-xerxes-0"
    assert TARGETS["v100s"]["shared_device"] is True
    assert TARGETS["v100s"]["report_only"] is True
    assert TARGETS["rtx3070"]["gpus"] == "device=1"
    assert TARGETS["rtx3070"]["lock_path"] == "/tmp/gpu-lock-xerxes-1"
    assert TARGETS["rtx3070"]["shared_device"] is True
    assert TARGETS["rtx3070"]["report_only"] is True
    isolated = TARGETS["rtx3070-isolated"]
    assert isolated["gpus"] == "device=1"
    assert isolated["lock_path"] == "/tmp/gpu-lock-xerxes-1"
    assert isolated["shared_device"] is False
    assert not isolated.get("report_only")
    assert TARGETS["cpu"]["report_only"] is True
    assert TARGETS["rtx5060ti"]["gpus"] == "device=0"
    assert TARGETS["rtx5060ti"]["lock_path"] == "/tmp/gpu-lock-shelly-0"
    assert TARGETS["rtx5060ti"]["shared_device"] is True
    assert TARGETS["rtx5060ti"]["report_only"] is True
    assert TARGETS["rtx5060ti"]["profiler"] is None
    assert TARGETS["rtx5060ti"]["runner"] == "shelly-1"


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
    handle = acquire_lock(None, str(lock), retries=0)
    try:
        acquire_lock(None, str(lock), retries=0)
        raise AssertionError("second acquire must fail")
    except GuardError as exc:
        assert "already held" in str(exc)
    release_lock(handle)
    again = acquire_lock(None, str(lock), retries=0)
    release_lock(again)


def test_local_gpu_lock_released_when_holder_dies(tmp_path: Path):
    import subprocess

    lock = tmp_path / "gpu-lock-xerxes-1"
    script = (
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_CREAT, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        try:
            acquire_lock(None, str(lock), retries=0)
            raise AssertionError("lock must be held by the child")
        except GuardError:
            pass
    finally:
        holder.kill()
        holder.wait()
    release_lock(acquire_lock(None, str(lock), retries=0))


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
        "rtx5060ti": None,
        "cpu": cpu,
    }
    loaded = {
        "accuracy": json.loads(acc.read_text(encoding="utf-8")),
        "v100s": None,
        "rtx3070": None,
        "rtx5060ti": None,
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


def test_comment_report_only_gpu_block_warns():
    loaded = {
        "accuracy": {"verdict": "pass"},
        "rtx3070": {"verdict": "block"},
        "rtx3070_isolated": None,
        "v100s": {"verdict": "block"},
        "rtx5060ti": {"verdict": "warn"},
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "warn"


def test_comment_isolated_block_fails():
    loaded = {
        "accuracy": {"verdict": "pass"},
        "rtx3070": {"verdict": "warn"},
        "rtx3070_isolated": {"verdict": "block"},
        "v100s": {"verdict": "pass"},
        "rtx5060ti": None,
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "block"
    loaded["rtx3070_isolated"] = {"verdict": "inconclusive"}
    assert overall_verdict(loaded) == "inconclusive"
    loaded["rtx3070_isolated"] = {"verdict": "pass"}
    assert overall_verdict(loaded) == "warn"


def _write_verdict(path: Path, verdict: str, **extra) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "kind": "ab", "verdict": verdict, "models": [], **extra}),
        encoding="utf-8",
    )
    return path


def test_comment_flagged_screen_requires_isolated(tmp_path: Path):
    acc = _write_verdict(tmp_path / "accuracy.json", "pass")
    screen = _write_verdict(tmp_path / "rtx3070.json", "block", flagged=True)
    jobs = {
        "accuracy": acc,
        "rtx3070": screen,
        "rtx3070_isolated": None,
        "v100s": None,
        "rtx5060ti": None,
        "cpu": None,
    }
    text, overall = render(
        jobs=jobs,
        run_url="https://example.test/run",
        notices=[],
        trusted=True,
        run_benches=True,
        screen_flagged=True,
    )
    assert overall == "error"
    assert "gpu-isolated" in text
    jobs["rtx3070_isolated"] = _write_verdict(tmp_path / "rtx3070-isolated.json", "pass")
    text, overall = render(
        jobs=jobs,
        run_url="https://example.test/run",
        notices=[],
        trusted=True,
        run_benches=True,
        screen_flagged=True,
    )
    assert overall == "warn"
    assert "**Isolated run:** `pass`" in text


def test_write_skip_json(tmp_path: Path):
    path = tmp_path / "skip.json"
    write_skip(path, job="bench-perf-v100s", reason="fork PR")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "skip"
    assert payload["verdict"] == "pass"


def test_format_foreign_load_parakeet():
    snap = {
        "compute_apps": [
            {
                "pid": "15727",
                "process_name": "/app/parakeet",
                "used_gpu_memory": "3054 MiB",
                "name": "parakeet",
            }
        ],
        "gpu": {"utilization_gpu": 14.0},
    }
    text = format_foreign_load(snap)
    assert text == "parakeet 3.0 GiB, util 14%"
    assert shared_device_note({"shared_device": True, "load_before": {**snap, "foreign_load": text}}) == (
        "shared GPU, foreign load: parakeet 3.0 GiB, util 14%"
    )
    assert shared_device_note({"shared_device": False, "load_before": snap}) is None


def test_display_name_skips_interpreters_and_inline_code():
    spawn = "/app/server/.venv/bin/python3 -c from multiprocessing.spawn import spawn_main"
    assert _display_name("/app/server/.venv/bin/python3", "169672", spawn) == "server"
    assert _display_name("[Not Found]", "7", "python3 -c import fuzzybits") == "pid 7"
    assert _display_name("python3", "8", "python3 -m vllm.entrypoints.api_server") == "vllm.entrypoints.api_server"
    assert _display_name("/app/parakeet", "15727") == "parakeet"


def test_shared_device_records_foreign_load_and_does_not_refuse(monkeypatch):
    apps = [
        {
            "pid": "15727",
            "process_name": "/app/parakeet",
            "used_gpu_memory": "3054 MiB",
            "name": "parakeet",
        }
    ]
    stats = {"utilization_gpu": 14.0, "memory_used_mib": 3200.0}
    monkeypatch.setattr("benchmarks.harness.guards.host_nproc", lambda ctx: 16)
    monkeypatch.setattr("benchmarks.harness.guards.host_loadavg", lambda ctx: (1.0, 1.0, 1.0))
    monkeypatch.setattr(
        "benchmarks.harness.guards.gpu_compute_apps",
        lambda ctx, gpus=None, **_: apps,
    )
    monkeypatch.setattr(
        "benchmarks.harness.guards.gpu_device_stats",
        lambda ctx, gpus=None, **_: stats,
    )
    snap = wait_for_idle(context=None, gpus="device=1", retries=6, shared_device=True)
    assert snap["compute_apps"] == apps
    assert "parakeet 3.0 GiB" in snap["foreign_load"]
    assert "util 14%" in snap["foreign_load"]
    assert snap["idle"] is False
    try:
        wait_for_idle(context=None, gpus="device=1", retries=0, shared_device=False)
    except GuardError as exc:
        assert "not idle" in str(exc)
    else:
        raise AssertionError("exclusive GPU must refuse foreign compute apps")


def test_rtx3070_screen_shared_load_is_visible_and_report_only(tmp_path: Path):
    path = tmp_path / "rtx3070.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ab",
                "verdict": "block",
                "dataset": "v1",
                "context": {
                    "target": "rtx3070",
                    "shared_device": True,
                    "load_before": {
                        "shared_device": True,
                        "foreign_load": "parakeet 3.0 GiB, util 14%",
                        "compute_apps": [
                            {
                                "name": "parakeet",
                                "used_gpu_memory": "3054 MiB",
                            }
                        ],
                    },
                },
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = {
        "accuracy": {"verdict": "pass"},
        "v100s": {"verdict": "pass"},
        "rtx3070": json.loads(path.read_text(encoding="utf-8")),
        "rtx5060ti": None,
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "warn"
    text, overall = render(
        jobs={"accuracy": None, "v100s": None, "rtx3070": path, "rtx5060ti": None, "cpu": None},
        run_url="https://example.test/run",
        notices=[],
        trusted=True,
        run_benches=True,
    )
    assert overall == "warn"
    assert "shared GPU, foreign load: parakeet 3.0 GiB" in text
    body = summarize(json.loads(path.read_text(encoding="utf-8")))
    assert "shared GPU, foreign load: parakeet 3.0 GiB" in body


def test_v100s_inconclusive_is_report_only():
    loaded = {
        "accuracy": {"verdict": "pass"},
        "v100s": {"verdict": "inconclusive"},
        "rtx3070": {"verdict": "pass"},
        "rtx5060ti": {"verdict": "error"},
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "warn"


def test_accuracy_failure_still_fails_overall():
    loaded = {
        "accuracy": {"verdict": "block"},
        "v100s": {"verdict": "pass"},
        "rtx3070": {"verdict": "pass"},
        "rtx5060ti": None,
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "block"


def test_paired_log_ratio_ci_cancels_drift():
    # Base and head drift together (co-hosted load ramps up); per-round ratio is 1.05.
    base = [1.0, 1.2, 1.5, 0.9, 1.3, 1.1]
    head = [b * 1.05 for b in base]
    ratio, lo, hi, t = paired_log_ratio_ci(base, head)
    assert abs(ratio - 1.05) < 1e-9
    assert abs(lo - 1.05) < 1e-9 and abs(hi - 1.05) < 1e-9
    assert t > 1e6
    same = [1.0, 1.1, 0.9, 1.05]
    ratio, lo, hi, t = paired_log_ratio_ci(same, list(same))
    assert ratio == 1.0 and lo == 1.0 and hi == 1.0 and t == 0.0


def test_paired_log_ratio_ci_widens_with_noise():
    base = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    head = [1.02, 0.97, 1.08, 1.01, 0.95, 1.06]
    ratio, lo, hi, _ = paired_log_ratio_ci(base, head)
    assert lo < 1.0 < hi
    assert lo < ratio < hi
    assert abs(t975(5) - 2.571) < 1e-9
    assert t975(1000) < 1.97


def test_flagged_ids_drive_the_retry():
    payload = {
        "verdict": "block",
        "models": [
            {"id": "das9201", "verdict": "pass"},
            {"id": "das9701", "verdict": "block"},
            {"id": "nus9601", "verdict": "inconclusive"},
            {"id": "edf9203", "verdict": "warn"},
        ],
    }
    assert _flagged_ids(payload) == {"das9701", "nus9601"}
    assert _payload_flagged(payload) is True
    assert _payload_flagged({"verdict": "pass", "models": [{"id": "a", "verdict": "pass"}]}) is False
    assert _payload_flagged({"verdict": "inconclusive", "models": []}) is True


def test_rtx5060ti_error_does_not_fail_overall():
    loaded = {
        "accuracy": {"verdict": "pass"},
        "v100s": {"verdict": "pass"},
        "rtx3070": {"verdict": "pass"},
        "rtx5060ti": {"verdict": "error", "kind": "ab"},
        "cpu": {"verdict": "pass"},
    }
    assert overall_verdict(loaded) == "warn"


def test_classify_oom_and_unsupported_jit():
    assert classify_run_failure("CUDA_ERROR_OUT_OF_MEMORY") == "error"
    assert classify_run_failure("out of memory on device") == "error"
    assert classify_run_failure("no kernel image is available for execution on the device") == "unsupported"
    assert classify_run_failure("ptxas fatal   : Not supported") == "unsupported"
    assert classify_run_failure("segfault") == "block"
    assert apply_report_only("block", report_only=True) == "warn"
    assert apply_report_only("error", report_only=True) == "error"
    assert combine_verdicts(["unsupported", "unsupported"]) == "unsupported"
    assert combine_verdicts(["pass", "unsupported"]) == "pass"


def test_gpu_profiler_fallback_writes_sample_s_ms():
    profiler = _gpu_profiler_with_fallback({"sample_s": 0.4, "profiler": {}})
    assert abs(profiler["gpu_total_ms"] - 400.0) < 1e-6
    assert profiler["gpu_total_source"] == "sample_s"
    kept = _gpu_profiler_with_fallback(
        {"sample_s": 0.4, "profiler": {"gpu_total_ms": 12.5, "total_gpu_kernel_ms": 12.5}}
    )
    assert kept["gpu_total_ms"] == 12.5
    assert "gpu_total_source" not in kept


def test_collect_timings_falls_back_to_sample_s_without_profiler():
    runs = [
        {"sample_s": 0.012, "profiler": {}},
        {"sample_s": 0.013, "wall_s": 0.04},
    ]
    values, err = collect_timings(runs, gpu=True, kind="total")
    assert err is None
    assert values is not None
    assert abs(values[0] - 12.0) < 1e-6
    assert abs(values[1] - 13.0) < 1e-6


def test_rtx5060ti_comment_is_report_only_shared(tmp_path: Path):
    path = tmp_path / "rtx5060ti.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ab",
                "verdict": "warn",
                "dataset": "v1",
                "context": {
                    "target": "rtx5060ti",
                    "shared_device": True,
                    "report_only": True,
                    "load_before": {
                        "shared_device": True,
                        "foreign_load": "EngineCore 3.6 GiB, python 3.5 GiB, util 0%",
                    },
                },
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    text, overall = render(
        jobs={"accuracy": None, "v100s": None, "rtx3070": None, "rtx5060ti": path, "cpu": None},
        run_url="https://example.test/run",
        notices=[],
        trusted=True,
        run_benches=True,
    )
    assert overall == "warn"
    assert "shared GPU" in text
    assert "report-only" in text
    assert "EngineCore 3.6 GiB" in text
    assert "never blocks" in text
