"""Reachability guard: approved PR GPU routing vs GPU labels elsewhere."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "check_self_hosted_reachability",
    ROOT / "scripts" / "ci" / "check-self-hosted-reachability.py",
)
assert _SPEC and _SPEC.loader
reach = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reach)


def _write_workflow(tmp_path: Path, name: str, data: dict) -> Path:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_approved_fuzzybits_ci_expression_matches_ci_yml():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert reach.APPROVED_PR_RUNS_ON_CI in text


def test_approved_gpu_and_cpu_expressions_are_in_bench_yml():
    bench = ROOT / ".github" / "workflows" / "bench.yml"
    if not bench.is_file():
        pytest.skip("bench.yml not in this tree")
    text = bench.read_text(encoding="utf-8")
    assert reach.APPROVED_PR_RUNS_ON_V100S in text
    assert reach.APPROVED_PR_RUNS_ON_RTX3070 in text
    assert reach.APPROVED_PR_RUNS_ON_RTX5060TI in text
    assert reach.APPROVED_PR_RUNS_ON_BENCH_CPU in text
    assert reach.APPROVED_PR_RUNS_ON_CI in text


def test_approved_gpu_routing_on_pull_request_passes(tmp_path: Path):
    _write_workflow(
        tmp_path,
        "bench.yml",
        {
            "on": {"pull_request": None, "workflow_dispatch": None},
            "jobs": {
                "bench-perf-v100s": {"runs-on": reach.APPROVED_PR_RUNS_ON_V100S},
                "bench-perf-rtx3070": {"runs-on": reach.APPROVED_PR_RUNS_ON_RTX3070},
                "bench-perf-rtx5060ti": {"runs-on": reach.APPROVED_PR_RUNS_ON_RTX5060TI},
                "bench-perf-cpu": {"runs-on": reach.APPROVED_PR_RUNS_ON_BENCH_CPU},
                "linux": {"runs-on": reach.APPROVED_PR_RUNS_ON_CI},
            },
        },
    )
    assert reach.check_workflows(tmp_path) == []


def test_raw_gpu_labels_on_pull_request_fail(tmp_path: Path):
    _write_workflow(
        tmp_path,
        "gpu.yml",
        {
            "on": ["pull_request"],
            "jobs": {
                "sm70": {
                    "runs-on": ["self-hosted", "Linux", "X64", "nvidia", "sm_70", "v100s"]
                }
            },
        },
    )
    errors = reach.check_workflows(tmp_path)
    assert errors
    assert "sm70" in errors[0]


def test_public_ci_gpu_without_same_repo_guard_fails(tmp_path: Path):
    _write_workflow(
        tmp_path,
        "oops.yml",
        {
            "on": {"pull_request": None},
            "jobs": {
                "bench": {
                    "runs-on": "${{ fromJSON('{\"group\":\"public-ci\",\"labels\":[\"self-hosted\",\"Linux\",\"X64\",\"nvidia\",\"sm_70\",\"v100s\"]}') }}"
                }
            },
        },
    )
    errors = reach.check_workflows(tmp_path)
    assert errors
    assert "bench" in errors[0]


def test_gpu_labels_on_push_only_workflow_pass(tmp_path: Path):
    _write_workflow(
        tmp_path,
        "gpu.yml",
        {
            "on": {"push": {"branches": ["develop"]}, "workflow_dispatch": None},
            "jobs": {
                "sm70": {
                    "runs-on": ["self-hosted", "Linux", "X64", "nvidia", "sm_70", "v100s"]
                }
            },
        },
    )
    assert reach.check_workflows(tmp_path) == []


def test_dependabot_unconditional_gpu_group_fails(tmp_path: Path):
    # Missing the dependabot/same-repo clause: any PR (including forks) would hit GPUs.
    _write_workflow(
        tmp_path,
        "bench.yml",
        {
            "on": ["pull_request"],
            "jobs": {
                "v100s": {
                    "runs-on": [
                        "self-hosted",
                        "Linux",
                        "X64",
                        "nvidia",
                        "sm_70",
                        "v100s",
                    ]
                }
            },
        },
    )
    errors = reach.check_workflows(tmp_path)
    assert errors


def test_real_workflows_dir_passes():
    pytest.importorskip("yaml")
    errors = reach.check_workflows(ROOT / ".github" / "workflows")
    assert errors == []
