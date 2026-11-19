#!/usr/bin/env python3
"""Fail if a self-hosted job can run on pull_request in an unapproved way.

Public repo + org self-hosted runners. In PR-triggered workflows the only
allowed self-hosted runs-on is APPROVED_PR_RUNS_ON: the `fuzzybits-ci` group
(this repository only) for same-repo, non-Dependabot PRs, with GitHub-hosted
ubuntu-24.04 for fork and Dependabot PRs. Fork PR runs also need maintainer
approval (repository setting). GPU runners and every other group stay
push/workflow_dispatch/schedule only (gpu.yml, images.yml).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("error: PyYAML is required (pip install pyyaml)\n")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

SELF_HOSTED_MARKERS = (
    "self-hosted",
    "sm_70",
    "sm_75",
    "sm_86",
    "sm_120",
    "v100s",
    "rtx_3070",
    "gtx_1660s",
    "rtx_5060ti",
    "rtx_5060ti_8gb",
    "fzb-xerxes",
    "shelly-1",
    "gpu-runners",
    "public-ci",
    "fuzzybits-ci",
    "fuzzybits-bench",
    "fzb-bench",
)

# The one self-hosted runs-on allowed in PR-triggered workflows. Changing it
# must change this guard too.
APPROVED_PR_RUNS_ON = (
    "${{ (github.event_name != 'pull_request' || (github.event.pull_request.head.repo.full_name == github.repository"
    " && github.event.pull_request.user.login != 'dependabot[bot]'))"
    " && fromJSON('{\"group\":\"fuzzybits-ci\",\"labels\":[\"self-hosted\",\"Linux\",\"X64\"]}') || 'ubuntu-24.04' }}"
)

FORBIDDEN_TRIGGERS = ("pull_request_target", "workflow_run")
PR_TRIGGERS = ("pull_request", "pull_request_target")


def _on_keys(on) -> set[str]:
    if on is None:
        return set()
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        keys: set[str] = set()
        for item in on:
            if isinstance(item, str):
                keys.add(item)
            elif isinstance(item, dict):
                keys.update(item.keys())
        return keys
    if isinstance(on, dict):
        return set(on.keys())
    return set()


def _strings(node) -> list[str]:
    out: list[str] = []
    if node is None:
        return out
    if isinstance(node, str):
        return [node]
    if isinstance(node, (int, float, bool)):
        return [str(node)]
    if isinstance(node, list):
        for item in node:
            out.extend(_strings(item))
        return out
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_strings(v))
        return out
    return out


def _is_self_hosted(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in SELF_HOSTED_MARKERS)


def _job_is_self_hosted(job: dict) -> bool:
    if not isinstance(job, dict):
        return False
    for s in _strings(job.get("runs-on")):
        if _is_self_hosted(s):
            return True
    strategy = job.get("strategy") or {}
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    for s in _strings(matrix):
        if _is_self_hosted(s):
            return True
    return False


def main() -> int:
    if not WORKFLOWS.is_dir():
        sys.stderr.write(f"error: missing {WORKFLOWS}\n")
        return 2

    errors: list[str] = []
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    if not files:
        sys.stderr.write(f"error: no workflow files in {WORKFLOWS}\n")
        return 2

    for path in files:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            errors.append(f"{path.name}: not a mapping")
            continue
        # PyYAML (YAML 1.1) parses the key `on` as the boolean True.
        on_keys = _on_keys(data.get("on", data.get(True)))
        rel = f".github/workflows/{path.name}"

        for forbidden in FORBIDDEN_TRIGGERS:
            if forbidden in on_keys:
                errors.append(
                    f"{rel}: trigger {forbidden!r} is forbidden "
                    "(never check out or execute PR code via these events)"
                )

        jobs = data.get("jobs") or {}
        if not isinstance(jobs, dict):
            continue
        self_hosted_jobs = [
            name for name, job in jobs.items()
            if _job_is_self_hosted(job)
            and not (isinstance(job, dict) and job.get("runs-on") == APPROVED_PR_RUNS_ON
                     and not _strings((job.get("strategy") or {}).get("matrix")
                                      if isinstance(job.get("strategy"), dict) else None))
        ]
        if self_hosted_jobs and (on_keys & set(PR_TRIGGERS)):
            errors.append(
                f"{rel}: self-hosted job(s) {self_hosted_jobs} are reachable "
                f"from {sorted(on_keys & set(PR_TRIGGERS))} — split them into "
                "a push/workflow_dispatch/schedule workflow, or use APPROVED_PR_RUNS_ON"
            )

    if errors:
        sys.stderr.write("self-hosted reachability check FAILED:\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        return 1

    print(f"ok: {len(files)} workflow file(s); PR-reachable self-hosted jobs use only the approved fuzzybits-ci routing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
