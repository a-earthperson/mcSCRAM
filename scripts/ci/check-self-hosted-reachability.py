#!/usr/bin/env python3
"""Fail if a self-hosted job can run on pull_request in an unapproved way.

Public repo + org self-hosted runners. In PR-triggered workflows the only
allowed self-hosted runs-on values are in APPROVED_PR_RUNS_ON:

* ``fuzzybits-ci`` (this repository only) for same-repo, non-Dependabot PRs,
  with GitHub-hosted ubuntu-24.04 for fork and Dependabot PRs.
* Bench GPU jobs in bench.yml: ``public-ci`` + V100S / RTX 3070 / RTX 5060 Ti
  labels, same same-repo non-Dependabot expression. GPU labels anywhere else
  in a PR-triggered workflow stay forbidden (#122).
* Bench CPU job: ``fuzzybits-bench`` group, same expression.

Fork PR runs also need maintainer approval (repository setting). Other GPU
workflows stay push/workflow_dispatch/schedule only (gpu.yml, images.yml).
The ``swarm-managers`` group (deploy-live in server-image.yml) has no PR
routing at all: it runs on release tags only.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    "swarm-managers",
    "swarm-manager",
    "fzb-gtw",
)

# Same-repo, non-Dependabot PRs (and non-pull_request events) may use a
# self-hosted group; fork and Dependabot PRs fall back to ubuntu-24.04.
# Changing the expression must change this guard too.


def same_repo_non_dependabot_runs_on(from_json: str) -> str:
    """GitHub-expression runs-on used by ci.yml and bench.yml."""
    # Not an f-string on the closer: `}}` in an f-string is a single `}`.
    return (
        "${{ (github.event_name != 'pull_request' || (github.event.pull_request.head.repo.full_name == github.repository"
        " && github.event.pull_request.user.login != 'dependabot[bot]'))"
        " && fromJSON('" + from_json + "') || 'ubuntu-24.04' }}"
    )


APPROVED_PR_RUNS_ON_CI = same_repo_non_dependabot_runs_on(
    '{"group":"fuzzybits-ci","labels":["self-hosted","Linux","X64"]}'
)
APPROVED_PR_RUNS_ON_V100S = same_repo_non_dependabot_runs_on(
    '{"group":"public-ci","labels":["self-hosted","Linux","X64","nvidia","sm_70","v100s"]}'
)
APPROVED_PR_RUNS_ON_RTX3070 = same_repo_non_dependabot_runs_on(
    '{"group":"public-ci","labels":["self-hosted","Linux","X64","nvidia","sm_86","rtx_3070"]}'
)
APPROVED_PR_RUNS_ON_RTX5060TI = same_repo_non_dependabot_runs_on(
    '{"group":"public-ci","labels":["self-hosted","Linux","X64","nvidia","fast","sm_120","rtx_5060ti_8gb"]}'
)
APPROVED_PR_RUNS_ON_BENCH_CPU = same_repo_non_dependabot_runs_on(
    '{"group":"fuzzybits-bench","labels":["self-hosted","Linux","X64"]}'
)

# Back-compat alias: the original single approved routing (fuzzybits-ci).
APPROVED_PR_RUNS_ON = APPROVED_PR_RUNS_ON_CI

APPROVED_PR_RUNS_ON_ALL = frozenset(
    {
        APPROVED_PR_RUNS_ON_CI,
        APPROVED_PR_RUNS_ON_V100S,
        APPROVED_PR_RUNS_ON_RTX3070,
        APPROVED_PR_RUNS_ON_RTX5060TI,
        APPROVED_PR_RUNS_ON_BENCH_CPU,
    }
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


def _job_runs_on_approved(job: dict) -> bool:
    if not isinstance(job, dict):
        return False
    runs_on = job.get("runs-on")
    if not isinstance(runs_on, str) or runs_on not in APPROVED_PR_RUNS_ON_ALL:
        return False
    strategy = job.get("strategy") or {}
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    if _strings(matrix):
        # Matrix must not smuggle extra self-hosted labels.
        return False
    return True


def _load_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("error: PyYAML is required (pip install pyyaml)") from exc
    return yaml


def check_workflows(workflows_dir: Path) -> list[str]:
    """Return human-readable errors for workflow files under ``workflows_dir``."""
    yaml = _load_yaml()
    errors: list[str] = []
    files = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    if not files:
        return [f"error: no workflow files in {workflows_dir}"]

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
            name
            for name, job in jobs.items()
            if _job_is_self_hosted(job) and not _job_runs_on_approved(job)
        ]
        if self_hosted_jobs and (on_keys & set(PR_TRIGGERS)):
            errors.append(
                f"{rel}: self-hosted job(s) {self_hosted_jobs} are reachable "
                f"from {sorted(on_keys & set(PR_TRIGGERS))} — split them into "
                "a push/workflow_dispatch/schedule workflow, or use an "
                "APPROVED_PR_RUNS_ON expression (fuzzybits-ci, public-ci GPU "
                "bench, or fuzzybits-bench CPU; same-repo non-Dependabot only)"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    workflows = WORKFLOWS
    if argv:
        workflows = Path(argv[0])
    if not workflows.is_dir():
        sys.stderr.write(f"error: missing {workflows}\n")
        return 2

    errors = check_workflows(workflows)
    files = list(workflows.glob("*.yml")) + list(workflows.glob("*.yaml"))
    if errors:
        sys.stderr.write("self-hosted reachability check FAILED:\n")
        for e in errors:
            sys.stderr.write(f"  - {e}\n")
        return 1

    print(
        f"ok: {len(files)} workflow file(s); PR-reachable self-hosted jobs use "
        "only approved same-repo non-Dependabot routing "
        "(fuzzybits-ci, public-ci GPU bench, fuzzybits-bench CPU)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
