"""Regenerate ``datasets/v1/manifest.json`` from the tree. Do not edit v1 in place."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

from ..loader import (
    DEFAULT_CPU_ACCURACY_S,
    DEFAULT_CPU_COMPILE_S,
    DEFAULT_CPU_PERF_S,
    DEFAULT_GPU_PERF_S,
    FZB_1E8,
    INFEASIBLE_HITS,
    LOW_POWER_HITS,
    NUS9601_CPU_ACCURACY_S,
    NUS9601_CPU_PERF_S,
    NUS9601_GPU_PERF_S,
    NUS9601_REF,
    NUS9601_REF_SOURCE,
    ORACLE_REL,
    PERF_REASONS,
    CPU_PERF_TRIALS,
    PERF_TRIALS,
    PWR_SUBMODULE,
    SAMPLER_MIN_P,
    TARGET_HITS,
    accuracy_trials,
    classify_accuracy,
    expected_hits,
    manifest_path,
    repo_root,
    sha256_file,
)
from .compat import SHIM_ID, SHIM_VERSION

GATE_RE = re.compile(r'<define-gate\s+name="([^"]+)"')
FT_RE = re.compile(r'<define-fault-tree\s+name="([^"]+)"')
PWR_NAME_RE = re.compile(r"converted_(.+)_et_(Grp-\d+)_")

# Remaining PWR files that the dataset-side generic_pwr shim cannot reconstruct.
# Shim v1 compiles EQK-BIN4/Grp-5 (G10/G3) and EQK-BIN5/Grp-2 (F44→FE44 + G3).
# Grp-6 still fails after G3 clones: cyclic FE56/FE57 fork nesting.
KNOWN_COMPILE_FAILURES: dict[str, dict] = {
    "EQK-BIN4/Grp-6": {
        "expected": "validity_error",
        "missing_elements": ["FE56", "FE57"],
        "reason": (
            "After cloning private G3 into referencing fault trees, MEF initialize "
            "raises ValidityError: Functional event FE56 must appear after "
            "functional event FE57 in event tree fork paths. The exported event "
            "tree nests FE56 then FE57 on some paths and FE57 then FE56 on others, "
            "so no define-functional-event reorder can satisfy "
            "CheckFunctionalEventOrder. Not reconstructable from JSInp gate/BE "
            "data without rewriting the event tree. Dataset, not the engine."
        ),
    },
}


def _first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        raise SystemExit("no match for " + pattern.pattern)
    return match.group(1)


def _git_submodule_sha(root: Path, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _git_actual_submodule_sha(root: Path, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root / path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def freeze(root: Path | None = None) -> Path:
    repo = root or repo_root()
    oracle = json.loads((repo / ORACLE_REL).read_text(encoding="utf-8"))
    pwr_sha = _git_actual_submodule_sha(repo, PWR_SUBMODULE)
    models: list[dict] = []

    for xml in sorted((repo / "input" / "Aralia").glob("*.xml")):
        text = xml.read_text(encoding="utf-8")
        model_id = xml.stem
        tags = ["accuracy"]
        notes = None
        cpu_acc = DEFAULT_CPU_ACCURACY_S
        cpu_perf = DEFAULT_CPU_PERF_S
        gpu_perf = DEFAULT_GPU_PERF_S
        if model_id in PERF_REASONS:
            tags.append("perf")
        if model_id == "nus9601":
            probability = NUS9601_REF
            reference = {
                "probability": probability,
                "source": NUS9601_REF_SOURCE,
                "kind": "exact",
                "uncertainty": 0.0,
                "citation": "https://github.com/fuzzybit-org/fuzzybits/issues/120",
            }
            tags = ["accuracy", "perf", "hard"]
            cpu_acc = NUS9601_CPU_ACCURACY_S
            cpu_perf = NUS9601_CPU_PERF_S
            gpu_perf = NUS9601_GPU_PERF_S
            n_acc = accuracy_trials(probability)
            notes = (
                f"Accuracy n is {n_acc} (96·2^20, same as fzb --trials 1e8) so n·p ≈ "
                f"{expected_hits(n_acc, probability):.0f}. CPU accuracy budget is "
                f"{cpu_acc}s (3× the 60s default) because the sampler walks 1,567 "
                "events. fzb inspect --exact refuses 1,567 SAMPLE events (cap 30); "
                "scram-cli --bdd/--zbdd OOMs on this model."
            )
        else:
            if model_id not in oracle:
                raise SystemExit(f"no BDD oracle for {model_id}")
            probability = float(oracle[model_id])
            reference = {
                "probability": probability,
                "source": "bdd",
                "kind": "exact",
                "uncertainty": 0.0,
                "citation": str(ORACLE_REL),
            }
            n_acc = accuracy_trials(probability)
            hits = expected_hits(n_acc, probability)
            if hits < TARGET_HITS:
                tags.append("hard")
                notes = (
                    f"n·p = {hits:.3g} at the {n_acc} trial cap (fzb --trials 1e8); "
                    f"{TARGET_HITS} hits would need n ≈ {math.ceil(TARGET_HITS / probability):.3g}, "
                    "which is outside the accuracy time budget."
                )
        spec = classify_accuracy(probability, n_acc)
        models.append(
            {
                "id": model_id,
                "path": str(xml.relative_to(repo)),
                "sha256": sha256_file(xml),
                "source": "aralia",
                "submodule_commit": None,
                "query": _first(GATE_RE, text),
                "query_note": None,
                "reference": reference,
                "tags": tags,
                "perf_reason": PERF_REASONS.get(model_id),
                "notes": notes,
                "accuracy": {"check": spec.check, "reason": spec.reason},
                "trials": {
                    "accuracy": n_acc,
                    "perf": PERF_TRIALS if "perf" in tags else None,
                    "cpu_perf": CPU_PERF_TRIALS.get(model_id) if "perf" in tags else None,
                },
                "expected_hits_accuracy": expected_hits(n_acc, probability),
                "time_budget_s": {
                    "cpu_accuracy": cpu_acc,
                    "cpu_perf": cpu_perf if "perf" in tags else DEFAULT_CPU_PERF_S,
                    "gpu_perf": gpu_perf if "perf" in tags else DEFAULT_GPU_PERF_S,
                    "cpu_compile": DEFAULT_CPU_COMPILE_S,
                },
            }
        )

    pwr_dir = repo / PWR_SUBMODULE / "data" / "openpsa"
    xmls = sorted(pwr_dir.glob("converted_*.xml"))
    if len(xmls) != 95:
        raise SystemExit(f"expected 95 PWR OpenPSA XMLs, found {len(xmls)} in {pwr_dir}")
    for xml in xmls:
        text = xml.read_text(encoding="utf-8")
        match = PWR_NAME_RE.search(xml.name)
        short = f"{match.group(1)}/{match.group(2)}" if match else xml.stem
        known = KNOWN_COMPILE_FAILURES.get(short)
        notes = (
            "v1 is compile/size/throughput only. SAPHIRE MCS probabilities "
            "are mostly 1e-10 to 1e-6 (README); values below 2.3e-10 "
            "quantize to zero (#17). Revisit accuracy with #92, #17, #6. "
            f"The generic_pwr shim v{SHIM_VERSION} rewrites a cached MEF copy "
            "from the matching SAPHSOLVE JSInp (and BE CSV); the submodule XML "
            "is not modified."
        )
        if known:
            notes = notes + " Known compile failure: " + known["reason"]
        model_row = {
            "id": short,
            "path": str(xml.relative_to(repo)),
            "sha256": sha256_file(xml),
            "source": "pwr",
            "submodule_commit": pwr_sha,
            "query": _first(GATE_RE, text),
            "query_note": (
                f"first fault-tree {_first(FT_RE, text)!r} top; event-tree "
                "sequences are not compiled (#92). SAPHIRE MCS references "
                "are not used for accuracy in v1."
            ),
            "reference": None,
            "tags": ["pwr"],
            "accuracy": {
                "check": "n/a",
                "reason": (
                    "v1 PWR is compile/size/throughput only; SAPHIRE MCS "
                    "references are not a 4-sigma gate (#92, #17, #6)"
                ),
            },
            "perf_reason": None,
            "notes": notes,
            "trials": {"accuracy": None, "perf": PERF_TRIALS, "cpu_perf": None},
            "expected_hits_accuracy": None,
            "time_budget_s": {
                "cpu_accuracy": DEFAULT_CPU_ACCURACY_S,
                "cpu_perf": DEFAULT_CPU_PERF_S,
                "gpu_perf": DEFAULT_GPU_PERF_S,
                "cpu_compile": DEFAULT_CPU_COMPILE_S,
            },
            "compat": {"shim": SHIM_ID, "version": SHIM_VERSION},
        }
        if known:
            model_row["compile"] = {
                "expected": known["expected"],
                "missing_elements": list(known["missing_elements"]),
                "reason": known["reason"],
            }
        models.append(model_row)

    payload = {
        "schema_version": 1,
        "dataset": "v1",
        "description": (
            "Frozen Aralia accuracy set (43 models, exact top-event probabilities) "
            "plus generic-pwr-model (95 event-tree-group XMLs, compile/size/throughput "
            "only). synthetic-models is excluded (no license)."
        ),
        "versioning": (
            "A dataset version never changes once used. Schema or contents changes "
            "are a new version (v2, …). The harness, PR gate (#122) and published "
            "history (#123) pin a dataset id."
        ),
        "submodules": {PWR_SUBMODULE: pwr_sha},
        "defaults": {
            "accuracy_target_hits": TARGET_HITS,
            "accuracy_infeasible_hits": INFEASIBLE_HITS,
            "accuracy_low_power_hits": LOW_POWER_HITS,
            "accuracy_sampler_min_p": SAMPLER_MIN_P,
            "accuracy_min_trials": 1 << 20,
            "accuracy_max_trials": FZB_1E8,
            "perf_trials": PERF_TRIALS,
            "time_budget_s": {
                "cpu_accuracy": DEFAULT_CPU_ACCURACY_S,
                "cpu_perf": DEFAULT_CPU_PERF_S,
                "gpu_perf": DEFAULT_GPU_PERF_S,
                "cpu_compile": DEFAULT_CPU_COMPILE_S,
            },
            "compile": {"pdag_c": 2, "no_kn": True},
        },
        "models": models,
    }
    out = manifest_path(repo)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    del argv
    out = freeze()
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
