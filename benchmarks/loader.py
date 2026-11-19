"""Load and validate a frozen benchmark dataset manifest (#120)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
DATASET_ID = "v1"
ALLOWED_TAGS = frozenset({"accuracy", "perf", "hard", "pwr"})
ALLOWED_SOURCES = frozenset({"aralia", "pwr"})
TARGET_HITS = 1000
# Below this expected-hit count a 4-sigma test has no power: 0 hits yields
# |z| = sqrt(n p_ref) < 4. The owner's "e.g. 100" is the same order of
# magnitude; 16 is the vacuous-pass bound, so isp9607 (n·p ≈ 96) is low-power
# rather than infeasible.
INFEASIBLE_HITS = 16
LOW_POWER_HITS = TARGET_HITS
# 2^-32 Bernoulli high-word threshold (#17): events with p below this become 0.
SAMPLER_MIN_P = 2.0 ** -32  # 2.3283064365386963e-10
# Engine pass size is 2^20; ``fzb run --trials 1e8`` actually executes 96·2^20.
TRIAL_UNIT = 1 << 20
FZB_1E8 = 96 << 20  # 100_663_296
MIN_ACCURACY_TRIALS = TRIAL_UNIT
MAX_ACCURACY_TRIALS = FZB_1E8
PERF_TRIALS = FZB_1E8
# CPU A/B sizes sampling to ~2–5 s (gateway i7-9700, in-process fuzzybits.run).
# Scaled from 1e8 probes: chinese 1.34 s, das9201 5.40 s, das9701 28.4 s (timeout at 30 s).
CPU_PERF_TRIALS: dict[str, int] = {
    "chinese": 225 << 20,  # ~3.1 s
    "das9201": 80 << 20,   # ~4.5 s
    "das9701": 16 << 20,   # ~4.7 s
    "edf9203": 32 << 20,   # unprobed; conservative vs das9201
    "nus9601": 8 << 20,    # 1e8 is ~minutes; keep the sample short
}
DEFAULT_CPU_ACCURACY_S = 60
DEFAULT_CPU_PERF_S = 60
DEFAULT_GPU_PERF_S = 15
DEFAULT_CPU_COMPILE_S = 120
# nus9601 walks 1,567 SAMPLE events; 1e8 trials need a longer CPU budget.
NUS9601_CPU_ACCURACY_S = 180
NUS9601_CPU_PERF_S = 180
NUS9601_GPU_PERF_S = 30
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NUS9601_REF = 9.94453321e-06
NUS9601_REF_SOURCE = "nombrillo (Kirill Shumilov, @kshumilov)"
ORACLE_REL = Path("tools/server/src/fuzzybits_server/model_oracle_p.json")
PWR_SUBMODULE = "input/generic-pwr-model"

PERF_REASONS: dict[str, str] = {
    "das9201": (
        "Sampler-dominated: on V100S at 1e8 trials, basic_event is ~89% of GPU "
        "kernel time (37.6 of 42.1 ms, PR #91). Gate kernels are a small tail."
    ),
    "das9701": (
        "Largest Aralia gate counts. The fused-kernel prototype in #20 is fastest "
        "here (3.11× vs unfused at -c 2), so gate-kernel and occupancy work shows up."
    ),
    "nus9601": (
        "1,567 basic events (largest Aralia sampler). #20: the fused BE tile does "
        "not fit in shared memory and falls back to per-work-group global scratch."
    ),
    "edf9203": (
        "Known fused-path regressor in #20 (2.2× slower than unfused at TB=8, 160 "
        "regs, chunking overhead). Guards a per-model fallback."
    ),
    "chinese": (
        "Small model; baseline for launch, JIT, and host overhead. Kernel time is "
        "~9–13 ms at 1e8, so fixed costs are visible."
    ),
}


class ManifestError(ValueError):
    """Manifest failed structural or on-disk validation."""


def repo_root(start: Path | None = None) -> Path:
    """Return the fuzzybits repository root (directory that contains ``input/``)."""
    here = Path(__file__).resolve().parent
    if start is not None:
        here = start.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "input" / "Aralia").is_dir() and (candidate / "toolchain.env").is_file():
            return candidate
    raise ManifestError("could not locate fuzzybits repository root from " + str(here))


def dataset_dir(root: Path | None = None, dataset: str = DATASET_ID) -> Path:
    return (root or repo_root()) / "benchmarks" / "datasets" / dataset


def manifest_path(root: Path | None = None, dataset: str = DATASET_ID) -> Path:
    return dataset_dir(root, dataset) / "manifest.json"


def schema_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "benchmarks" / "schemas" / "dataset-manifest.v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_trials(n: int, *, unit: int = TRIAL_UNIT, cap: int = MAX_ACCURACY_TRIALS) -> int:
    n = max(int(n), MIN_ACCURACY_TRIALS)
    rounded = ((n + unit - 1) // unit) * unit
    return min(rounded, cap)


def accuracy_trials(probability: float, *, cap: int = MAX_ACCURACY_TRIALS) -> int:
    """Trials so that n·p ≥ 1000, capped at ``fzb --trials 1e8`` (96·2^20)."""
    if not math.isfinite(probability) or probability <= 0.0:
        raise ManifestError(f"accuracy_trials: probability must be in (0, 1], got {probability}")
    return round_trials(math.ceil(TARGET_HITS / probability), cap=cap)


def expected_hits(trials: int, probability: float) -> float:
    return float(trials) * float(probability)


def infeasible_accuracy_reason(
    probability: float,
    trials: int,
    *,
    infeasible_hits: float = INFEASIBLE_HITS,
) -> str | None:
    """Why a 4-sigma check is not a CI gate, or None if it has any power.

    Rare-event methods (#6, #17, #91) are out of scope for this dataset. GPUs
    help by a few orders of magnitude and still do not reach n·p ≈ 1000 here.
    """
    reasons: list[str] = []
    if probability < SAMPLER_MIN_P:
        reasons.append(
            f"p_ref {probability:.6g} is below the 2^-32 Bernoulli threshold "
            f"(~{SAMPLER_MIN_P:.3g}, #17); the sampler represents those events as 0"
        )
    hits = expected_hits(trials, probability)
    if hits < infeasible_hits:
        need = math.ceil(TARGET_HITS / probability) if probability > 0.0 else float("inf")
        reasons.append(
            f"expected hits n·p_ref = {hits:.3g} at the budgeted {trials} trials "
            f"(far below {infeasible_hits:g}; {TARGET_HITS} hits would need n ≈ {need:.3g}). "
            "Rare-event methods are #6 / #17 / #91, not a larger CI budget"
        )
    if not reasons:
        return None
    return "; ".join(reasons)


@dataclass(frozen=True)
class AccuracySpec:
    """Planned accuracy-check status. ``infeasible`` is visible, never a pass."""

    check: str  # feasible | low_power | infeasible | n/a
    reason: str | None = None

    @property
    def gated(self) -> bool:
        """Counts toward the accuracy gate (pass / pass-low-power / block)."""
        return self.check in {"feasible", "low_power"}


def classify_accuracy(
    probability: float | None,
    trials: int | None,
    *,
    infeasible_hits: float = INFEASIBLE_HITS,
    low_power_hits: float = LOW_POWER_HITS,
) -> AccuracySpec:
    if probability is None or trials is None:
        return AccuracySpec(
            check="n/a",
            reason="not an accuracy model in this dataset version",
        )
    reason = infeasible_accuracy_reason(
        probability, trials, infeasible_hits=infeasible_hits
    )
    if reason:
        return AccuracySpec(check="infeasible", reason=reason)
    hits = expected_hits(trials, probability)
    if hits < low_power_hits:
        return AccuracySpec(
            check="low_power",
            reason=(
                f"expected hits n·p_ref = {hits:.3g} at {trials} trials "
                f"(below {low_power_hits:g}; the 4-sigma check is underpowered)"
            ),
        )
    return AccuracySpec(check="feasible", reason=None)


@dataclass(frozen=True)
class Reference:
    probability: float | None
    source: str | None
    kind: str | None = None
    uncertainty: float = 0.0
    citation: str | None = None

    @property
    def is_exact(self) -> bool:
        return self.probability is not None and float(self.uncertainty) == 0.0


@dataclass(frozen=True)
class Model:
    id: str
    path: str
    sha256: str
    source: str
    query: str
    tags: tuple[str, ...]
    trials_accuracy: int | None
    trials_perf: int | None
    time_budget_s: Mapping[str, int]
    reference: Reference
    submodule_commit: str | None = None
    perf_reason: str | None = None
    notes: str | None = None
    query_note: str | None = None
    accuracy: AccuracySpec = field(default_factory=lambda: AccuracySpec("n/a"))
    trials_cpu_perf: int | None = None
    compile_expected: str = "ok"
    compile_missing: tuple[str, ...] = ()
    compile_reason: str | None = None

    @property
    def known_compile_failure(self) -> bool:
        return self.compile_expected != "ok"

    @property
    def abs_path(self) -> Path:
        return repo_root() / self.path

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass
class Manifest:
    schema_version: int
    dataset: str
    description: str
    versioning: str
    submodules: dict[str, str]
    defaults: dict[str, Any]
    models: list[Model] = field(default_factory=list)
    path: Path | None = None

    def by_id(self, model_id: str) -> Model:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(model_id)

    def tagged(self, tag: str) -> list[Model]:
        return [model for model in self.models if model.has_tag(tag)]


def _require(obj: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in obj:
        raise ManifestError(f"{where}: missing {key!r}")
    return obj[key]


def _as_tags(raw: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{where}: tags must be a non-empty list")
    tags = tuple(str(item) for item in raw)
    unknown = [tag for tag in tags if tag not in ALLOWED_TAGS]
    if unknown:
        raise ManifestError(f"{where}: unknown tags {unknown}; allowed {sorted(ALLOWED_TAGS)}")
    if len(set(tags)) != len(tags):
        raise ManifestError(f"{where}: duplicate tags")
    return tags


def _as_reference(raw: Any, *, where: str, tags: tuple[str, ...]) -> Reference:
    if raw is None:
        if "accuracy" in tags:
            raise ManifestError(f"{where}: accuracy models require a reference")
        return Reference(probability=None, source=None, kind=None, uncertainty=0.0)
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: reference must be an object or null")
    prob = raw.get("probability")
    if prob is None:
        raise ManifestError(f"{where}: reference.probability is required when reference is present")
    try:
        probability = float(prob)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{where}: reference.probability must be a number") from exc
    if not math.isfinite(probability) or not (0.0 < probability < 1.0):
        raise ManifestError(f"{where}: reference.probability must be in (0, 1), got {probability}")
    uncertainty = float(raw.get("uncertainty") or 0.0)
    if uncertainty < 0.0 or not math.isfinite(uncertainty):
        raise ManifestError(f"{where}: reference.uncertainty must be ≥ 0")
    source = raw.get("source")
    if not source:
        raise ManifestError(f"{where}: reference.source is required")
    kind = str(raw.get("kind") or ("exact" if uncertainty == 0.0 else "approximate"))
    citation = raw.get("citation")
    return Reference(
        probability=probability,
        source=str(source),
        kind=kind,
        uncertainty=uncertainty,
        citation=None if citation is None else str(citation),
    )


def _as_trials(raw: Any, *, where: str, tags: tuple[str, ...]) -> tuple[int | None, int | None, int | None]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: trials must be an object")
    accuracy = raw.get("accuracy")
    perf = raw.get("perf")
    if "accuracy" in tags:
        if accuracy is None:
            raise ManifestError(f"{where}: trials.accuracy required for accuracy models")
        accuracy_n = int(accuracy)
        if accuracy_n <= 0:
            raise ManifestError(f"{where}: trials.accuracy must be positive")
    else:
        accuracy_n = None if accuracy is None else int(accuracy)
    if "perf" in tags:
        if perf is None:
            raise ManifestError(f"{where}: trials.perf required for perf models")
        perf_n = int(perf)
        if perf_n <= 0:
            raise ManifestError(f"{where}: trials.perf must be positive")
    else:
        perf_n = None if perf is None else int(perf)
    cpu_perf = raw.get("cpu_perf")
    cpu_perf_n = None if cpu_perf is None else int(cpu_perf)
    if cpu_perf_n is not None and cpu_perf_n <= 0:
        raise ManifestError(f"{where}: trials.cpu_perf must be positive")
    return accuracy_n, perf_n, cpu_perf_n


def _as_budget(raw: Any, *, where: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: time_budget_s must be an object")
    budget: dict[str, int] = {}
    for key, value in raw.items():
        n = int(value)
        if n <= 0:
            raise ManifestError(f"{where}: time_budget_s.{key} must be positive")
        budget[str(key)] = n
    return budget


def _parse_model(raw: Mapping[str, Any], *, index: int) -> Model:
    where = f"models[{index}]"
    model_id = str(_require(raw, "id", where=where))
    where = f"models[{index}] ({model_id})"
    path = str(_require(raw, "path", where=where))
    digest = str(_require(raw, "sha256", where=where)).lower()
    if not SHA256_RE.match(digest):
        raise ManifestError(f"{where}: sha256 must be 64 lowercase hex chars")
    source = str(_require(raw, "source", where=where))
    if source not in ALLOWED_SOURCES:
        raise ManifestError(f"{where}: source must be one of {sorted(ALLOWED_SOURCES)}")
    query = str(_require(raw, "query", where=where))
    if not query:
        raise ManifestError(f"{where}: query must be a non-empty string")
    tags = _as_tags(_require(raw, "tags", where=where), where=where)
    if source == "pwr" and "pwr" not in tags:
        raise ManifestError(f"{where}: PWR models must include the pwr tag")
    if source == "pwr" and "accuracy" in tags:
        raise ManifestError(f"{where}: v1 PWR models are not in the accuracy set")
    reference = _as_reference(raw.get("reference"), where=where, tags=tags)
    trials_accuracy, trials_perf, trials_cpu_perf = _as_trials(
        _require(raw, "trials", where=where), where=where, tags=tags
    )
    budget = _as_budget(_require(raw, "time_budget_s", where=where), where=where)
    submodule = raw.get("submodule_commit")
    if source == "pwr" and not submodule:
        raise ManifestError(f"{where}: PWR models require submodule_commit")
    if source == "aralia" and submodule not in (None, ""):
        raise ManifestError(f"{where}: Aralia models must not set submodule_commit")
    acc_raw = raw.get("accuracy")
    if isinstance(acc_raw, Mapping) and acc_raw.get("check"):
        accuracy = AccuracySpec(
            check=str(acc_raw["check"]),
            reason=None if acc_raw.get("reason") is None else str(acc_raw["reason"]),
        )
        if accuracy.check not in {"feasible", "low_power", "infeasible", "n/a"}:
            raise ManifestError(f"{where}: accuracy.check {accuracy.check!r} is not valid")
        if "accuracy" in tags and accuracy.check == "infeasible" and not (accuracy.reason or "").strip():
            raise ManifestError(f"{where}: infeasible accuracy requires a reason")
    elif "accuracy" in tags:
        accuracy = classify_accuracy(reference.probability, trials_accuracy)
    else:
        accuracy = AccuracySpec(check="n/a")
    compile_raw = raw.get("compile")
    compile_expected = "ok"
    compile_missing: tuple[str, ...] = ()
    compile_reason: str | None = None
    if isinstance(compile_raw, Mapping):
        compile_expected = str(compile_raw.get("expected") or "ok")
        if compile_expected not in {"ok", "undefined_element"}:
            raise ManifestError(f"{where}: compile.expected {compile_expected!r} is not valid")
        missing_raw = compile_raw.get("missing_elements") or []
        if not isinstance(missing_raw, list):
            raise ManifestError(f"{where}: compile.missing_elements must be an array")
        compile_missing = tuple(str(item) for item in missing_raw)
        if compile_raw.get("reason") is not None:
            compile_reason = str(compile_raw["reason"])
        if compile_expected != "ok" and not (compile_reason or "").strip():
            raise ManifestError(f"{where}: known compile failure requires a reason")
    elif compile_raw is not None:
        raise ManifestError(f"{where}: compile must be an object")
    return Model(
        id=model_id,
        path=path,
        sha256=digest,
        source=source,
        query=query,
        tags=tags,
        trials_accuracy=trials_accuracy,
        trials_perf=trials_perf,
        trials_cpu_perf=trials_cpu_perf,
        time_budget_s=budget,
        reference=reference,
        submodule_commit=None if not submodule else str(submodule),
        perf_reason=None if raw.get("perf_reason") is None else str(raw["perf_reason"]),
        notes=None if raw.get("notes") is None else str(raw["notes"]),
        query_note=None if raw.get("query_note") is None else str(raw["query_note"]),
        accuracy=accuracy,
        compile_expected=compile_expected,
        compile_missing=compile_missing,
        compile_reason=compile_reason,
    )


def load_manifest(
    path: str | Path | None = None,
    *,
    root: Path | None = None,
    check_files: bool = False,
) -> Manifest:
    """Parse a dataset manifest. Set ``check_files=True`` to verify paths and sha256."""
    repo = root or repo_root()
    manifest_file = Path(path) if path is not None else manifest_path(repo)
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {manifest_file}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {manifest_file}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest root must be an object")
    schema_version = int(_require(payload, "schema_version", where="manifest"))
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema_version {schema_version} (want {SCHEMA_VERSION})")
    dataset = str(_require(payload, "dataset", where="manifest"))
    models_raw = _require(payload, "models", where="manifest")
    if not isinstance(models_raw, list) or not models_raw:
        raise ManifestError("models must be a non-empty array")
    models = [_parse_model(item, index=i) for i, item in enumerate(models_raw)]
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate model id")
    submodules = payload.get("submodules") or {}
    if not isinstance(submodules, Mapping):
        raise ManifestError("submodules must be an object")
    manifest = Manifest(
        schema_version=schema_version,
        dataset=dataset,
        description=str(payload.get("description") or ""),
        versioning=str(payload.get("versioning") or ""),
        submodules={str(k): str(v) for k, v in submodules.items()},
        defaults=dict(payload.get("defaults") or {}),
        models=models,
        path=manifest_file,
    )
    if check_files:
        validate_manifest(manifest, root=repo)
    return manifest


def validate_manifest(manifest: Manifest, *, root: Path | None = None) -> None:
    """Check files exist, sha256 matches, and references are in (0, 1)."""
    repo = root or repo_root()
    errors: list[str] = []
    for model in manifest.models:
        path = repo / model.path
        if not path.is_file():
            errors.append(f"{model.id}: missing file {model.path}")
            continue
        digest = sha256_file(path)
        if digest != model.sha256:
            errors.append(
                f"{model.id}: sha256 mismatch for {model.path}: "
                f"manifest={model.sha256} disk={digest}"
            )
        if model.reference.probability is not None:
            p = model.reference.probability
            if not (0.0 < p < 1.0):
                errors.append(f"{model.id}: reference.probability {p} not in (0, 1)")
        if model.has_tag("perf") and not (model.perf_reason or "").strip():
            errors.append(f"{model.id}: perf models require perf_reason")
        if model.has_tag("accuracy") and model.reference.probability is None:
            errors.append(f"{model.id}: accuracy models require a reference probability")
        if model.has_tag("accuracy") and model.accuracy.check == "infeasible" and not (
            model.accuracy.reason or ""
        ).strip():
            errors.append(f"{model.id}: infeasible accuracy requires a reason")
        if model.source == "pwr":
            expected = manifest.submodules.get(PWR_SUBMODULE)
            if expected and model.submodule_commit != expected:
                errors.append(
                    f"{model.id}: submodule_commit {model.submodule_commit} "
                    f"!= submodules[{PWR_SUBMODULE!r}] {expected}"
                )
    if errors:
        raise ManifestError("manifest validation failed:\n  " + "\n  ".join(errors))
