"""``python -m benchmarks run|ab|summarize|validate``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..datasets.compat import patch_generic_pwr
from ..loader import (
    FZB_1E8,
    Manifest,
    Model,
    load_manifest,
    repo_root,
    validate_manifest,
)
from . import dockerutil
from .execute import compile_and_inspect, container_env, fzb_run, probe_pass_size
from .guards import (
    GuardError,
    acquire_lock,
    release_lock,
    snapshot_load,
    wait_for_idle,
)
from .stats import (
    Z_BLOCK,
    bootstrap_median_ratio_ci,
    iqr,
    paired_ratios,
    z_score,
)
from .summarize import summarize, summarize_path
from .inject import choose_pass_counts, parse_inject_regression, pass_size_from_run
from .verdict import (
    VERDICT_BLOCK,
    VERDICT_ERROR,
    VERDICT_INFEASIBLE,
    VERDICT_KNOWN_FAILURE,
    VERDICT_PASS,
    VERDICT_PASS_LOW_POWER,
    VERDICT_TIMEOUT,
    VERDICT_WARN,
    ab_timeout_verdict,
    accuracy_power_verdict,
    collect_timings,
    combine_verdicts,
    cpu_wall_verdict,
    gpu_kernel_verdict,
    gpu_total_verdict,
)

# Device indices are nvidia-smi / NVML / docker `--gpus device=N` ordinals,
# matching images.yml and gpu.yml. Do not derive them from CUDA's default
# FASTEST_FIRST ordering, which is the reverse on xerxes:
#   GPU 0 = Tesla V100S (PCI 42:00.0), lock /tmp/gpu-lock-xerxes-0
#   GPU 1 = RTX 3070   (PCI 43:00.0), lock /tmp/gpu-lock-xerxes-1
TARGETS: dict[str, dict[str, Any]] = {
    "cpu": {
        "docker_context": "gateway",
        "gpus": None,
        "backend": "omp",
        "profiler": None,
        "lock_host": None,
        "lock_path": None,
        "device": "cpu",
        "runner": "fzb-bench-gtw-1",
        "models_dir": "/tmp/fuzzybits-bench/input",
        "work_dir": "/tmp/fuzzybits-bench/work",
        "jit_volume": "fuzzybits-bench-jit-cpu",
        "cpuset_cpus": "0-7",
        "omp_num_threads": 8,
    },
    "v100s": {
        "docker_context": "xerxes",
        "gpus": "device=0",
        "backend": "cuda",
        "profiler": "nvprof",
        "profiler_bin": "/opt/cuda/cuda/bin/nvprof",
        "profiler_volume": "pr91-nvprof:/opt/cuda:ro",
        "lock_host": "xerxes",
        "lock_path": "/tmp/gpu-lock-xerxes-0",
        "device": "Tesla V100S-PCIE-32GB",
        "runner": "fzb-xerxes-5",
        "models_dir": "/tmp/fuzzybits-bench/input",
        "work_dir": "/tmp/fuzzybits-bench/work",
        "jit_volume": "fuzzybits-bench-jit-v100s",
    },
    "rtx3070": {
        "docker_context": "xerxes",
        "gpus": "device=1",
        "backend": "cuda",
        "profiler": "nsys",
        "profiler_bin": "/opt/pr91/nvidia/nsight-systems/2025.3.2/bin/nsys",
        "profiler_volume": "pr91-data:/opt/pr91:ro",
        "lock_host": "xerxes",
        "lock_path": "/tmp/gpu-lock-xerxes-1",
        "device": "NVIDIA GeForce RTX 3070",
        "runner": "fzb-xerxes-3",
        "models_dir": "/tmp/fuzzybits-bench/input",
        "work_dir": "/tmp/fuzzybits-bench/work",
        "jit_volume": "fuzzybits-bench-jit-rtx3070",
    },
}


def _run_error(prefix: str, run: dict[str, Any]) -> str:
    parts = [prefix]
    if run.get("error"):
        parts.append(str(run["error"]))
    tail = (run.get("stderr") or run.get("docker_stderr") or run.get("stdout") or "")[-800:]
    if tail.strip():
        parts.append(tail.strip())
    return ": ".join(parts) if len(parts) > 1 else prefix


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pin_image(
    image: str,
    *,
    allow_local: bool = False,
    context: str | None = None,
) -> dict[str, str]:
    """Resolve a tag to ``name@sha256:...`` from the registry (once)."""
    return dockerutil.pin_image(image, allow_local=allow_local, context=context)


def _git_sha(root: Path, rev: str = "HEAD") -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", rev],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _ensure_volume(context: str, name: str) -> None:
    if ":" in name:
        name = name.split(":", 1)[0]
    inspect = subprocess.run(
        dockerutil.docker_cmd(context, "volume", "inspect", name),
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        create = subprocess.run(
            dockerutil.docker_cmd(context, "volume", "create", name),
            check=False,
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            raise dockerutil.DockerError(create.stderr.strip())


def _runtime_host(target: dict[str, Any]) -> str | None:
    """SSH host for locks/staging, or None when docker is local (CI on the runner)."""
    context = target.get("docker_context") or "default"
    if _is_local_context(context):
        return None
    return target.get("lock_host") or context


def _ensure_host_dir(host: str | None, path: str) -> None:
    if not host:
        Path(path).mkdir(parents=True, exist_ok=True)
        return
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {path}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _is_local_context(context: str | None) -> bool:
    return context in {None, "", "default"}


def _volumes(target: dict[str, Any], jit_name: str) -> list[str]:
    volumes = [
        f"{target['models_dir']}:/models:ro",
        f"{target['work_dir']}:/work",
        f"{jit_name}:/home/coder/.cache/fuzzybits/acpp",
    ]
    extra = target.get("profiler_volume")
    if extra:
        volumes.append(extra)
    return volumes


def _select_models(manifest: Manifest, args: argparse.Namespace) -> list[Model]:
    models = list(manifest.models)
    if args.tag:
        wanted = set(args.tag)
        models = [m for m in models if wanted.issubset(m.tags)]
    if args.ids:
        wanted_ids = set(args.ids)
        models = [m for m in models if m.id in wanted_ids]
        missing = wanted_ids - {m.id for m in models}
        if missing:
            raise SystemExit(f"unknown model id(s): {sorted(missing)}")
    if args.mode == "accuracy":
        models = [m for m in models if m.has_tag("accuracy")]
    elif args.mode == "perf":
        models = [m for m in models if m.has_tag("perf")]
    elif args.mode == "compile":
        models = [m for m in models if m.has_tag("pwr") or m.source == "pwr"]
    return models


def _model_xml(model: Model) -> str:
    # Mount is repo-style input/ at /models, so strip the leading "input/".
    rel = model.path
    prefix = "input/"
    if rel.startswith(prefix):
        rel = rel[len(prefix) :]
    return "/models/" + rel


def _compat_xml(
    model: Model,
    *,
    root: Path,
    host: str | None,
    work_dir: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return the container XML path. PWR models use a cached MEF rewrite."""
    if model.source != "pwr":
        return _model_xml(model), None
    result = patch_generic_pwr(
        root / model.path,
        cache_dir=Path("/tmp/fuzzybits-pwr-compat"),
    )
    meta: dict[str, Any] = {
        "shim": result.shim,
        "version": result.version,
        "injected_gates": list(result.injected_gates),
        "injected_events": list(result.injected_events),
        "rewritten_functional_events": [list(pair) for pair in result.rewritten_functional_events],
        "unchanged": result.unchanged,
        "sources": result.sources,
    }
    if result.unchanged:
        return _model_xml(model), meta
    dest_name = model.id.replace("/", "_") + ".xml"
    dockerutil.stage_file(
        result.path,
        host=host,
        dest_dir=f"{work_dir.rstrip('/')}/compat",
        dest_name=dest_name,
    )
    return f"/work/compat/{dest_name}", meta


def _pcfb_path(model: Model) -> str:
    safe = model.id.replace("/", "_")
    return f"/work/{safe}.pcfb"


def _power_thresholds(manifest: Manifest | None) -> tuple[float, float]:
    defaults = (manifest.defaults if manifest is not None else {}) or {}
    return (
        float(defaults.get("accuracy_infeasible_hits", 16)),
        float(defaults.get("accuracy_low_power_hits", 1000)),
    )


def _accuracy_row(
    model: Model,
    run: dict[str, Any],
    *,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    infeasible_hits, low_power_hits = _power_thresholds(manifest)
    timed_out = bool(run.get("timed_out"))
    queries = (run.get("run") or {}).get("queries") or []
    p_ref = float(model.reference.probability or 0.0)
    base = {
        "id": model.id,
        "mode": "accuracy",
        "p_ref": p_ref,
        "p_ref_source": model.reference.source,
        "accuracy_check": model.accuracy.check,
        "wall_s": run.get("wall_s"),
        "query": model.query,
        "gated": model.accuracy.gated,
    }
    if timed_out and not queries:
        return {
            **base,
            "ok": False,
            "status": "timeout",
            "verdict": VERDICT_TIMEOUT,
            "estimate": None,
            "n": None,
            "hits": None,
            "expected_hits": None,
            "z": None,
            "reason": f"wall-clock timeout after {run.get('wall_s')}s",
        }
    if not queries:
        return {
            **base,
            "ok": False,
            "status": "error",
            "error": "no queries",
            "verdict": VERDICT_BLOCK,
            "z": None,
        }
    row = queries[0]
    estimate = float(row["estimate"])
    n = int(row["n"])
    hits = int(row["hits"])
    z = z_score(estimate, n, p_ref, model.reference.uncertainty)
    judged = accuracy_power_verdict(
        p_ref=p_ref,
        n=n,
        hits=hits,
        z=z,
        infeasible_hits=infeasible_hits,
        low_power_hits=low_power_hits,
    )
    verdict = judged["verdict"]
    if timed_out:
        verdict = VERDICT_TIMEOUT
    return {
        **base,
        "ok": not timed_out,
        "status": "timeout" if timed_out else judged["verdict"],
        "estimate": estimate,
        "ci_low": float(row.get("ci_low") or 0.0),
        "ci_high": float(row.get("ci_high") or 0.0),
        "n": n,
        "hits": hits,
        "expected_hits": judged["expected_hits"],
        "z": judged["z"],
        "verdict": verdict,
        "gated": judged["gated"] and not timed_out,
        "reason": judged.get("reason"),
        "query": row.get("name") or model.query,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = load_manifest(args.manifest, root=root, check_files=False)
    validate_manifest(manifest, root=root)
    print(
        f"ok  dataset={manifest.dataset}  models={len(manifest.models)}  "
        f"accuracy={len(manifest.tagged('accuracy'))}  "
        f"perf={len(manifest.tagged('perf'))}  pwr={len(manifest.tagged('pwr'))}"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = load_manifest(args.manifest, root=root, check_files=not args.skip_validate)
    if not args.skip_validate:
        validate_manifest(manifest, root=root)
    target = dict(TARGETS[args.target])
    if args.docker_context:
        target["docker_context"] = args.docker_context
    if getattr(args, "models_dir", None):
        target["models_dir"] = args.models_dir
    if getattr(args, "work_dir", None):
        target["work_dir"] = args.work_dir
    if args.image is None:
        raise SystemExit("run: --image is required (build with `ab --build` or pass nightly-generic)")
    try:
        pinned = _pin_image(args.image, allow_local=True, context=target["docker_context"])
    except dockerutil.DockerError as exc:
        raise SystemExit(f"run: {exc}") from exc
    image = pinned["ref"]
    context = target["docker_context"]
    backend = args.backend or target["backend"]
    gpus = target["gpus"]
    jit = args.jit_volume or target["jit_volume"]
    _ensure_volume(context, jit)
    host = _runtime_host(target)
    _ensure_host_dir(host, target["work_dir"])
    volumes = _volumes(target, jit)
    env = container_env(
        backend=backend,
        extra={"OMP_NUM_THREADS": str(target["omp_num_threads"])} if target.get("omp_num_threads") else None,
    )
    cpuset = target.get("cpuset_cpus")
    lock_held = False
    try:
        try:
            if target.get("lock_path") and not args.no_lock:
                acquire_lock(
                    host,
                    target["lock_path"],
                    retries=args.guard_retries,
                )
                lock_held = True
            if args.mode == "accuracy":
                # Correctness, not a timing gate. CI runners are often busy.
                load_before = snapshot_load(context)
            else:
                load_before = wait_for_idle(
                    context=context,
                    gpus=gpus,
                    retries=args.guard_retries,
                )
        except GuardError as exc:
            payload = {
                "schema_version": 1,
                "kind": "run",
                "dataset": manifest.dataset,
                "created_utc": _now(),
                "verdict": "inconclusive",
                "context": {
                    "error": str(exc),
                    "target": args.target,
                    "image": image,
                    "docker_context": context,
                    "backend": backend,
                    "device": target["device"],
                    "runner": target["runner"],
                },
                "models": [],
            }
            _write_json(Path(args.out), payload)
            print(summarize(payload))
            return 2

        models = _select_models(manifest, args)
        rows: list[dict[str, Any]] = []
        for model in models:
            xml, compat = _compat_xml(
                model, root=root, host=host, work_dir=target["work_dir"]
            )
            pcfb = _pcfb_path(model)
            compiled = compile_and_inspect(
                context=context,
                image=image,
                model_xml=xml,
                pcfb_out=pcfb,
                volumes=volumes,
                env=env,
                timeout_s=int(model.time_budget_s.get("cpu_compile") or model.time_budget_s.get("cpu_accuracy", 60)),
                gpus=gpus if backend == "cuda" else None,
                cpuset_cpus=cpuset,
            )
            if args.mode == "compile" or model.has_tag("pwr"):
                inspect = compiled.get("inspect") or {}
                timed_out = bool(compiled.get("timed_out"))
                ok = bool(compiled.get("ok")) and not timed_out
                if timed_out:
                    compile_verdict = VERDICT_TIMEOUT
                    status = "timeout"
                elif ok and model.known_compile_failure:
                    compile_verdict = VERDICT_WARN
                    status = "ok"
                elif ok:
                    compile_verdict = VERDICT_PASS
                    status = "ok"
                elif model.known_compile_failure:
                    compile_verdict = VERDICT_KNOWN_FAILURE
                    status = "known_failure"
                else:
                    compile_verdict = VERDICT_BLOCK
                    status = "error"
                row = {
                    "id": model.id,
                    "mode": "compile",
                    "ok": ok,
                    "status": status,
                    "compile_wall_s": compiled.get("compile_wall_s") or compiled.get("wall_s"),
                    "inspect": {
                        "tensors": inspect.get("tensors"),
                        "operators": inspect.get("operators"),
                        "samples": inspect.get("samples"),
                        "queries": inspect.get("queries"),
                        "query_names": inspect.get("query_names"),
                    },
                    "error": None if ok else (
                        compiled.get("error")
                        or compiled.get("stderr")
                        or compiled.get("docker_stderr")
                    ),
                    "verdict": compile_verdict,
                    "timed_out": timed_out,
                    "known_compile_failure": model.known_compile_failure,
                    "compile_missing": list(model.compile_missing),
                    "compile_reason": model.compile_reason,
                    "compat": compat,
                }
                rows.append(row)
                print(
                    f"{model.id}: compile {row['status']} wall={row.get('compile_wall_s')}",
                    file=sys.stderr,
                )
                continue
            if compiled.get("timed_out"):
                rows.append(
                    {
                        "id": model.id,
                        "mode": args.mode,
                        "ok": False,
                        "status": "timeout",
                        "timed_out": True,
                        "wall_s": compiled.get("wall_s"),
                        "error": compiled.get("error") or compiled.get("stderr"),
                        "verdict": VERDICT_TIMEOUT,
                    }
                )
                continue
            if not compiled.get("ok"):
                rows.append(
                    {
                        "id": model.id,
                        "mode": args.mode,
                        "ok": False,
                        "error": compiled.get("error") or compiled.get("stderr"),
                        "verdict": "block",
                    }
                )
                continue
            trials = (
                model.trials_accuracy
                if args.mode == "accuracy"
                else (
                    (model.trials_cpu_perf or model.trials_perf or FZB_1E8)
                    if backend == "omp"
                    else (model.trials_perf or FZB_1E8)
                )
            )
            budget_key = "cpu_accuracy" if args.mode == "accuracy" else (
                "gpu_perf" if backend == "cuda" else "cpu_perf"
            )
            run = fzb_run(
                context=context,
                image=image,
                pcfb=pcfb,
                trials=int(trials or FZB_1E8),
                backend=backend,
                volumes=volumes,
                env=env,
                timeout_s=int(model.time_budget_s.get(budget_key, 60)),
                gpus=gpus if backend == "cuda" else None,
                profiler=target.get("profiler") if args.mode == "perf" else None,
                profiler_bin=target.get("profiler_bin"),
                cpuset_cpus=cpuset,
                inprocess=backend == "omp",
            )
            if args.mode == "accuracy":
                row = _accuracy_row(model, run, manifest=manifest)
                row["inspect"] = compiled.get("inspect")
                row["compile_wall_s"] = compiled.get("compile_wall_s")
            else:
                profiler = run.get("profiler") or {}
                timed_out = bool(run.get("timed_out"))
                row = {
                    "id": model.id,
                    "mode": "perf",
                    "ok": run.get("ok") and not timed_out,
                    "status": "timeout" if timed_out else ("ok" if run.get("ok") else "error"),
                    "wall_s": run.get("wall_s"),
                    "sample_s": run.get("sample_s"),
                    "profiler": profiler,
                    "run": run.get("run"),
                    "verdict": VERDICT_TIMEOUT if timed_out else (
                        "pass" if run.get("ok") else "block"
                    ),
                    "timed_out": timed_out,
                    "error": None if run.get("ok") and not timed_out else (
                        run.get("error") or run.get("stderr")
                    ),
                    "compile_wall_s": compiled.get("compile_wall_s"),
                    "inspect": compiled.get("inspect"),
                }
            rows.append(row)
            print(
                f"{model.id}: {row.get('verdict')} z={row.get('z')} wall={row.get('wall_s')}",
                file=sys.stderr,
            )

        load_after = snapshot_load(context)
        payload = {
            "schema_version": 1,
            "kind": "run",
            "dataset": manifest.dataset,
            "created_utc": _now(),
            "verdict": combine_verdicts(r.get("verdict") or "pass" for r in rows),
            "context": {
                "target": args.target,
                "image": image,
                "image_tag": pinned["tag"],
                "image_digest": pinned["digest"],
                "image_pin_source": pinned["source"],
                "docker_context": context,
                "backend": backend,
                "device": target["device"],
                "runner": target["runner"],
                "commit": args.commit or _git_sha(root),
                "recipe": "docker/runtime.Dockerfile --target runtime",
                "load_before": load_before,
                "load_after": dict(load_after),
            },
            "models": rows,
        }
        _write_json(Path(args.out), payload)
        text = summarize(payload)
        if args.md:
            Path(args.md).write_text(text, encoding="utf-8")
        print(text)
        return 0 if payload["verdict"] in {
            VERDICT_PASS,
            VERDICT_PASS_LOW_POWER,
            VERDICT_INFEASIBLE,
            VERDICT_KNOWN_FAILURE,
            VERDICT_WARN,
            VERDICT_TIMEOUT,
        } else 1
    finally:
        if lock_held:
            release_lock(host, target["lock_path"])


def _ratio_row(
    *,
    samples_base: list[float],
    samples_head: list[float],
    metric: str,
    gpu: bool,
) -> dict[str, Any] | tuple[None, str]:
    try:
        ratio, lo, hi = bootstrap_median_ratio_ci(samples_base, samples_head, seed=1)
    except ValueError as exc:
        return None, str(exc)
    if gpu:
        if metric == "gpu_kernel_ms":
            verdict = gpu_kernel_verdict(ratio, lo, hi)
        else:
            verdict = gpu_total_verdict(ratio, lo, hi)
    else:
        verdict = cpu_wall_verdict(ratio, lo, hi)
    ratios = paired_ratios(samples_base, samples_head)
    return {
        "metric": metric,
        "ratio": ratio,
        "ratio_ci": [lo, hi],
        "iqr_ratio": iqr(ratios),
        "base_samples": samples_base,
        "head_samples": samples_head,
        "verdict": verdict,
    }


def cmd_ab(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = load_manifest(args.manifest, root=root, check_files=not args.skip_validate)
    target = dict(TARGETS[args.target])
    if args.docker_context:
        target["docker_context"] = args.docker_context
    if getattr(args, "models_dir", None):
        target["models_dir"] = args.models_dir
    if getattr(args, "work_dir", None):
        target["work_dir"] = args.work_dir
    context = target["docker_context"]
    backend = args.backend or target["backend"]
    gpus = target["gpus"]
    gpu = backend == "cuda"
    base_sha = args.base or _git_sha(root, "origin/develop")
    head_sha = args.head or _git_sha(root, "HEAD")
    if args.build:
        print(f"building base {base_sha[:12]} on {context}", file=sys.stderr)
        base_info = dockerutil.build_runtime_image(sha=base_sha, context=context, root=root)
        print(f"building head {head_sha[:12]} on {context}", file=sys.stderr)
        head_info = dockerutil.build_runtime_image(sha=head_sha, context=context, root=root)
        base_image = base_info["tag"]
        head_image = head_info["tag"]
        base_digest = base_info["digest"]
        head_digest = head_info["digest"]
        base_pin_source = "build"
        head_pin_source = "build"
    else:
        if not args.base_image or not args.head_image:
            raise SystemExit("ab: pass --base-image and --head-image, or --build")
        try:
            if args.base_image == args.head_image:
                pinned = _pin_image(args.base_image)
                base_image = head_image = pinned["ref"]
                base_digest = head_digest = pinned["digest"]
                base_pin_source = head_pin_source = pinned["source"]
            else:
                base_pin = _pin_image(args.base_image)
                head_pin = _pin_image(args.head_image)
                base_image = base_pin["ref"]
                head_image = head_pin["ref"]
                base_digest = base_pin["digest"]
                head_digest = head_pin["digest"]
                base_pin_source = base_pin["source"]
                head_pin_source = head_pin["source"]
        except dockerutil.DockerError as exc:
            raise SystemExit(f"ab: {exc}") from exc
        print(
            f"pinned base {base_image} digest {base_digest} ({base_pin_source})",
            file=sys.stderr,
        )
        print(
            f"pinned head {head_image} digest {head_digest} ({head_pin_source})",
            file=sys.stderr,
        )

    try:
        inject = parse_inject_regression(args.inject_regression)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    extra_runs = inject.extra_runs
    sleep_ms = inject.sleep_ms
    scale = inject.scale

    jit_base = (args.jit_volume or target["jit_volume"]) + "-base"
    jit_head = (args.jit_volume or target["jit_volume"]) + "-head"
    _ensure_volume(context, jit_base)
    _ensure_volume(context, jit_head)
    host = _runtime_host(target)
    _ensure_host_dir(host, target["work_dir"])

    lock_held = False
    try:
        try:
            if target.get("lock_path") and not args.no_lock:
                acquire_lock(
                    host,
                    target["lock_path"],
                    retries=args.guard_retries,
                )
                lock_held = True
            load_before = wait_for_idle(
                context=context,
                gpus=gpus,
                retries=args.guard_retries,
            )
        except GuardError as exc:
            payload = {
                "schema_version": 1,
                "kind": "ab",
                "dataset": manifest.dataset,
                "created_utc": _now(),
                "verdict": "inconclusive",
                "context": {"error": str(exc), "target": args.target},
                "models": [],
            }
            _write_json(Path(args.out), payload)
            print(summarize(payload))
            return 2

        models = _select_models(manifest, args)
        if args.mode == "run":
            # Default A/B set: perf models (and their accuracy check on the same run).
            if not args.ids and not args.tag:
                models = [m for m in manifest.models if m.has_tag("perf")]
        rows: list[dict[str, Any]] = []
        rounds = args.rounds
        for model in models:
            xml, _compat = _compat_xml(
                model, root=root, host=host, work_dir=target["work_dir"]
            )
            pcfb = _pcfb_path(model)
            omp_extra = (
                {"OMP_NUM_THREADS": str(target["omp_num_threads"])}
                if target.get("omp_num_threads")
                else None
            )
            env_base = container_env(backend=backend, extra=omp_extra)
            env_head = container_env(backend=backend, extra=omp_extra)
            cpuset = target.get("cpuset_cpus")
            vol_base = _volumes(target, jit_base)
            vol_head = _volumes(target, jit_head)
            compile_timeout = int(
                model.time_budget_s.get("cpu_compile")
                or model.time_budget_s.get("cpu_accuracy", 60)
            )
            compiled_base = compile_and_inspect(
                context=context,
                image=base_image,
                model_xml=xml,
                pcfb_out=pcfb,
                volumes=vol_base,
                env=env_base,
                timeout_s=compile_timeout,
                gpus=gpus if gpu else None,
                cpuset_cpus=cpuset,
            )
            compiled_head = compile_and_inspect(
                context=context,
                image=head_image,
                model_xml=xml,
                pcfb_out=pcfb,
                volumes=vol_head,
                env=env_head,
                timeout_s=compile_timeout,
                gpus=gpus if gpu else None,
                cpuset_cpus=cpuset,
            )
            if compiled_base.get("timed_out") or compiled_head.get("timed_out"):
                rows.append(
                    {
                        "id": model.id,
                        "status": "timeout",
                        "verdict": ab_timeout_verdict(
                            base_timeout=bool(compiled_base.get("timed_out")),
                            head_timeout=bool(compiled_head.get("timed_out")),
                        ),
                        "error": {
                            "base": compiled_base.get("error") or compiled_base.get("stderr"),
                            "head": compiled_head.get("error") or compiled_head.get("stderr"),
                        },
                    }
                )
                continue
            if not compiled_base.get("ok") or not compiled_head.get("ok"):
                rows.append(
                    {
                        "id": model.id,
                        "verdict": "block",
                        "error": {
                            "base": compiled_base.get("error") or compiled_base.get("stderr"),
                            "head": compiled_head.get("error") or compiled_head.get("stderr"),
                        },
                    }
                )
                continue
            if gpu:
                trials = int(model.trials_perf or model.trials_accuracy or FZB_1E8)
            else:
                trials = int(
                    model.trials_cpu_perf or model.trials_perf or model.trials_accuracy or FZB_1E8
                )
            head_trials = trials
            scale_meta: dict[str, Any] | None = None
            budget_key = "gpu_perf" if gpu else "cpu_perf"
            timeout_s = int(model.time_budget_s.get(budget_key, 60))
            profiler = target.get("profiler") if gpu else None
            if scale != 1.0:
                probe = probe_pass_size(
                    context=context,
                    image=head_image,
                    pcfb=pcfb,
                    trials=int(trials),
                    backend=backend,
                    volumes=vol_head,
                    env=env_head,
                    timeout_s=timeout_s,
                    gpus=gpus if gpu else None,
                    cpuset_cpus=cpuset,
                )
                if probe.get("timed_out") or not probe.get("ok"):
                    rows.append(
                        {
                            "id": model.id,
                            "verdict": VERDICT_ERROR,
                            "error": _run_error("pass-size probe", probe),
                        }
                    )
                    continue
                try:
                    pass_size = pass_size_from_run(probe)
                    min_passes = max(1, (int(trials) + pass_size - 1) // pass_size)
                    n_base, n_head = choose_pass_counts(scale, min_passes=min_passes)
                except ValueError as exc:
                    rows.append(
                        {
                            "id": model.id,
                            "verdict": VERDICT_ERROR,
                            "error": str(exc),
                        }
                    )
                    continue
                trials = n_base * pass_size
                head_trials = n_head * pass_size
                timeout_s = int(timeout_s * max(1.0, n_head / min_passes) + 1)
                scale_meta = {
                    "factor": scale,
                    "pass_size": pass_size,
                    "base_passes": n_base,
                    "head_passes": n_head,
                    "realized_ratio": n_head / n_base,
                    "base_trials": trials,
                    "head_trials": head_trials,
                }
                print(
                    f"{model.id}: scale:{scale:g} pass_size={pass_size} "
                    f"base_passes={n_base} head_passes={n_head} "
                    f"realized={n_head / n_base:.4f}",
                    file=sys.stderr,
                )
            # Warm caches (round 0 will still be dropped).
            warmup_timeout = False
            for side, image, volumes, env, extra, sleep, n_trials in (
                ("base", base_image, vol_base, env_base, 0, 0, trials),
                ("head", head_image, vol_head, env_head, extra_runs, sleep_ms, head_trials),
            ):
                warm = fzb_run(
                    context=context,
                    image=image,
                    pcfb=pcfb,
                    trials=n_trials,
                    backend=backend,
                    volumes=volumes,
                    env=env,
                    timeout_s=timeout_s,
                    gpus=gpus if gpu else None,
                    profiler=profiler,
                    profiler_bin=target.get("profiler_bin"),
                    extra_runs=extra,
                    sleep_ms=sleep,
                    cpuset_cpus=cpuset,
                    inprocess=not gpu,
                )
                if warm.get("timed_out"):
                    rows.append(
                        {
                            "id": model.id,
                            "status": "timeout",
                            "verdict": ab_timeout_verdict(
                                base_timeout=side == "base",
                                head_timeout=side == "head",
                            ),
                            "error": f"{side} warmup: wall-clock timeout after {warm.get('wall_s')}s",
                        }
                    )
                    warmup_timeout = True
                    break
                if not warm.get("ok"):
                    rows.append(
                        {
                            "id": model.id,
                            "verdict": "block",
                            "error": _run_error(f"{side} warmup", warm),
                        }
                    )
                    warmup_timeout = True
                    break
            if warmup_timeout:
                continue
            base_runs: list[dict[str, Any]] = []
            head_runs: list[dict[str, Any]] = []
            for round_i in range(rounds):
                for side, image, volumes, env, bucket, extra, sleep, n_trials in (
                    ("base", base_image, vol_base, env_base, base_runs, 0, 0, trials),
                    ("head", head_image, vol_head, env_head, head_runs, extra_runs, sleep_ms, head_trials),
                ):
                    timed = fzb_run(
                        context=context,
                        image=image,
                        pcfb=pcfb,
                        trials=n_trials,
                        backend=backend,
                        volumes=volumes,
                        env=env,
                        timeout_s=timeout_s,
                        gpus=gpus if gpu else None,
                        profiler=profiler,
                        profiler_bin=target.get("profiler_bin"),
                        extra_runs=extra,
                        sleep_ms=sleep,
                        cpuset_cpus=cpuset,
                        inprocess=not gpu,
                    )
                    if timed.get("timed_out"):
                        rows.append(
                            {
                                "id": model.id,
                                "status": "timeout",
                                "verdict": ab_timeout_verdict(
                                    base_timeout=side == "base",
                                    head_timeout=side == "head",
                                ),
                                "error": f"{side} round {round_i}: wall-clock timeout after {timed.get('wall_s')}s",
                            }
                        )
                        break
                    if not timed.get("ok"):
                        rows.append(
                            {
                                "id": model.id,
                                "verdict": "block",
                                "error": _run_error(f"{side} round {round_i}", timed),
                            }
                        )
                        break
                    bucket.append(timed)
                else:
                    continue
                break
            else:
                # Drop first round as warm-up.
                kept_base = base_runs[1:]
                kept_head = head_runs[1:]
                if gpu:
                    total_b, total_err = collect_timings(kept_base, gpu=True, kind="total")
                    total_h, total_err_h = collect_timings(kept_head, gpu=True, kind="total")
                    err = total_err or total_err_h
                    if err:
                        rows.append(
                            {
                                "id": model.id,
                                "metric": "gpu_total_ms",
                                "verdict": VERDICT_ERROR,
                                "error": err,
                            }
                        )
                        continue
                    assert total_b is not None and total_h is not None
                    total_row = _ratio_row(
                        samples_base=total_b,
                        samples_head=total_h,
                        metric="gpu_total_ms",
                        gpu=True,
                    )
                    if isinstance(total_row, tuple):
                        rows.append(
                            {
                                "id": model.id,
                                "metric": "gpu_total_ms",
                                "verdict": VERDICT_ERROR,
                                "error": total_row[1],
                                "base_samples": total_b,
                                "head_samples": total_h,
                            }
                        )
                        continue
                    kernel_b, kernel_err = collect_timings(
                        kept_base, gpu=True, kind="kernel"
                    )
                    kernel_h, kernel_err_h = collect_timings(
                        kept_head, gpu=True, kind="kernel"
                    )
                    kernel_row = None
                    if kernel_err is None and kernel_err_h is None:
                        assert kernel_b is not None and kernel_h is not None
                        kernel_row = _ratio_row(
                            samples_base=kernel_b,
                            samples_head=kernel_h,
                            metric="gpu_kernel_ms",
                            gpu=True,
                        )
                        if isinstance(kernel_row, tuple):
                            kernel_row = None
                    # Dual-gate: total is required; kernel is additional when present.
                    parts = [total_row["verdict"]]
                    if isinstance(kernel_row, dict):
                        parts.append(kernel_row["verdict"])
                    verdict = combine_verdicts(parts)
                    metric = "gpu_total_ms"
                    ratio = total_row["ratio"]
                    lo, hi = total_row["ratio_ci"]
                    extras = {
                        "gpu_total": total_row,
                        "gpu_kernel": kernel_row if isinstance(kernel_row, dict) else None,
                    }
                    base_ss = total_b
                    head_ss = total_h
                else:
                    base_ss, err_b = collect_timings(kept_base, gpu=False)
                    head_ss, err_h = collect_timings(kept_head, gpu=False)
                    err = err_b or err_h
                    if err:
                        rows.append(
                            {
                                "id": model.id,
                                "metric": "cpu_sample_s",
                                "verdict": VERDICT_ERROR,
                                "error": err,
                            }
                        )
                        continue
                    assert base_ss is not None and head_ss is not None
                    cpu_row = _ratio_row(
                        samples_base=base_ss,
                        samples_head=head_ss,
                        metric="cpu_sample_s",
                        gpu=False,
                    )
                    if isinstance(cpu_row, tuple):
                        rows.append(
                            {
                                "id": model.id,
                                "metric": "cpu_sample_s",
                                "verdict": VERDICT_ERROR,
                                "error": cpu_row[1],
                                "base_samples": base_ss,
                                "head_samples": head_ss,
                            }
                        )
                        continue
                    verdict = cpu_row["verdict"]
                    metric = "cpu_sample_s"
                    ratio = cpu_row["ratio"]
                    lo, hi = cpu_row["ratio_ci"]
                    extras = {}
                acc = None
                if model.has_tag("accuracy") and model.reference.probability is not None:
                    acc = _accuracy_row(model, head_runs[-1], manifest=manifest)
                    if acc.get("verdict") == VERDICT_BLOCK and acc.get("gated"):
                        verdict = VERDICT_BLOCK
                rows.append(
                    {
                        "id": model.id,
                        "metric": metric,
                        "ratio": ratio,
                        "ratio_ci": [lo, hi],
                        "iqr_ratio": iqr(paired_ratios(base_ss, head_ss)),
                        "base_samples": base_ss,
                        "head_samples": head_ss,
                        "verdict": verdict,
                        "accuracy": acc,
                        "n_rounds": len(base_ss),
                        "head_trials": head_trials,
                        "base_trials": trials,
                        "scale": scale_meta,
                        **extras,
                    }
                )
                print(
                    f"{model.id}: ratio={ratio:.4f} [{lo:.4f},{hi:.4f}] {verdict}",
                    file=sys.stderr,
                )

        load_after = snapshot_load(context)
        payload = {
            "schema_version": 1,
            "kind": "ab",
            "dataset": manifest.dataset,
            "created_utc": _now(),
            "verdict": combine_verdicts(r.get("verdict") or "pass" for r in rows),
            "context": {
                "target": args.target,
                "docker_context": context,
                "backend": backend,
                "device": target["device"],
                "runner": target["runner"],
                "base_sha": base_sha,
                "head_sha": head_sha,
                "base_image": base_image,
                "head_image": head_image,
                "base_digest": base_digest,
                "head_digest": head_digest,
                "base_pin_source": base_pin_source,
                "head_pin_source": head_pin_source,
                "recipe": "docker/runtime.Dockerfile --target runtime",
                "toolchain": dockerutil.toolchain_image(root),
                "rounds": rounds,
                "inject_regression": args.inject_regression,
                "load_before": load_before,
                "load_after": dict(load_after),
            },
            "models": rows,
        }
        _write_json(Path(args.out), payload)
        text = summarize(payload)
        if args.md:
            Path(args.md).write_text(text, encoding="utf-8")
        print(text)
        return 0 if payload["verdict"] in {
            VERDICT_PASS,
            VERDICT_PASS_LOW_POWER,
            VERDICT_INFEASIBLE,
            VERDICT_KNOWN_FAILURE,
            VERDICT_WARN,
            VERDICT_TIMEOUT,
        } else 1
    finally:
        if lock_held:
            release_lock(host, target["lock_path"])


def cmd_summarize(args: argparse.Namespace) -> int:
    print(summarize_path(args.results, markdown_out=args.md), end="")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Frozen dataset loader and A/B harness (#120, #121).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="validate the dataset manifest (CPU, cheap)")
    p_val.add_argument("--manifest", default=None)
    p_val.set_defaults(func=cmd_validate)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", default=None)
        p.add_argument("--dataset", default="v1")
        p.add_argument("--target", choices=sorted(TARGETS), required=True)
        p.add_argument("--docker-context", default=None)
        p.add_argument("--backend", choices=("omp", "cuda"), default=None)
        p.add_argument("--ids", nargs="+", default=None)
        p.add_argument("--tag", nargs="+", default=None)
        p.add_argument("--mode", choices=("accuracy", "perf", "compile", "run"), default="run")
        p.add_argument("--out", required=True)
        p.add_argument("--md", default=None)
        p.add_argument("--jit-volume", default=None)
        p.add_argument("--no-lock", action="store_true")
        p.add_argument("--skip-validate", action="store_true")
        p.add_argument("--guard-retries", type=int, default=6)
        p.add_argument("--commit", default=None)
        p.add_argument("--models-dir", default=None)
        p.add_argument("--work-dir", default=None)

    p_run = sub.add_parser("run", help="run a dataset against one image")
    add_common(p_run)
    p_run.add_argument("--image", required=True)
    p_run.set_defaults(func=cmd_run)

    p_ab = sub.add_parser("ab", help="interleaved A/B of two commits or images")
    add_common(p_ab)
    p_ab.add_argument("--base", default=None, help="base git SHA (default origin/develop)")
    p_ab.add_argument("--head", default=None, help="head git SHA (default HEAD)")
    p_ab.add_argument("--base-image", default=None)
    p_ab.add_argument("--head-image", default=None)
    p_ab.add_argument("--build", action="store_true", help="build both SHAs with the nightly recipe")
    p_ab.add_argument("--rounds", type=int, default=6)
    p_ab.add_argument(
        "--inject-regression",
        default=None,
        help=(
            "head-only: repeat (2× fzb run under one profiler), "
            "sleep:MS (outside the sampled region), or scale:F "
            "(calibrate base/head to integer device passes near F, e.g. scale:1.10)"
        ),
    )
    p_ab.set_defaults(func=cmd_ab)

    p_sum = sub.add_parser("summarize", help="markdown from a results JSON")
    p_sum.add_argument("results")
    p_sum.add_argument("--md", default=None)
    p_sum.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
