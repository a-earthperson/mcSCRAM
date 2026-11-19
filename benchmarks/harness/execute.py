"""Run ``fzb compile`` / ``fzb inspect`` / ``fzb run`` inside a runtime image."""

from __future__ import annotations

import json
import shlex
import textwrap
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .dockerutil import docker_cmd, run_cmd, stop_container
from .profile import parse_profiler

# GNU timeout 124; 137 is SIGKILL after --kill-after.
_TIMEOUT_RCS = {124, 128 + 9, 137}

INNER_RUNNER = textwrap.dedent(
    r"""
    import json, subprocess, sys, time
    payload = json.loads(sys.argv[1])
    inner = list(payload["cmd"])
    timeout = payload.get("timeout")
    extra = payload.get("extra_runs", 0)
    sleep_ms = payload.get("sleep_ms", 0)
    t0 = time.perf_counter()
    if sleep_ms:
        time.sleep(sleep_ms / 1000.0)
    cmd = list(inner)
    if timeout:
        cmd = ["timeout", "--signal=TERM", "--kill-after=5", str(int(timeout)), *inner]
    runs = []
    rc = 0
    stdout = ""
    stderr = ""
    timed_out = False
    for i in range(1 + extra):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=(int(timeout) + 15) if timeout else None,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += exc.stdout or ""
            stderr += (exc.stderr or "") + "\nwall-clock timeout"
            rc = 124
            runs.append({"rc": 124, "stdout": exc.stdout or "", "stderr": "timeout"})
            break
        runs.append({"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        stdout += proc.stdout
        stderr += proc.stderr
        rc = proc.returncode
        if rc in (124, 137):
            timed_out = True
            break
        if rc != 0:
            break
    wall = time.perf_counter() - t0
    json.dump(
        {"wall_s": wall, "rc": rc, "stdout": stdout, "stderr": stderr,
         "runs": runs, "timed_out": timed_out},
        sys.stdout,
    )
    sys.exit(0)
    """
).strip()


# Runtime sample shape at the configured trial count. A 1-trial fzb run
# executes one uint64 bitpack (n=64), which is not the device pass used at
# 1e8 (chinese on V100S is 32·2^20).
INNER_PASS_SHAPE = textwrap.dedent(
    r"""
    import json, sys, time
    payload = json.loads(sys.argv[1])
    t0 = time.perf_counter()
    rc = 0
    stderr = ""
    shape = {}
    try:
        import fuzzybits
        fuzzybits.init(backend=payload["backend"])
        session = fuzzybits.Session(
            payload["pcfb"],
            backend=payload["backend"],
            num_trials=int(payload["trials"]),
        )
        try:
            shape = dict(session.sample_shape)
        finally:
            session.close()
    except Exception as exc:
        rc = 1
        stderr = f"{type(exc).__name__}: {exc}"
    tpp = int(shape.get("trials_per_pass") or 0)
    stdout = json.dumps({"sample_shape": shape, "queries": [{"name": "pass_size", "n": tpp}]})
    json.dump(
        {"wall_s": time.perf_counter() - t0, "rc": rc, "stdout": stdout,
         "stderr": stderr, "runs": [], "timed_out": False, "sample_shape": shape},
        sys.stdout,
    )
    """
).strip()


# Time sampling in-process after AdaptiveCpp init + a tiny JIT warmup so
# CPU A/B does not include container start or interpreter startup.
INNER_INPROCESS = textwrap.dedent(
    r"""
    import json, signal, sys, time
    payload = json.loads(sys.argv[1])
    backend = payload["backend"]
    pcfb = payload["pcfb"]
    trials = int(payload["trials"])
    extra = int(payload.get("extra_runs", 0))
    sleep_ms = int(payload.get("sleep_ms", 0))
    warmup_trials = int(payload.get("warmup_trials", 1048576))
    timeout = payload.get("timeout")
    timed_out = False
    if timeout:
        def _alarm(signum, frame):
            raise TimeoutError("in-process sampling timed out")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout))
    if sleep_ms:
        time.sleep(sleep_ms / 1000.0)
    import fuzzybits
    fuzzybits.init(backend=backend)
    fuzzybits.run(pcfb, trials=warmup_trials, backend=backend)
    t0 = time.perf_counter()
    rows = None
    rc = 0
    stderr = ""
    try:
        for i in range(1 + extra):
            rows = fuzzybits.run(pcfb, trials=trials, backend=backend)
    except TimeoutError as exc:
        timed_out = True
        rc = 124
        stderr = str(exc)
    except Exception as exc:
        rc = 1
        stderr = f"{type(exc).__name__}: {exc}"
    if timeout:
        signal.alarm(0)
    wall = time.perf_counter() - t0
    queries = []
    if rows is not None:
        queries = [
            {
                "name": r.name,
                "estimate": r.estimate,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "n": r.n,
                "hits": r.hits,
            }
            for r in rows
        ]
    stdout = json.dumps({"queries": queries, "sample_s": wall})
    json.dump(
        {"wall_s": wall, "sample_s": wall, "rc": rc, "stdout": stdout,
         "stderr": stderr, "runs": [], "timed_out": timed_out},
        sys.stdout,
    )
    sys.exit(0)
    """
).strip()


def _volume_args(volumes: Sequence[str]) -> list[str]:
    args: list[str] = []
    for spec in volumes:
        args.extend(["-v", spec])
    return args


def _env_args(env: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    return args


def container_env(*, backend: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {
        "ACPP_VISIBILITY_MASK": backend,
        "ACPP_DEBUG_LEVEL": "0",
        "ACPP_PERSISTENT_RUNTIME": "1",
        "ACPP_APPDB_DIR": "/home/coder/.cache/fuzzybits/acpp",
    }
    if backend == "omp":
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"
        env["OMP_NUM_THREADS"] = str((extra or {}).get("OMP_NUM_THREADS") or "8")
    if extra:
        env.update(extra)
    return env


def docker_run(
    *,
    context: str | None,
    image: str,
    args: Sequence[str],
    volumes: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    gpus: str | None = None,
    entrypoint: str = "python3",
    user: str | None = None,
    timeout: float | None = None,
    name: str | None = None,
    cpuset_cpus: str | None = None,
) -> tuple[int, str, str, bool]:
    """Run a container. The fourth value is True on a wall-clock timeout."""
    container = name or f"fzb-bench-{uuid.uuid4().hex[:12]}"
    cmd = docker_cmd(context, "run", "--rm", "--name", container)
    if gpus:
        cmd += ["--gpus", gpus]
    if cpuset_cpus:
        cmd += ["--cpuset-cpus", cpuset_cpus]
    if user:
        cmd += ["--user", user]
    cmd += _volume_args(volumes)
    if env:
        cmd += _env_args(env)
    cmd += ["--entrypoint", entrypoint, image, *args]
    try:
        proc = run_cmd(cmd, timeout=timeout)
    except TimeoutError:
        stop_container(context, container)
        return 124, "", f"wall-clock timeout; docker stop {container}", True
    timed_out = proc.returncode in _TIMEOUT_RCS
    return proc.returncode, proc.stdout, proc.stderr, timed_out


def run_inner(
    *,
    context: str | None,
    image: str,
    inner_cmd: Sequence[str],
    volumes: Sequence[str],
    env: Mapping[str, str],
    gpus: str | None,
    timeout_s: int,
    extra_runs: int = 0,
    sleep_ms: int = 0,
    profiler: str | None = None,
    profiler_bin: str | None = None,
    cpuset_cpus: str | None = None,
    inprocess: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if inprocess and inprocess.get("mode") == "pass_shape":
        payload = {**dict(inprocess), "timeout": timeout_s}
        py = ["python3", "-c", INNER_PASS_SHAPE, json.dumps(payload, separators=(",", ":"))]
    elif inprocess:
        payload = {
            **dict(inprocess),
            "timeout": timeout_s,
            "extra_runs": extra_runs,
            "sleep_ms": sleep_ms,
        }
        py = ["python3", "-c", INNER_INPROCESS, json.dumps(payload, separators=(",", ":"))]
    else:
        payload = {
            "cmd": list(inner_cmd),
            "timeout": timeout_s,
            "extra_runs": extra_runs,
            "sleep_ms": sleep_ms,
        }
        py = ["python3", "-c", INNER_RUNNER, json.dumps(payload, separators=(",", ":"))]
    # Profile the python wrapper *and* its fzb children so --inject-regression
    # repeat (two fzb runs in one session) is visible as doubled kernel time.
    if profiler == "nvprof":
        entrypoint = profiler_bin or "/opt/cuda/cuda/bin/nvprof"
        args = ["--print-gpu-summary", "--profile-child-processes", "--", *py]
    elif profiler == "nsys":
        entrypoint = profiler_bin or "/opt/pr91/nvidia/nsight-systems/2025.3.2/bin/nsys"
        args = [
            "profile",
            "--stats=true",
            "--trace-fork-before-exec=true",
            "--force-overwrite=true",
            "-o",
            "/tmp/bench-nsys",
            "--",
            *py,
        ]
    elif profiler:
        raise ValueError(f"unknown profiler {profiler!r}")
    else:
        entrypoint = "python3"
        args = py[1:]

    rc, stdout, stderr, docker_timeout = docker_run(
        context=context,
        image=image,
        args=args,
        volumes=volumes,
        env=env,
        gpus=gpus,
        entrypoint=entrypoint,
        timeout=timeout_s + 90,
        cpuset_cpus=cpuset_cpus,
    )
    combined = stdout + "\n" + stderr
    parsed: dict[str, Any] | None = None
    # Inner JSON is the last JSON object in stdout (profilers print before/after).
    for candidate in _json_objects(stdout):
        if isinstance(candidate, dict) and "wall_s" in candidate:
            parsed = candidate
    if parsed is None:
        return {
            "ok": False,
            "rc": rc,
            "wall_s": timeout_s if docker_timeout else None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": docker_timeout,
            "error": "wall-clock timeout" if docker_timeout else "inner runner produced no JSON",
        }
    timed_out = bool(parsed.get("timed_out")) or docker_timeout
    result = {
        "ok": parsed.get("rc", rc) == 0 and not timed_out,
        "rc": parsed.get("rc", rc),
        "wall_s": parsed.get("wall_s"),
        "sample_s": parsed.get("sample_s"),
        "stdout": parsed.get("stdout") or "",
        "stderr": parsed.get("stderr") or "",
        "docker_rc": rc,
        "docker_stderr": stderr,
        "timed_out": timed_out,
    }
    if profiler:
        inner_text = (
            combined
            + "\n"
            + (parsed.get("stdout") or "")
            + "\n"
            + (parsed.get("stderr") or "")
        )
        try:
            result["profiler"] = parse_profiler(inner_text, profiler)
        except Exception as exc:  # noqa: BLE001 - keep the run; mark profiler failure
            result["profiler_error"] = str(exc)
            result["profiler_raw_tail"] = inner_text[-4000:]
    return result


def _json_objects(text: str) -> list[Any]:
    objects: list[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        while i < len(text) and text[i] not in "{[":
            i += 1
        if i >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        objects.append(obj)
        i = end
    return objects


def parse_fzb_run_json(stdout: str) -> dict[str, Any]:
    objects = _json_objects(stdout)
    for obj in reversed(objects):
        if isinstance(obj, dict) and "queries" in obj:
            return obj
    raise ValueError("no fzb run JSON in stdout")


def parse_inspect_json(stdout: str) -> dict[str, Any]:
    objects = _json_objects(stdout)
    for obj in reversed(objects):
        if isinstance(obj, dict) and "schema_version" in obj:
            return obj
    raise ValueError("no fzb inspect JSON in stdout")


def compile_and_inspect(
    *,
    context: str | None,
    image: str,
    model_xml: str,
    pcfb_out: str,
    volumes: Sequence[str],
    env: Mapping[str, str],
    timeout_s: int = 120,
    gpus: str | None = None,
    cpuset_cpus: str | None = None,
) -> dict[str, Any]:
    compile_cmd = ["fzb", "compile", model_xml, "-c", "2", "--no-kn", "-o", pcfb_out]
    compiled = run_inner(
        context=context,
        image=image,
        inner_cmd=compile_cmd,
        volumes=volumes,
        env=env,
        gpus=gpus,
        timeout_s=timeout_s,
        cpuset_cpus=cpuset_cpus,
    )
    if compiled.get("timed_out") or not compiled["ok"]:
        compiled["stage"] = "compile"
        return compiled
    inspected = run_inner(
        context=context,
        image=image,
        inner_cmd=["fzb", "inspect", pcfb_out, "--json"],
        volumes=volumes,
        env=env,
        gpus=gpus,
        timeout_s=min(timeout_s, 60),
        cpuset_cpus=cpuset_cpus,
    )
    if inspected.get("timed_out") or not inspected["ok"]:
        inspected["stage"] = "inspect"
        inspected["compile_wall_s"] = compiled.get("wall_s")
        return inspected
    try:
        summary = parse_inspect_json(inspected["stdout"])
    except ValueError as exc:
        inspected["ok"] = False
        inspected["error"] = str(exc)
        inspected["stage"] = "inspect"
        return inspected
    return {
        "ok": True,
        "stage": "compile",
        "compile_wall_s": compiled.get("wall_s"),
        "inspect": summary,
        "compile_stderr": compiled.get("stderr"),
        "pcfb": pcfb_out,
        "timed_out": False,
    }


def fzb_run(
    *,
    context: str | None,
    image: str,
    pcfb: str,
    trials: int,
    backend: str,
    volumes: Sequence[str],
    env: Mapping[str, str],
    timeout_s: int,
    gpus: str | None = None,
    profiler: str | None = None,
    profiler_bin: str | None = None,
    extra_runs: int = 0,
    sleep_ms: int = 0,
    cpuset_cpus: str | None = None,
    inprocess: bool = False,
) -> dict[str, Any]:
    inner = ["fzb", "run", pcfb, "--trials", str(trials), "--backend", backend, "--json"]
    inprocess_payload = None
    if inprocess:
        inprocess_payload = {
            "backend": backend,
            "pcfb": pcfb,
            "trials": int(trials),
        }
    result = run_inner(
        context=context,
        image=image,
        inner_cmd=inner,
        volumes=volumes,
        env=env,
        gpus=gpus,
        timeout_s=timeout_s,
        extra_runs=extra_runs,
        sleep_ms=sleep_ms,
        profiler=profiler,
        profiler_bin=profiler_bin,
        cpuset_cpus=cpuset_cpus,
        inprocess=inprocess_payload,
    )
    if result["ok"]:
        try:
            parsed = parse_fzb_run_json(result["stdout"])
            result["run"] = parsed
            if parsed.get("sample_s") is not None:
                result["sample_s"] = parsed["sample_s"]
        except ValueError as exc:
            result["ok"] = False
            result["error"] = str(exc)
    return result


def probe_pass_size(
    *,
    context: str | None,
    image: str,
    pcfb: str,
    trials: int,
    backend: str,
    volumes: Sequence[str],
    env: Mapping[str, str],
    timeout_s: int,
    gpus: str | None = None,
    cpuset_cpus: str | None = None,
) -> dict[str, Any]:
    """Read ``sample_shape.trials_per_pass`` at ``trials`` without sampling."""
    result = run_inner(
        context=context,
        image=image,
        inner_cmd=["true"],
        volumes=volumes,
        env=env,
        gpus=gpus,
        timeout_s=timeout_s,
        cpuset_cpus=cpuset_cpus,
        inprocess={
            "mode": "pass_shape",
            "backend": backend,
            "pcfb": pcfb,
            "trials": int(trials),
        },
    )
    if result.get("ok"):
        try:
            parsed = parse_fzb_run_json(result["stdout"])
            result["run"] = parsed
            if parsed.get("sample_shape") is not None:
                result["sample_shape"] = parsed["sample_shape"]
        except ValueError as exc:
            result["ok"] = False
            result["error"] = str(exc)
    return result


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)
