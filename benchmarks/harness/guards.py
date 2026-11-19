"""Idle / load guards. Fail open to ``inconclusive`` rather than reporting noise."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_LOAD_FRACTION = 0.50
DEFAULT_RETRIES = 6
DEFAULT_RETRY_S = 15.0


class GuardError(RuntimeError):
    """Host is too busy to time."""


def _run(cmd: Sequence[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), check=False, capture_output=True, text=True, timeout=timeout)


def docker_prefix(context: str | None) -> list[str]:
    if context:
        return ["docker", "--context", context]
    return ["docker"]


def host_nproc(context: str | None) -> int:
    proc = _run([*docker_prefix(context), "run", "--rm", "alpine:3.20", "nproc"])
    if proc.returncode != 0:
        raise GuardError(f"nproc failed: {proc.stderr.strip()}")
    return int(proc.stdout.strip())


def host_loadavg(context: str | None) -> tuple[float, float, float]:
    proc = _run(
        [*docker_prefix(context), "run", "--rm", "alpine:3.20", "cat", "/proc/loadavg"]
    )
    if proc.returncode != 0:
        raise GuardError(f"loadavg failed: {proc.stderr.strip()}")
    parts = proc.stdout.split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def gpu_compute_apps(
    context: str | None,
    *,
    gpus: str,
    cuda_image: str = "nvidia/cuda:12.8.1-base-ubuntu24.04",
) -> list[dict[str, str]]:
    proc = _run(
        [
            *docker_prefix(context),
            "run",
            "--rm",
            "--gpus",
            gpus,
            "--entrypoint",
            "nvidia-smi",
            cuda_image,
            "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    if proc.returncode != 0:
        # Empty query with no apps sometimes exits 0 with empty stdout; a real failure is fatal.
        err = (proc.stderr or "").strip()
        if "NVIDIA-SMI has failed" in err or "couldn't communicate" in err.lower():
            raise GuardError(f"nvidia-smi failed: {err}")
        if proc.stdout.strip():
            raise GuardError(f"nvidia-smi compute-apps failed: {err or proc.stdout}")
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "gpu_bus_id": parts[1],
                "pid": parts[2],
                "process_name": parts[3],
                "used_gpu_memory": parts[4],
            }
        )
    return rows


def _acquire_lock_once(host: str | None, path: str) -> None:
    if not host:
        try:
            Path(path).mkdir(mode=0o700)
        except FileExistsError as exc:
            raise GuardError(f"could not acquire {path}: already held") from exc
        except OSError as exc:
            raise GuardError(f"could not acquire {path}: {exc}") from exc
        return
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir {path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise GuardError(
            f"could not acquire {host}:{path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def acquire_lock(
    host: str | None,
    path: str,
    *,
    retries: int = 0,
    retry_s: float = DEFAULT_RETRY_S,
) -> None:
    """``mkdir`` a directory lock. Local when ``host`` is empty; else ssh.

    CI jobs run on the GPU host itself (``--docker-context default``) and
    take ``/tmp/gpu-lock-xerxes-{0,1}`` locally so they interlock with
    developer sessions that acquire the same path over ssh. Retry when the
    lock is held; do not steal it.
    """
    last: GuardError | None = None
    for attempt in range(retries + 1):
        try:
            _acquire_lock_once(host, path)
            return
        except GuardError as exc:
            last = exc
            if attempt < retries:
                time.sleep(retry_s)
    assert last is not None
    raise last


def release_lock(host: str | None, path: str) -> None:
    if not host:
        try:
            Path(path).rmdir()
        except OSError:
            pass
        return
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"rmdir {path}"],
        check=False,
        capture_output=True,
        text=True,
    )


def wait_for_idle(
    *,
    context: str | None,
    gpus: str | None,
    load_fraction: float = DEFAULT_LOAD_FRACTION,
    retries: int = DEFAULT_RETRIES,
    retry_s: float = DEFAULT_RETRY_S,
) -> dict[str, Any]:
    """Return a snapshot of load (and GPU apps). Raises ``GuardError`` after retries."""
    last: dict[str, Any] = {}
    nproc = host_nproc(context)
    for attempt in range(retries + 1):
        load = host_loadavg(context)
        snapshot: dict[str, Any] = {
            "nproc": nproc,
            "loadavg": list(load),
            "load_fraction": load[0] / float(nproc),
            "attempt": attempt,
        }
        # GPU jobs lock a single device. Host loadavg includes the sibling GPU
        # (and Tdarr/Plex on xerxes); do not treat that as this device being busy.
        busy_cpu = False
        if not gpus:
            busy_cpu = snapshot["load_fraction"] > load_fraction
        busy_gpu = False
        if gpus:
            apps = gpu_compute_apps(context, gpus=gpus)
            snapshot["compute_apps"] = apps
            busy_gpu = bool(apps)
        last = snapshot
        if not busy_cpu and not busy_gpu:
            snapshot["idle"] = True
            return snapshot
        if attempt < retries:
            time.sleep(retry_s)
    last["idle"] = False
    raise GuardError(f"host not idle after {retries} retries: {last}")


def snapshot_load(context: str | None) -> Mapping[str, Any]:
    nproc = host_nproc(context)
    load = host_loadavg(context)
    return {
        "nproc": nproc,
        "loadavg": list(load),
        "load_fraction": load[0] / float(nproc),
    }
