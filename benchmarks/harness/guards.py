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


def _nvidia_smi_cmd(
    context: str | None,
    gpus: str,
    args: Sequence[str],
    *,
    cuda_image: str = "nvidia/cuda:12.8.1-base-ubuntu24.04",
    pid_host: bool = True,
) -> list[str]:
    cmd = [*docker_prefix(context), "run", "--rm", "--gpus", gpus]
    if pid_host:
        cmd += ["--pid", "host"]
    cmd += ["--entrypoint", "nvidia-smi", cuda_image, *args]
    return cmd


def _run_nvidia_smi(
    context: str | None,
    gpus: str,
    args: Sequence[str],
    *,
    cuda_image: str = "nvidia/cuda:12.8.1-base-ubuntu24.04",
) -> subprocess.CompletedProcess[str]:
    proc = _run(_nvidia_smi_cmd(context, gpus, args, cuda_image=cuda_image, pid_host=True))
    if proc.returncode == 0:
        return proc
    err = (proc.stderr or "").lower()
    if "pid" in err or "permission" in err or "unknown flag" in err:
        return _run(_nvidia_smi_cmd(context, gpus, args, cuda_image=cuda_image, pid_host=False))
    return proc


def _parse_mib(text: Any) -> float | None:
    if text is None:
        return None
    token = str(text).strip().lower().replace("mib", "").replace("mb", "").strip()
    token = token.split()[0] if token else ""
    try:
        return float(token)
    except ValueError:
        return None


def _format_mib(mib: float) -> str:
    if mib >= 1024.0:
        return f" {mib / 1024.0:.1f} GiB"
    return f" {mib:.0f} MiB"


def _display_name(process_name: str, pid: str, cmdline: str = "") -> str:
    skip = {"", "[not found]", "[n/a]", "n/a", "python", "python3", "python3.12", "bash", "sh"}
    for source in (process_name, cmdline):
        for token in str(source).split():
            base = Path(token).name.strip()
            if base.lower() not in skip and not base.startswith("-"):
                return base
    return f"pid {pid}" if pid else "unknown"


def _pid_cmdlines(context: str | None, pids: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    missing: list[str] = []
    for pid in pids:
        path = Path(f"/proc/{pid}/cmdline")
        try:
            raw = path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except OSError:
            raw = ""
        if raw:
            found[pid] = raw
        else:
            missing.append(pid)
    if not missing:
        return found
    script = " ; ".join(
        f'printf "PID {pid} "; tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null; echo'
        for pid in missing
    )
    proc = _run(
        [*docker_prefix(context), "run", "--rm", "--pid", "host", "alpine:3.20", "sh", "-c", script]
    )
    if proc.returncode != 0:
        return found
    current: str | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("PID "):
            rest = line[4:]
            pid, _, cmd = rest.partition(" ")
            current = pid.strip()
            cmd = cmd.strip()
            if current and cmd:
                found[current] = cmd
                current = None
    return found


def gpu_compute_apps(
    context: str | None,
    *,
    gpus: str,
    cuda_image: str = "nvidia/cuda:12.8.1-base-ubuntu24.04",
) -> list[dict[str, str]]:
    proc = _run_nvidia_smi(
        context,
        gpus,
        [
            "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        cuda_image=cuda_image,
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
    cmdlines = _pid_cmdlines(context, [row["pid"] for row in rows if row.get("pid")])
    for row in rows:
        cmdline = cmdlines.get(row["pid"], "")
        if cmdline:
            row["cmdline"] = cmdline
        row["name"] = _display_name(row.get("process_name") or "", row.get("pid") or "", cmdline)
    return rows


def gpu_device_stats(
    context: str | None,
    *,
    gpus: str,
    cuda_image: str = "nvidia/cuda:12.8.1-base-ubuntu24.04",
) -> dict[str, Any]:
    proc = _run_nvidia_smi(
        context,
        gpus,
        [
            "--query-gpu=index,name,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cuda_image=cuda_image,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    line = proc.stdout.splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 7:
        return {}
    return {
        "index": parts[0],
        "name": parts[1],
        "pci_bus_id": parts[2],
        "utilization_gpu": _parse_mib(parts[3]),
        "utilization_memory": _parse_mib(parts[4]),
        "memory_used_mib": _parse_mib(parts[5]),
        "memory_total_mib": _parse_mib(parts[6]),
    }


def gpu_occupancy(context: str | None, gpus: str) -> dict[str, Any]:
    """Snapshot of compute apps + device util. Never decides idle vs busy."""
    return {
        "compute_apps": gpu_compute_apps(context, gpus=gpus),
        "gpu": gpu_device_stats(context, gpus=gpus),
    }


def format_foreign_load(snapshot: Mapping[str, Any] | None) -> str:
    """PR/history note, e.g. ``parakeet 3.0 GiB, util 14%`` or ``none``."""
    if not snapshot:
        return "none"
    parts: list[str] = []
    for app in snapshot.get("compute_apps") or []:
        if not isinstance(app, Mapping):
            continue
        name = str(app.get("name") or _display_name(
            str(app.get("process_name") or ""),
            str(app.get("pid") or ""),
            str(app.get("cmdline") or ""),
        ))
        mib = _parse_mib(app.get("used_gpu_memory") or app.get("memory_used_mib"))
        parts.append(f"{name}{_format_mib(mib) if mib is not None else ''}".strip())
    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), Mapping) else {}
    util = gpu.get("utilization_gpu") if isinstance(gpu, Mapping) else None
    if util is not None:
        parts.append(f"util {util:.0f}%")
    return ", ".join(parts) if parts else "none"


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
    shared_device: bool = False,
) -> dict[str, Any]:
    """Return a snapshot of load (and GPU occupancy).

    Exclusive GPUs (``shared_device=False``) raise ``GuardError`` if compute
    apps stay present. Shared GPUs (rtx3070, rtx5060ti) record foreign processes and
    return immediately — they never refuse or skip because of them. Host
    loadavg is ignored for GPU jobs (sibling GPU / Tdarr / Plex on xerxes).
    """
    last: dict[str, Any] = {}
    nproc = host_nproc(context)
    attempts = 1 if shared_device else retries + 1
    for attempt in range(attempts):
        load = host_loadavg(context)
        snapshot: dict[str, Any] = {
            "nproc": nproc,
            "loadavg": list(load),
            "load_fraction": load[0] / float(nproc),
            "attempt": attempt,
            "shared_device": shared_device,
        }
        busy_cpu = False
        if not gpus:
            busy_cpu = snapshot["load_fraction"] > load_fraction
        busy_gpu = False
        if gpus:
            snapshot.update(gpu_occupancy(context, gpus))
            snapshot["foreign_load"] = format_foreign_load(snapshot)
            # NVML reports containerized processes as process_name "[Not Found]".
            # Those are live occupants (e.g. /app/parakeet on xerxes GPU 1).
            busy_gpu = bool(snapshot.get("compute_apps")) and not shared_device
        last = snapshot
        if shared_device or (not busy_cpu and not busy_gpu):
            snapshot["idle"] = not busy_cpu and not bool(snapshot.get("compute_apps"))
            return snapshot
        if attempt < attempts - 1:
            time.sleep(retry_s)
    last["idle"] = False
    raise GuardError(f"host not idle after {retries} retries: {last}")


def snapshot_load(context: str | None, gpus: str | None = None) -> dict[str, Any]:
    nproc = host_nproc(context)
    load = host_loadavg(context)
    snapshot: dict[str, Any] = {
        "nproc": nproc,
        "loadavg": list(load),
        "load_fraction": load[0] / float(nproc),
    }
    if gpus:
        snapshot.update(gpu_occupancy(context, gpus))
        snapshot["foreign_load"] = format_foreign_load(snapshot)
    return snapshot
