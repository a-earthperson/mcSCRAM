"""Build the nightly runtime image from a git SHA (PR #91 recipe)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..loader import repo_root

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DockerError(RuntimeError):
    pass


def stage_file(
    local: Path,
    *,
    host: str | None,
    dest_dir: str,
    dest_name: str,
) -> str:
    """Copy ``local`` to ``dest_dir/dest_name`` on ``host`` (ssh) or this machine."""
    dest = f"{dest_dir.rstrip('/')}/{dest_name}"
    if host:
        mkdir = run_cmd(
            ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {shlex.quote(dest_dir)}"]
        )
        if mkdir.returncode != 0:
            raise DockerError(mkdir.stderr.strip() or f"mkdir failed on {host}:{dest_dir}")
        proc = subprocess.run(
            ["scp", "-o", "BatchMode=yes", str(local), f"{host}:{dest}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise DockerError((proc.stderr or proc.stdout or "scp failed").strip())
        return dest
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dest)
    return dest


def docker_cmd(context: str | None, *args: str) -> list[str]:
    cmd = ["docker"]
    if context:
        cmd += ["--context", context]
    cmd.extend(args)
    return cmd


def run_cmd(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True if stdin is None else False,
            cwd=None if cwd is None else str(cwd),
            timeout=timeout,
            env=merged,
            input=stdin,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"command timed out after {timeout}s: {cmd[0]}") from exc
    if stdin is not None:
        # Re-decode after binary stdin.
        proc = subprocess.CompletedProcess(
            proc.args,
            proc.returncode,
            proc.stdout.decode() if isinstance(proc.stdout, bytes) else proc.stdout,
            proc.stderr.decode() if isinstance(proc.stderr, bytes) else proc.stderr,
        )
    return proc


def stop_container(context: str | None, name: str, *, grace_s: int = 5) -> None:
    """Stop a named container left behind after a wall-clock timeout."""
    run_cmd(docker_cmd(context, "stop", "-t", str(grace_s), name), timeout=grace_s + 20)
    run_cmd(docker_cmd(context, "rm", "-f", name), timeout=20)


def parse_toolchain_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def toolchain_image(root: Path | None = None) -> str:
    repo = root or repo_root()
    env_path = repo / "toolchain.env"
    env = parse_toolchain_env(env_path)
    sha12 = hashlib.sha256(env_path.read_bytes()).hexdigest()[:12]
    ref = env["ACPP_REF"]
    return f"ghcr.io/fuzzybit-org/fuzzybits-toolchain:{ref}-generic-{sha12}"


def image_name(image: str) -> str:
    """Repository name without tag or digest (``ghcr.io/org/name``)."""
    stripped = image.strip()
    if "@" in stripped:
        stripped = stripped.split("@", 1)[0]
    # Drop a tag, but not a registry port (localhost:5000/name).
    name, sep, rest = stripped.rpartition(":")
    if sep and "/" in rest:
        return stripped
    if sep and name:
        return name
    return stripped


def parse_digest(text: str) -> str | None:
    raw = text.strip()
    if "@sha256:" in raw:
        raw = raw.rsplit("@", 1)[1]
    if _DIGEST_RE.match(raw):
        return raw
    return None


def pin_ref(image: str, digest: str) -> str:
    """``name:tag`` + ``sha256:...`` → ``name@sha256:...`` (immutable)."""
    parsed = parse_digest(digest)
    if parsed is None:
        raise DockerError(f"not a full image digest: {digest!r}")
    if "@sha256:" in image:
        return f"{image_name(image)}@{parsed}"
    return f"{image_name(image)}@{parsed}"


def registry_digest(image: str) -> str:
    """Resolve ``name:tag`` to the registry manifest digest (not a local image id).

    A/B and history must pin ``image@sha256:...`` once from the registry. Local
    ``docker inspect .Id`` differs per host for a mutable tag (gateway vs xerxes
    resolved ``nightly-generic`` to different config ids).
    """
    already = parse_digest(image)
    if already and "@sha256:" in image:
        return already
    inspect = run_cmd(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "--format",
            "{{.Manifest.Digest}}",
            image,
        ]
    )
    digest = parse_digest(inspect.stdout or "")
    if inspect.returncode == 0 and digest:
        return digest
    manifest = run_cmd(["docker", "manifest", "inspect", image])
    if manifest.returncode == 0 and manifest.stdout.strip():
        try:
            payload = json.loads(manifest.stdout)
        except json.JSONDecodeError as exc:
            raise DockerError(
                f"manifest inspect of {image} is not JSON: {exc}"
            ) from exc
        descriptor = payload.get("Descriptor") or {}
        digest = parse_digest(str(descriptor.get("digest") or payload.get("digest") or ""))
        if digest:
            return digest
    detail = (inspect.stderr or manifest.stderr or "").strip()
    raise DockerError(
        f"could not resolve {image} from the registry "
        f"(pass image@sha256:...): {detail[-500:]}"
    )


def pin_image(
    image: str,
    *,
    allow_local: bool = False,
    context: str | None = None,
) -> dict[str, str]:
    """Return ``{tag, ref, digest, source}`` with ``ref`` suitable for ``docker run``.

    A/B of a registry tag must pin from the registry (``allow_local=False``).
    Fresh ``--build`` images and one-off local tags may fall back to inspect.
    """
    already = parse_digest(image)
    if already and "@sha256:" in image:
        ref = pin_ref(image, already)
        return {"tag": image, "ref": ref, "digest": already, "source": "pinned"}
    try:
        digest = registry_digest(image)
    except DockerError:
        if not allow_local:
            raise
        digest = image_digest(context, image)
        return {"tag": image, "ref": image, "digest": digest, "source": "local"}
    ref = pin_ref(image, digest)
    return {"tag": image, "ref": ref, "digest": digest, "source": "registry"}


def image_digest(context: str | None, image: str) -> str:
    """Digest of a local image. Prefer the registry pin when the ref already has one."""
    already = parse_digest(image)
    if already:
        return already
    proc = run_cmd(
        docker_cmd(
            context,
            "inspect",
            "--format",
            "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
            image,
        )
    )
    if proc.returncode != 0:
        raise DockerError(proc.stderr.strip() or f"inspect failed for {image}")
    digest = parse_digest(proc.stdout)
    if digest:
        return digest
    # Bare local id (sha256:… of the config), not a registry manifest.
    raw = proc.stdout.strip()
    if _DIGEST_RE.match(raw):
        return raw
    raise DockerError(f"inspect of {image} returned no digest: {raw!r}")


def git_archive_tar(root: Path, sha: str) -> bytes:
    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"
    proc = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", sha],
        check=False,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise DockerError(proc.stderr.decode() if proc.stderr else f"git archive {sha} failed")
    return proc.stdout


def build_runtime_image(
    *,
    sha: str,
    context: str,
    root: Path | None = None,
    tag: str | None = None,
    toolchain: str | None = None,
) -> dict[str, Any]:
    """``git archive <sha> | docker build -`` of ``docker/runtime.Dockerfile --target runtime``."""
    repo = root or repo_root()
    image_tag = tag or f"fuzzybits-runtime:bench-{sha[:12]}"
    toolchain_ref = toolchain or toolchain_image(repo)
    env = parse_toolchain_env(repo / "toolchain.env")
    archive = git_archive_tar(repo, sha)
    build_cmd = docker_cmd(
        context,
        "build",
        "--target",
        "runtime",
        "-f",
        "docker/runtime.Dockerfile",
        "--build-arg",
        f"TOOLCHAIN_IMAGE={toolchain_ref}",
        "--build-arg",
        "CMAKE_PRESET=generic",
        "--build-arg",
        "FUZZYBITS_VARIANT=generic",
        "--build-arg",
        "INCLUDE_SSCP_JIT=1",
        "--build-arg",
        "ACPP_TARGETS=generic",
        "--build-arg",
        f"CLANG_VERSION={env.get('CLANG_VERSION', '20')}",
        "--build-arg",
        f"ACPP_REF={env.get('ACPP_REF', 'v25.10.0')}",
        "--build-arg",
        f"ACPP_COMMIT={env.get('ACPP_COMMIT', '')}",
        "--build-arg",
        f"CUDA_VERSION={env.get('CUDA_VERSION', '12.8.1')}",
        "--build-arg",
        f"FUZZYBITS_REVISION={sha}",
        "-t",
        image_tag,
        "-",
    )
    proc = subprocess.run(
        build_cmd,
        check=False,
        capture_output=True,
        env={**os.environ, "COPYFILE_DISABLE": "1"},
        input=archive,
    )
    stdout = proc.stdout.decode() if isinstance(proc.stdout, bytes) else (proc.stdout or "")
    stderr = proc.stderr.decode() if isinstance(proc.stderr, bytes) else (proc.stderr or "")
    if proc.returncode != 0:
        raise DockerError(
            f"docker build of {sha[:12]} failed (rc={proc.returncode}):\n{stderr[-4000:]}"
        )
    digest = image_digest(context, image_tag)
    return {
        "tag": image_tag,
        "digest": digest,
        "sha": sha,
        "toolchain": toolchain_ref,
        "recipe": "docker/runtime.Dockerfile --target runtime",
        "log_tail": (stdout + stderr)[-2000:],
    }
