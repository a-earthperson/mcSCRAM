#!/usr/bin/env python3
"""Cross-arch SASS vs PTX vs SSCP matrix runner. JSONL to stdout."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time

N = "2e9"
FLAGS = f"--monte-carlo --probability --pdag -c 2 --no-kn -P wald -N {N} --no-report -V 3"
V100S_UUID = "GPU-b7a28dca-3e96-ebb9-8db5-e8902a70f33b"


def run(cmd, timeout=90):
    t0 = time.time()
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
    )
    return p.returncode, p.stdout, time.time() - t0


def v100s_busy():
    rc, out, _ = run(
        [
            "ssh",
            "xerxes",
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader",
        ],
        timeout=20,
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip() and V100S_UUID in ln]


def wait_v100s(max_wait=90):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        busy = v100s_busy()
        if not busy:
            return True, []
        time.sleep(3)
    return False, v100s_busy()


def parse_scram(out: str) -> dict:
    d = {}
    m = re.search(r"Root Event Probability:\s*([0-9.eE+-]+)", out)
    if m:
        d["p"] = float(m.group(1))
    m = re.search(r"Calculated observed tallies for .* in ([0-9.eE+-]+)", out)
    if m:
        d["tally_s"] = float(m.group(1))
    m = re.search(r"bitpacks_per_batch=\s*(\d+)", out)
    if m:
        d["bitpacks"] = int(m.group(1))
    m = re.search(r"num_bitpacks=\s*(\d+)", out)
    if m:
        d["num_bitpacks"] = int(m.group(1))
    m = re.search(r"num_events:(\d+)", out)
    if m:
        d["num_events"] = int(m.group(1))
    errs = [ln for ln in out.splitlines() if re.search(r"error|Error|terminat|CUDA", ln)]
    if errs:
        d["err_snip"] = "\n".join(errs[:8])
    return d


def parse_nsys_kernels(out: str) -> dict:
    be = {"total_ns": 0, "instances": 0, "avg_ns": 0.0}
    gate = {"total_ns": 0, "instances": 0, "avg_ns": 0.0}
    tally = {"total_ns": 0, "instances": 0, "avg_ns": 0.0}
    other = []
    in_kern = False
    for ln in out.splitlines():
        if "cuda_gpu_kern_sum" in ln:
            in_kern = True
            continue
        if in_kern and ln.startswith("[") and "Executing" in ln:
            in_kern = False
            continue
        if not in_kern:
            continue
        m = re.match(r"\s*[\d.]+\s+(\d+)\s+(\d+)\s+([\d.]+)\s+", ln)
        if not m:
            continue
        total_ns, inst, avg = int(m.group(1)), int(m.group(2)), float(m.group(3))
        name = ln[m.end() :].strip()
        lname = name.lower()
        if "basic_event" in lname:
            be["total_ns"] += total_ns
            be["instances"] += inst
        elif "tally" in lname:
            tally["total_ns"] += total_ns
            tally["instances"] += inst
        elif (
            "kernel::op" in lname
            or "connective" in lname
            or "optype" in lname
            or "<scram::mc::kernel::op" in lname
        ):
            gate["total_ns"] += total_ns
            gate["instances"] += inst
        else:
            other.append({"total_ns": total_ns, "instances": inst, "name": name[:80]})
    for rec in (be, gate, tally):
        if rec["instances"]:
            rec["avg_ns"] = rec["total_ns"] / rec["instances"]
    return {"be": be, "gate": gate, "tally": tally, "other": other[:4]}


def norm_ms_per_1e9(total_ns, passes, bitpacks):
    if not (total_ns and passes and bitpacks):
        return None
    trials = passes * bitpacks * 64
    return (total_ns / 1e6) / trials * 1e9


def emit(obj):
    print(json.dumps(obj, default=str), flush=True)


def cell(
    tag,
    ctx,
    gpus,
    image,
    model,
    path,
    extra=None,
    nsys=False,
    nsys_bin=False,
    env_add=None,
    wait_v100=False,
):
    rec = {
        "tag": tag,
        "ctx": ctx,
        "gpus": gpus,
        "image": image,
        "model": model,
        "nsys": nsys,
    }
    if wait_v100:
        ok, busy = wait_v100s()
        rec["v100s_wait_ok"] = ok
        rec["v100s_busy"] = busy
        if not ok:
            rec["status"] = "skipped_busy"
            emit(rec)
            return rec
    extra = list(extra or [])
    extra += ["-v", "/tmp/mcscram-inputs:/extra-input:ro"]
    if env_add:
        for k, v in env_add.items():
            extra += ["-e", f"{k}={v}"]
    cmd = ["timeout", "60", "docker", "--context", ctx, "run", "--rm", "--gpus", gpus]
    cmd += extra
    if nsys:
        cmd += [
            "--user",
            "root",
            "--cap-add",
            "SYS_ADMIN",
            "--security-opt",
            "seccomp=unconfined",
        ]
        if nsys_bin:
            cmd += [
                "-v",
                "/tmp/mcprof/tools/ncu/host/target-linux-x64:/nsys:ro",
                "--entrypoint",
                "/nsys/nsys",
                image,
                "profile",
                "-t",
                "cuda",
                "--force-overwrite=true",
                "-o",
                "/tmp/rep",
                "--stats=true",
                "/usr/local/bin/scram-cli",
            ]
            cmd += FLAGS.split() + [path]
        else:
            cmd += [
                "--entrypoint",
                "bash",
                image,
                "-lc",
                f"nsys profile -t cuda --force-overwrite=true -o /tmp/rep --stats=true /usr/local/bin/scram-cli {FLAGS} {path}",
            ]
    else:
        cmd += [image] + FLAGS.split() + [path]
    rec["cmd"] = " ".join(cmd[:14]) + " ..."
    try:
        rc, out, wall = run(cmd, timeout=75)
    except subprocess.TimeoutExpired:
        rec.update(status="timeout", wall_s=75)
        emit(rec)
        return rec
    rec["rc"] = rc
    rec["wall_s"] = round(wall, 3)
    rec.update(parse_scram(out))
    if nsys:
        rec["nsys_kern"] = parse_nsys_kernels(out)
        bp = rec.get("bitpacks") or rec.get("num_bitpacks")
        nk = rec["nsys_kern"]
        rec["be_ms_per_1e9"] = norm_ms_per_1e9(
            nk["be"]["total_ns"], nk["be"]["instances"], bp
        )
        rec["gate_ms_per_1e9"] = norm_ms_per_1e9(
            nk["gate"]["total_ns"], nk["be"]["instances"] or nk["gate"]["instances"], bp
        )
        rec["be_avg_us"] = (nk["be"]["avg_ns"] / 1e3) if nk["be"]["avg_ns"] else None
        rec["gate_avg_us"] = (nk["gate"]["avg_ns"] / 1e3) if nk["gate"]["avg_ns"] else None
        rec["be_instances"] = nk["be"]["instances"]
        rec["gate_instances"] = nk["gate"]["instances"]
    rec["status"] = "ok" if rc == 0 and "p" in rec else "fail"
    rec["out_tail"] = "\n".join(out.splitlines()[-10:])
    emit(rec)
    return rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = [
        ("das9201", "/app/input/das9201.xml"),
        ("das9701", "/extra-input/das9701.xml"),
    ]
    chinese = ("chinese", "/app/input/chinese.xml")
    ptx_env = {
        "CUDA_FORCE_PTX_JIT": "1",
        "CUDA_CACHE_PATH": "/cuda-cache",
        "CUDA_CACHE_DISABLE": "0",
    }

    if which in ("all", "v100s"):
        ok, busy = wait_v100s()
        emit({"tag": "v100s_idle", "ok": ok, "busy": busy})
        for name, path in models:
            cell(
                f"v100s.sass.{name}",
                "xerxes",
                "device=0",
                "mcscram:cuda12.8-sm70",
                name,
                path,
                wait_v100=True,
            )
            cell(
                f"v100s.sass.{name}.nsys",
                "xerxes",
                "device=0",
                "mcscram:cuda12.8-sm70-prof",
                name,
                path,
                nsys=True,
                wait_v100=True,
            )
        cell(
            "v100s.sass.chinese",
            "xerxes",
            "device=0",
            "mcscram:cuda12.8-sm70",
            *chinese,
            wait_v100=True,
        )
        extra_ptx = ["-v", "/tmp/mcscram-ptx-jit-sm70:/cuda-cache"]
        run(
            [
                "timeout",
                "20",
                "docker",
                "--context",
                "xerxes",
                "run",
                "--rm",
                "-v",
                "/tmp/mcscram-ptx-jit-sm70:/c",
                "alpine:3.20",
                "sh",
                "-c",
                "rm -rf /c/*; mkdir -p /c; chmod 777 /c",
            ],
            timeout=25,
        )
        cell(
            "v100s.ptx.das9201.cold",
            "xerxes",
            "device=0",
            "mcscram:cuda12.8-sm70",
            "das9201",
            "/app/input/das9201.xml",
            extra=extra_ptx,
            env_add=ptx_env,
            wait_v100=True,
        )
        for name, path in models:
            cell(
                f"v100s.ptx.{name}.warm",
                "xerxes",
                "device=0",
                "mcscram:cuda12.8-sm70",
                name,
                path,
                extra=extra_ptx,
                env_add=ptx_env,
                wait_v100=True,
            )
            cell(
                f"v100s.ptx.{name}.nsys",
                "xerxes",
                "device=0",
                "mcscram:cuda12.8-sm70-prof",
                name,
                path,
                nsys=True,
                extra=extra_ptx,
                env_add=ptx_env,
                wait_v100=True,
            )
        for name, path in [
            ("das9201", "/app/input/das9201.xml"),
            ("das9701", "/app/input/das9701.xml"),
            ("chinese", "/app/input/chinese.xml"),
        ]:
            cell(
                f"v100s.oldgen.{name}",
                "xerxes",
                "device=0",
                "mcscram-benchmarks-server:latest",
                name,
                path,
                extra=["--entrypoint", "/usr/local/bin/scram-cli"],
                wait_v100=True,
            )
            if name != "chinese":
                cell(
                    f"v100s.oldgen.{name}.nsys",
                    "xerxes",
                    "device=0",
                    "mcscram-benchmarks-server:latest",
                    name,
                    path,
                    nsys=True,
                    nsys_bin=True,
                    extra=["--entrypoint", "/usr/local/bin/scram-cli"],
                    wait_v100=True,
                )

    if which in ("all", "rtx3070"):
        extra_ptx = ["-v", "/tmp/mcscram-ptx-jit-sm86:/cuda-cache"]
        for name, path in models:
            cell(
                f"rtx3070.sass.{name}",
                "xerxes",
                "device=1",
                "mcscram:cuda12.8-sm86",
                name,
                path,
            )
            cell(
                f"rtx3070.sass.{name}.nsys",
                "xerxes",
                "device=1",
                "mcscram:cuda12.8-sm86-prof",
                name,
                path,
                nsys=True,
            )
        cell(
            "rtx3070.sass.chinese",
            "xerxes",
            "device=1",
            "mcscram:cuda12.8-sm86",
            *chinese,
        )
        run(
            [
                "timeout",
                "20",
                "docker",
                "--context",
                "xerxes",
                "run",
                "--rm",
                "-v",
                "/tmp/mcscram-ptx-jit-sm86:/c",
                "alpine:3.20",
                "sh",
                "-c",
                "rm -rf /c/*; mkdir -p /c; chmod 777 /c",
            ],
            timeout=25,
        )
        cell(
            "rtx3070.ptx.das9201.cold",
            "xerxes",
            "device=1",
            "mcscram:cuda12.8-sm86",
            "das9201",
            "/app/input/das9201.xml",
            extra=extra_ptx,
            env_add=ptx_env,
        )
        for name, path in models:
            cell(
                f"rtx3070.ptx.{name}.warm",
                "xerxes",
                "device=1",
                "mcscram:cuda12.8-sm86",
                name,
                path,
                extra=extra_ptx,
                env_add=ptx_env,
            )
            cell(
                f"rtx3070.ptx.{name}.nsys",
                "xerxes",
                "device=1",
                "mcscram:cuda12.8-sm86-prof",
                name,
                path,
                nsys=True,
                extra=extra_ptx,
                env_add=ptx_env,
            )
        for name, path in [
            ("das9201", "/app/input/das9201.xml"),
            ("das9701", "/app/input/das9701.xml"),
            ("chinese", "/app/input/chinese.xml"),
        ]:
            cell(
                f"rtx3070.oldgen.{name}",
                "xerxes",
                "device=1",
                "mcscram-benchmarks-server:latest",
                name,
                path,
                extra=["--entrypoint", "/usr/local/bin/scram-cli"],
            )
            if name != "chinese":
                cell(
                    f"rtx3070.oldgen.{name}.nsys",
                    "xerxes",
                    "device=1",
                    "mcscram-benchmarks-server:latest",
                    name,
                    path,
                    nsys=True,
                    nsys_bin=True,
                    extra=["--entrypoint", "/usr/local/bin/scram-cli"],
                )


if __name__ == "__main__":
    main()
