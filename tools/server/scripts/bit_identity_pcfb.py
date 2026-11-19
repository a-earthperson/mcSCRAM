#!/usr/bin/env python3
"""Bit-identity: legacy NativeSampler vs pcfb Session, plus one DAG payload pair.

Run inside a container that has the fuzzybits wheel and this server package.
Every Monte Carlo loop is expected to sit under `timeout 60` of the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _xml(name: str) -> Path:
    candidates = [
        Path("/models") / f"{name}.xml",
        REPO / "docker" / "inputs" / f"{name}.xml",
        REPO / "input" / "Aralia" / f"{name}.xml",
        REPO / "tests" / "python" / "data" / f"{name}.xml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{name}.xml not found; looked in {candidates}")


def _options(watch: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "compilation_level": 2,
        "no_kn": True,
        "watch": watch,
    }
    raw_tpp = os.getenv("MCSCRAM_REQUESTED_TRIALS_PER_PASS", "")
    if raw_tpp:
        options["requested_trials_per_pass"] = float(raw_tpp)
    return options


def _index_names(graph: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for node in graph.get("nodes") or []:
        index = int(node["index"])
        label = str(node.get("id") or node.get("name") or index)
        names[index] = label
    return names


def _named_stream(sampler: Any, names: dict[int, str], passes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for _ in range(passes):
            obs = sampler.step()
            keyed = {}
            for index, counters in obs.nodes.items():
                key = names.get(int(index), str(index))
                keyed[key] = {
                    "index": int(index),
                    "num_one_bits": counters.num_one_bits,
                    "total_bits": counters.total_bits,
                }
            target_key = names.get(int(obs.target_index), str(obs.target_index))
            rows.append(
                {
                    "pass_index": obs.pass_index,
                    "target_index": obs.target_index,
                    "target_name": target_key,
                    "trials_per_pass": obs.trials_per_pass,
                    "nodes": keyed,
                }
            )
    finally:
        sampler.close()
    return rows


def _payload_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        diffs: list[str] = []
        keys = sorted(set(a) | set(b))
        for key in keys:
            here = f"{path}.{key}" if path else str(key)
            if key not in a:
                diffs.append(f"{here}: missing on legacy")
            elif key not in b:
                diffs.append(f"{here}: missing on pcfb")
            else:
                diffs.extend(_payload_diff(a[key], b[key], here))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        diffs = []
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            diffs.extend(_payload_diff(left, right, f"{path}[{i}]"))
        return diffs
    if a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=16)
    parser.add_argument("--models", nargs="+", default=["chinese", "das9201"])
    parser.add_argument("--watch", default="g1")
    parser.add_argument("--dag-model", default="chinese")
    parser.add_argument("--skip-dag", action="store_true")
    parser.add_argument("--dag-only", action="store_true")
    args = parser.parse_args()

    backend = os.getenv("FUZZYBITS_BACKEND", "omp")
    import fuzzybits

    fuzzybits.init(backend=backend)
    print("backend", fuzzybits.active_backend(), "variant", fuzzybits.active_variant())
    print("devices", fuzzybits.list_devices())

    sys.path.insert(0, str(REPO / "tools" / "server" / "src"))
    os.environ.setdefault("FUZZYBITS_SERVER_CACHE_DIR", "/tmp/fuzzybits-server-pcfb-cache")

    from fuzzybits_server.convergence.sampling import NativeSampler, make_sampler
    from fuzzybits_server.fault_tree_payload import build_dag_payload

    watch = [part for part in args.watch.split(",") if part]
    options = _options(watch)
    print("options", json.dumps(options, sort_keys=True))
    failed = 0

    if args.dag_only:
        args.models = []
        args.skip_dag = False

    for model in args.models:
        xml = _xml(model)
        files = [str(xml)]
        os.environ["FUZZYBITS_SERVER_ENGINE"] = "pcfb"
        pcfb = make_sampler(files, options)
        pcfb_names = _index_names(pcfb._session.graph())
        os.environ["FUZZYBITS_SERVER_ENGINE"] = "legacy"
        legacy = NativeSampler(files, options)
        legacy_names = _index_names(legacy._pdag.to_graph())
        print(
            f"{model}: trials_per_pass pcfb={pcfb.shape.trials_per_pass} "
            f"legacy={legacy.shape.trials_per_pass} "
            f"pcfb_nodes={pcfb.node_indices} legacy_nodes={legacy.node_indices}"
        )
        if pcfb.shape.trials_per_pass != legacy.shape.trials_per_pass:
            print(f"FAIL {model}: trials_per_pass mismatch")
            failed += 1
            pcfb.close()
            legacy.close()
            continue
        pcfb_rows = _named_stream(pcfb, pcfb_names, args.passes)
        legacy_rows = _named_stream(legacy, legacy_names, args.passes)
        print(f"{model}: passes={len(pcfb_rows)}")
        print("pass\tname\tpcfb_ones/total@idx\tlegacy_ones/total@idx\tmatch")
        model_ok = True
        for left, right in zip(pcfb_rows, legacy_rows, strict=True):
            keys = sorted(set(left["nodes"]) | set(right["nodes"]))
            for key in keys:
                if key not in left["nodes"] or key not in right["nodes"]:
                    model_ok = False
                    print(f"{left['pass_index']}\t{key}\tMISSING")
                    continue
                l_n = left["nodes"][key]
                r_n = right["nodes"][key]
                match = (l_n["num_one_bits"], l_n["total_bits"]) == (
                    r_n["num_one_bits"],
                    r_n["total_bits"],
                )
                if not match:
                    model_ok = False
                print(
                    f"{left['pass_index']}\t{key}\t"
                    f"{l_n['num_one_bits']}/{l_n['total_bits']}@{l_n['index']}\t"
                    f"{r_n['num_one_bits']}/{r_n['total_bits']}@{r_n['index']}\t{match}"
                )
        if not model_ok:
            print(f"FAIL {model}: per-pass named node streams differ")
            failed += 1
        else:
            print(f"OK {model}: {args.passes} passes bit-identical")

    if args.skip_dag:
        return 1 if failed else 0

    dag_xml = _xml(args.dag_model)
    os.environ["FUZZYBITS_SERVER_ENGINE"] = "legacy"
    legacy_dag = build_dag_payload([str(dag_xml)], options)
    os.environ["FUZZYBITS_SERVER_ENGINE"] = "pcfb"
    pcfb_dag = build_dag_payload([str(dag_xml)], options)
    diffs = _payload_diff(legacy_dag, pcfb_dag)
    print(f"/model/dag {args.dag_model}: field diffs={len(diffs)}")
    for line in diffs[:40]:
        print("  ", line)
    if len(diffs) > 40:
        print(f"  … {len(diffs) - 40} more")
    if diffs:
        print("DAG payload differences listed above")
    else:
        print("OK /model/dag: identical")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
