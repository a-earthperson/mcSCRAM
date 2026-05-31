"""Parse MEF fault-tree XML into the JSON graph consumed by the site canvas."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _node_id(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def aralia_model_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("MCSCRAM_ARALIA_INPUT_DIR")
    if env_root:
        roots.append(Path(env_root))
    # The server image copies `input/*` into `/app/input/`, which flattens the
    # Aralia XML files beside the legacy fixture inputs.
    roots.append(Path("/app/input"))
    roots.append(Path("/app/input/Aralia"))
    repo_aralia = Path(__file__).resolve().parents[4] / "input" / "Aralia"
    if repo_aralia.is_dir():
        roots.append(repo_aralia)
    return roots


def resolve_model_xml_path(model_id: str) -> Path:
    token = model_id.strip().lower().removesuffix(".xml")
    if not token:
        raise ValueError("model_id is required")
    for root in aralia_model_search_roots():
        candidate = root / f"{token}.xml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unknown model_id: {model_id}")


def build_fault_tree_payload(input_files: list[str]) -> dict[str, Any]:
    if not input_files:
        return {"model": None, "top": None, "nodes": [], "edges": []}

    path = Path(input_files[0])
    tree = ET.parse(path)
    root = tree.getroot()
    fault_tree = next((child for child in root.iter() if _tag(child) == "define-fault-tree"), None)
    if fault_tree is None:
        return {"model": path.name, "top": None, "nodes": [], "edges": []}

    model_name = fault_tree.attrib.get("name") or path.stem
    gate_defs = [child for child in fault_tree if _tag(child) == "define-gate"]
    basic_defs = [child for child in fault_tree if _tag(child) == "define-basic-event"]
    top_gate = gate_defs[0].attrib.get("name") if gate_defs else None

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for basic in basic_defs:
        name = basic.attrib.get("name")
        if not name:
            continue
        nodes[_node_id("basic_event", name)] = {
            "id": _node_id("basic_event", name),
            "kind": "basic_event",
            "label": name,
        }

    for gate in gate_defs:
        name = gate.attrib.get("name")
        if not name:
            continue
        gate_id = _node_id("gate", name)
        connective = next((child for child in gate if _tag(child) not in {"label", "description"}), None)
        connective_name = _tag(connective) if connective is not None else "unknown"
        nodes[gate_id] = {
            "id": gate_id,
            "kind": "gate",
            "label": name,
            "connective": connective_name,
            "min": connective.attrib.get("min") if connective is not None else None,
        }
        if connective is None:
            continue
        for child in connective:
            child_tag = _tag(child)
            child_name = child.attrib.get("name")
            if not child_name:
                continue
            if child_tag == "gate":
                child_id = _node_id("gate", child_name)
                nodes.setdefault(child_id, {
                    "id": child_id,
                    "kind": "gate",
                    "label": child_name,
                    "connective": "unknown",
                })
            elif child_tag == "basic-event":
                child_id = _node_id("basic_event", child_name)
                nodes.setdefault(child_id, {
                    "id": child_id,
                    "kind": "basic_event",
                    "label": child_name,
                })
            else:
                continue
            edges.append({"source": gate_id, "target": child_id})

    return {
        "model": model_name,
        "source": str(path),
        "top": _node_id("gate", top_gate) if top_gate else None,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
