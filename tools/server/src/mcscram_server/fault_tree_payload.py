"""Parse MEF fault-tree XML into the JSON graph consumed by the site canvas."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DAG_NODE_TYPES: tuple[str, ...] = (
    "basic_event",
    "unknown_event",
    "and_gate",
    "nand_gate",
    "or_gate",
    "nor_gate",
    "xor_gate",
    "xnor_gate",
    "atleast_gate",
    "not_gate",
)

NODE_TYPE_BY_CONNECTIVE = {
    "and": 2,
    "nand": 3,
    "or": 4,
    "nor": 5,
    "xor": 6,
    "xnor": 7,
    "atleast": 8,
    "not": 9,
}


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


def build_dag_payload(input_files: list[str]) -> dict[str, Any]:
    """Build a render-ready, columnar DAG payload for the WebGL2 viewer.

    Shape contract:
      - `nodes.x[i]`, `nodes.y[i]`, and `nodes.type[i]` describe vertex `i`.
      - `edges.source[j]` and `edges.target[j]` are integer node indices.
      - coordinates are scale-agnostic world units; the client only applies a
        uniform affine transform for pan/zoom.
      - `nodeTypes[n]` is a stable semantic type string. Shape, fill, and
        shader styling remain frontend concerns.
    """
    if not input_files:
        return _empty_dag_payload(None)

    path = Path(input_files[0])
    tree = ET.parse(path)
    root = tree.getroot()
    fault_tree = next((child for child in root.iter() if _tag(child) == "define-fault-tree"), None)
    if fault_tree is None:
        return _empty_dag_payload(path.name, source=str(path))

    model_name = fault_tree.attrib.get("name") or path.stem
    graph = _parse_fault_tree_graph(fault_tree)
    node_ids = list(graph["nodes"].keys())
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    depths = _dag_depths(node_ids, graph["children_by_source"], graph["parents_by_target"])

    layers: dict[int, list[str]] = {}
    for node_id in node_ids:
        layer = depths.get(node_id, 0)
        layers.setdefault(layer, []).append(node_id)

    x_coordinates = [0.0] * len(node_ids)
    y_coordinates = [0.0] * len(node_ids)
    node_types = [0] * len(node_ids)
    min_x = min_y = 0.0
    max_x = max_y = 0.0
    seen_coordinate = False

    for layer in sorted(layers):
        ids = layers[layer]
        midpoint = (len(ids) - 1) / 2
        for offset, node_id in enumerate(ids):
            index = index_by_id[node_id]
            x = float(offset - midpoint)
            y = float(layer)
            x_coordinates[index] = x
            y_coordinates[index] = y
            node_types[index] = int(graph["nodes"][node_id]["type"])
            if not seen_coordinate:
                min_x = max_x = x
                min_y = max_y = y
                seen_coordinate = True
            else:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    sources: list[int] = []
    targets: list[int] = []
    for source_id, target_id in graph["edges"]:
        source = index_by_id.get(source_id)
        target = index_by_id.get(target_id)
        if source is None or target is None:
            continue
        sources.append(source)
        targets.append(target)

    return {
        "version": 1,
        "model": model_name,
        "source": str(path),
        "top": index_by_id.get(graph["top"]) if graph["top"] else None,
        "bounds": {
            "minX": min_x,
            "minY": min_y,
            "maxX": max_x,
            "maxY": max_y,
            "width": max(max_x - min_x, 1.0),
            "height": max(max_y - min_y, 1.0),
        },
        "nodeTypes": list(DAG_NODE_TYPES),
        "nodes": {
            "count": len(node_ids),
            "x": x_coordinates,
            "y": y_coordinates,
            "type": node_types,
        },
        "edges": {
            "count": len(sources),
            "source": sources,
            "target": targets,
        },
        "counts": graph["counts"],
    }


def _empty_dag_payload(model: str | None, source: str | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "model": model,
        "source": source,
        "top": None,
        "bounds": {"minX": 0.0, "minY": 0.0, "maxX": 1.0, "maxY": 1.0, "width": 1.0, "height": 1.0},
        "nodeTypes": list(DAG_NODE_TYPES),
        "nodes": {"count": 0, "x": [], "y": [], "type": []},
        "edges": {"count": 0, "source": [], "target": []},
        "counts": {"gate": 0, "basic_event": 0, "unknown_event": 0},
    }


def _parse_fault_tree_graph(fault_tree: ET.Element) -> dict[str, Any]:
    gate_defs = [child for child in fault_tree if _tag(child) == "define-gate"]
    basic_defs = [child for child in fault_tree if _tag(child) == "define-basic-event"]
    top_gate = gate_defs[0].attrib.get("name") if gate_defs else None

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str]] = []
    children_by_source: dict[str, list[str]] = {}
    parents_by_target: dict[str, list[str]] = {}

    for basic in basic_defs:
        name = basic.attrib.get("name")
        if name:
            nodes[_node_id("basic_event", name)] = {"kind": "basic_event", "type": 0}

    for gate in gate_defs:
        name = gate.attrib.get("name")
        if not name:
            continue
        gate_id = _node_id("gate", name)
        connective = next((child for child in gate if _tag(child) not in {"label", "description"}), None)
        connective_name = _tag(connective) if connective is not None else "unknown"
        nodes[gate_id] = {
            "kind": "gate",
            "type": NODE_TYPE_BY_CONNECTIVE.get(connective_name, 1),
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
                nodes.setdefault(child_id, {"kind": "gate", "type": 1})
            elif child_tag == "basic-event":
                child_id = _node_id("basic_event", child_name)
                nodes.setdefault(child_id, {"kind": "unknown_event", "type": 1})
            else:
                continue
            edges.append((gate_id, child_id))
            children_by_source.setdefault(gate_id, []).append(child_id)
            parents_by_target.setdefault(child_id, []).append(gate_id)

    counts = {
        "gate": sum(1 for node in nodes.values() if node["kind"] == "gate"),
        "basic_event": sum(1 for node in nodes.values() if node["kind"] == "basic_event"),
        "unknown_event": sum(1 for node in nodes.values() if node["kind"] == "unknown_event"),
    }
    return {
        "top": _node_id("gate", top_gate) if top_gate else None,
        "nodes": nodes,
        "edges": edges,
        "children_by_source": children_by_source,
        "parents_by_target": parents_by_target,
        "counts": counts,
    }


def _dag_depths(
    node_ids: list[str],
    children_by_source: dict[str, list[str]],
    parents_by_target: dict[str, list[str]],
) -> dict[str, int]:
    indegree = {node_id: len(parents_by_target.get(node_id, [])) for node_id in node_ids}
    depth = {node_id: 0 for node_id in node_ids}
    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]

    cursor = 0
    while cursor < len(queue):
        node_id = queue[cursor]
        cursor += 1
        parent_depth = depth[node_id]
        for child_id in children_by_source.get(node_id, []):
            if child_id not in indegree:
                continue
            depth[child_id] = max(depth[child_id], parent_depth + 1)
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)

    return depth