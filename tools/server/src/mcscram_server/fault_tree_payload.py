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
    "null_gate",
    "constant_event",
)
NODE_TYPE_BY_NAME = {name: index for index, name in enumerate(DAG_NODE_TYPES)}

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


def _normalise_model_id_token(model_id: str) -> str:
    token = model_id.strip().lower().removesuffix(".xml")
    if not token:
        raise ValueError("model_id is required")
    if any(part in {"", ".", ".."} for part in token.split("/")):
        raise ValueError(f"invalid model_id: {model_id}")
    return token


def _model_id_for_path(root: Path, path: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().lower()


def resolve_model_xml_path(model_id: str) -> Path:
    token = _normalise_model_id_token(model_id)
    rel_parts = token.split("/")
    for root in aralia_model_search_roots():
        if not root.is_dir():
            continue
        candidate = root.joinpath(*rel_parts).with_suffix(".xml")
        if candidate.is_file():
            return candidate
        for path in root.rglob("*.xml"):
            if path.is_file() and _model_id_for_path(root, path) == token:
                return path
    raise FileNotFoundError(f"Unknown model_id: {model_id}")


def list_aralia_xml_inputs() -> dict[str, Any]:
    """Discover ``*.xml`` under :func:`aralia_model_search_roots` (recursive).

    Each entry's ``modelId`` is the path relative to its search root, without
    the ``.xml`` suffix, lowercased and POSIX-normalised.  The first root in
    search order wins when the same ``modelId`` appears in multiple roots.
    """
    models: list[dict[str, str]] = []
    seen: set[str] = set()
    roots_out: list[str] = []

    for root in aralia_model_search_roots():
        if not root.is_dir():
            continue
        roots_out.append(str(root.resolve()))
        for path in sorted(root.rglob("*.xml")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            model_id = _model_id_for_path(root, path)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                {
                    "modelId": model_id,
                    "relativePath": rel.as_posix(),
                    "root": str(root.resolve()),
                }
            )

    models.sort(key=lambda row: row["modelId"])
    return {
        "roots": roots_out,
        "models": models,
        "modelIds": [row["modelId"] for row in models],
    }


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


def build_dag_payload(input_files: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_compiled_dag_payload(input_files, options)


def build_source_dag_payload(input_files: list[str]) -> dict[str, Any]:
    return _build_source_dag_payload(input_files)


def build_compiled_dag_payload(input_files: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the WebGL DAG payload from the native compiled PDAG."""
    return _build_native_dag_payload(input_files, options, graph_kind="compiled")


def build_kernel_dag_payload(input_files: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the WebGL DAG payload from the DirectEval kernel execution DAG."""
    return _build_native_dag_payload(input_files, options, graph_kind="kernel")


def build_dag_bundle_payload(input_files: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build source, compiled, and kernel DAG payloads with one native compilation."""
    source_payload = _build_source_dag_payload(input_files)
    if not input_files:
        empty = _empty_dag_payload(None)
        return {"source": source_payload, "compiled": empty, "kernel": empty}

    try:
        import mcscram_native  # type: ignore[import-not-found]
        run_compile = mcscram_native.run_compile
    except (ImportError, AttributeError) as exc:
        compiled_payload = dict(source_payload)
        compiled_payload["compiled"] = False
        compiled_payload["compileError"] = str(exc)
        kernel_payload = dict(compiled_payload)
        kernel_payload["graphKind"] = "kernel_dag"
        return {"source": source_payload, "compiled": compiled_payload, "kernel": kernel_payload}

    compiled_pdag = run_compile(input_files, options or {})
    return {
        "source": source_payload,
        "compiled": _native_graph_to_payload(compiled_pdag.to_graph()),
        "kernel": _native_graph_to_payload(compiled_pdag.to_kernel_graph()),
    }


def _build_native_dag_payload(
    input_files: list[str],
    options: dict[str, Any] | None,
    *,
    graph_kind: str,
) -> dict[str, Any]:
    if not input_files:
        return _empty_dag_payload(None)

    try:
        import mcscram_native  # type: ignore[import-not-found]
        run_compile = mcscram_native.run_compile
    except (ImportError, AttributeError) as exc:
        payload = _build_source_dag_payload(input_files)
        payload["compiled"] = False
        payload["compileError"] = str(exc)
        return payload

    compiled_pdag = run_compile(input_files, options or {})
    compiled_graph = compiled_pdag.to_kernel_graph() if graph_kind == "kernel" else compiled_pdag.to_graph()
    return _native_graph_to_payload(compiled_graph)


def _native_graph_to_payload(compiled_graph: dict[str, Any]) -> dict[str, Any]:
    render_graph = _compiled_graph_to_render_graph(compiled_graph)
    payload = _columnar_dag_payload(
        model_name=compiled_graph.get("model"),
        source=compiled_graph.get("source"),
        top=render_graph["top"],
        graph=render_graph,
    )
    payload["compiled"] = True
    payload["target"] = compiled_graph.get("target")
    payload["rootComplement"] = compiled_graph.get("root_complement", False)
    payload["compilation"] = compiled_graph.get("compilation", {})
    payload["graphKind"] = compiled_graph.get("graph_kind", "compiled_pdag")
    if kernel := compiled_graph.get("kernel"):
        payload["kernel"] = kernel
    return payload


def _build_source_dag_payload(input_files: list[str]) -> dict[str, Any]:
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
    return _columnar_dag_payload(model_name=model_name, source=str(path), top=graph["top"], graph=graph)


def _columnar_dag_payload(
    model_name: str | None,
    source: str | None,
    top: str | None,
    graph: dict[str, Any],
) -> dict[str, Any]:
    node_ids = list(graph["nodes"].keys())
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    depths = _dag_depths(node_ids, graph["children_by_source"], graph["parents_by_target"])
    has_kernel_layout = all(
        "layer" in graph["nodes"][node_id] and "partition" in graph["nodes"][node_id]
        for node_id in node_ids
    )

    layers: dict[int, list[str]] = {}
    for node_id in node_ids:
        layer = int(graph["nodes"][node_id]["layer"]) if has_kernel_layout else depths.get(node_id, 0)
        layers.setdefault(layer, []).append(node_id)

    x_coordinates = [0.0] * len(node_ids)
    y_coordinates = [0.0] * len(node_ids)
    node_types = [0] * len(node_ids)
    node_layers = [0] * len(node_ids)
    node_partitions = [0] * len(node_ids)
    min_x = min_y = 0.0
    max_x = max_y = 0.0
    seen_coordinate = False

    for layer in sorted(layers):
        ids = layers[layer]
        positioned = _position_layer_nodes(ids, graph["nodes"]) if has_kernel_layout else [
            (node_id, float(offset - (len(ids) - 1) / 2))
            for offset, node_id in enumerate(ids)
        ]
        for node_id, x in positioned:
            index = index_by_id[node_id]
            node = graph["nodes"][node_id]
            partition = int(node.get("partition", 0))
            y = float(layer)
            x_coordinates[index] = x
            y_coordinates[index] = y
            node_types[index] = int(node["type"])
            node_layers[index] = int(layer)
            node_partitions[index] = partition
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
        source_index = index_by_id.get(source_id)
        target_index = index_by_id.get(target_id)
        if source_index is None or target_index is None:
            continue
        sources.append(source_index)
        targets.append(target_index)

    return {
        "version": 1,
        "model": model_name,
        "source": source,
        "top": index_by_id.get(top) if top else None,
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
            "layer": node_layers,
            "partition": node_partitions,
        },
        "edges": {
            "count": len(sources),
            "source": sources,
            "target": targets,
        },
        "counts": graph["counts"],
    }


def _position_layer_nodes(node_ids: list[str], nodes: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    groups: dict[int, list[str]] = {}
    for node_id in node_ids:
        groups.setdefault(int(nodes[node_id].get("partition", 0)), []).append(node_id)

    ordered_groups = [(partition, groups[partition]) for partition in sorted(groups)]
    widths = [max(1.0, 0.35 * (len(ids) - 1) + 1.0) for _, ids in ordered_groups]
    total_width = sum(widths) + max(0, len(widths) - 1) * 0.8
    cursor = -total_width / 2
    positioned: list[tuple[str, float]] = []

    for (_, ids), width in zip(ordered_groups, widths, strict=True):
        ids.sort(key=lambda node_id: int(node_id))
        center = cursor + width / 2
        midpoint = (len(ids) - 1) / 2
        for offset, node_id in enumerate(ids):
            positioned.append((node_id, center + (offset - midpoint) * 0.35))
        cursor += width + 0.8

    return positioned


def _compiled_graph_to_render_graph(compiled_graph: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str]] = []
    children_by_source: dict[str, list[str]] = {}
    parents_by_target: dict[str, list[str]] = {}
    counts = {"gate": 0, "basic_event": 0, "unknown_event": 0, "constant": 0}

    for node in compiled_graph.get("nodes", []):
        node_id = str(int(node["index"]))
        kind = str(node.get("kind", "unknown_event"))
        node_type = _compiled_node_type(node)
        nodes[node_id] = {
            "kind": kind,
            "type": NODE_TYPE_BY_NAME.get(node_type, NODE_TYPE_BY_NAME["unknown_event"]),
        }
        if "layer" in node:
            nodes[node_id]["layer"] = int(node["layer"])
        if "partition" in node:
            nodes[node_id]["partition"] = int(node["partition"])
        if kind in counts:
            counts[kind] += 1
        else:
            counts["unknown_event"] += 1

    for edge in compiled_graph.get("edges", []):
        source_id = str(int(edge["source"]))
        target_id = str(abs(int(edge["target"])))
        if source_id not in nodes or target_id not in nodes:
            continue
        edges.append((source_id, target_id))
        children_by_source.setdefault(source_id, []).append(target_id)
        parents_by_target.setdefault(target_id, []).append(source_id)

    top = compiled_graph.get("top")
    return {
        "top": str(abs(int(top))) if top is not None else None,
        "nodes": nodes,
        "edges": edges,
        "children_by_source": children_by_source,
        "parents_by_target": parents_by_target,
        "counts": counts,
    }


def _compiled_node_type(node: dict[str, Any]) -> str:
    kind = node.get("kind")
    if kind == "basic_event":
        return "basic_event"
    if kind == "constant":
        return "constant_event"
    if kind != "gate":
        return "unknown_event"
    connective = str(node.get("connective") or "unknown")
    if connective == "atleast":
        return "atleast_gate"
    candidate = f"{connective}_gate"
    return candidate if candidate in NODE_TYPE_BY_NAME else "unknown_event"


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
                # A <basic-event> reference is a basic event by its tag, even
                # when its <define-basic-event> lives under <model-data> rather
                # than inside <define-fault-tree>. Type 0 keeps the circle
                # filled; `unknown_event` (type 1) is reserved for truly
                # unidentifiable references.
                child_id = _node_id("basic_event", child_name)
                nodes.setdefault(child_id, {"kind": "basic_event", "type": 0})
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