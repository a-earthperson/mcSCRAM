"""Make generic-pwr-model OpenPSA XML self-contained for MEF initialize.

The SAPHIRE→OpenPSA exporter leaves some gate / basic-event / functional-event
definitions in sibling files (``data/saphsolve/*.JSInp``,
``data/openpsa/be_name_label_and_value_V1.2.csv``). This shim rewrites a copy
of one converted XML: it does **not** modify the submodule. The engine loader
stays strict.

Naming (OpenPSA README): ``BE`` / ``G`` / ``S`` / ``INIT`` / ``FE`` / ``FT``
prefixes; ``BE0`` is the single-child placeholder.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

SHIM_VERSION = 1
SHIM_ID = "generic_pwr"
_KN_RE = re.compile(r"^(\d+)/(\d+)$")
_CONVERTED_RE = re.compile(r"^converted_(.+)\.xml$")
_FE_TYPO_RE = re.compile(r"^F(\d+)$")


@dataclass(frozen=True)
class PatchResult:
    path: Path
    original: Path
    shim: str = SHIM_ID
    version: int = SHIM_VERSION
    injected_gates: tuple[str, ...] = ()
    injected_events: tuple[str, ...] = ()
    rewritten_functional_events: tuple[tuple[str, str], ...] = ()
    sources: dict[str, str] = field(default_factory=dict)
    unchanged: bool = False


def matching_jsinp(xml_path: Path) -> Path:
    """``converted_EQK-BIN4_et_Grp-5_….xml`` → ``saphsolve/EQK-BIN4_et_Grp-5_….JSInp``."""
    match = _CONVERTED_RE.match(xml_path.name)
    if not match:
        raise FileNotFoundError(f"not a converted OpenPSA XML: {xml_path.name}")
    stem = match.group(1)
    candidate = xml_path.parent.parent / "saphsolve" / f"{stem}.JSInp"
    if not candidate.is_file():
        raise FileNotFoundError(f"no SAPHSOLVE JSInp for {xml_path.name}: {candidate}")
    return candidate


def be_csv_path(xml_path: Path) -> Path:
    csv = xml_path.parent / "be_name_label_and_value_V1.2.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"missing basic-event CSV: {csv}")
    return csv


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsinp(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["saphiresolveinput"]


def _index_jsinp(jsinp: dict[str, Any]) -> tuple[dict[int, dict[int, dict]], dict[int, dict]]:
    """Return (ftid → gateid → gate, event_id → event)."""
    by_ft: dict[int, dict[int, dict]] = {}
    for tree in jsinp.get("faulttreelist") or []:
        header = tree.get("ftheader") or {}
        ftid = int(header["ftid"])
        gates: dict[int, dict] = {}
        for gate in tree.get("gatelist") or []:
            gates[int(gate["gateid"])] = gate
        by_ft[ftid] = gates
    events: dict[int, dict] = {}
    for event in jsinp.get("eventlist") or []:
        events[int(event["id"])] = event
    return by_ft, events


def _load_be_csv(path: Path) -> dict[str, tuple[str, str]]:
    """BE name → (label, value-string). First row wins."""
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",", 3)
        if len(parts) < 4:
            continue
        _, name, label, value = parts
        out.setdefault(name.strip(), (label.strip(), value.strip()))
    return out


def _ft_id(name: str) -> int | None:
    if name.startswith("FT") and name[2:].isdigit():
        return int(name[2:])
    return None


def _gate_id(name: str) -> int | None:
    if name.startswith("G") and name[1:].isdigit():
        return int(name[1:])
    return None


def _connective(gate: dict) -> tuple[str, dict[str, str]]:
    raw = str(gate.get("gatetype") or "or").lower()
    kn = _KN_RE.match(raw)
    if kn:
        return "atleast", {"min": kn.group(1)}
    if raw in {"or", "and", "xor", "nand", "nor", "not"}:
        return raw, {}
    return "or", {}


def _append_inputs(parent: ET.Element, gate: dict) -> None:
    for gid in gate.get("gateinput") or []:
        ET.SubElement(parent, "gate", {"name": f"G{int(gid)}"})
    for eid in gate.get("eventinput") or []:
        ET.SubElement(parent, "basic-event", {"name": f"BE{int(eid)}"})
    if len(list(parent)) < 2:
        ET.SubElement(parent, "basic-event", {"name": "BE0"})


def _define_gate(gate: dict) -> ET.Element:
    gid = int(gate["gateid"])
    el = ET.Element("define-gate", {"name": f"G{gid}", "role": "private"})
    tag, attrs = _connective(gate)
    connective = ET.SubElement(el, tag, attrs)
    _append_inputs(connective, gate)
    return el


def _define_basic_event(
    eid: int,
    events: dict[int, dict],
    csv: dict[str, tuple[str, str]],
) -> ET.Element:
    name = f"BE{eid}"
    row = events.get(eid) or {}
    label = str(row.get("name") or csv.get(name, ("", ""))[0] or name)
    value = row.get("value")
    if value is None:
        value = csv.get(name, ("", "0.0"))[1] or "0.0"
    el = ET.Element("define-basic-event", {"name": name})
    lab = ET.SubElement(el, "label")
    lab.text = label
    ET.SubElement(el, "float", {"value": f"{float(value):.6E}"})
    return el


def _rewrite_functional_events(root: ET.Element) -> list[tuple[str, str]]:
    defined = {
        el.attrib["name"]
        for el in root.iter("define-functional-event")
        if "name" in el.attrib
    }
    rewritten: list[tuple[str, str]] = []
    for fork in root.iter("fork"):
        raw = fork.attrib.get("functional-event")
        if not raw or raw in defined:
            continue
        match = _FE_TYPO_RE.match(raw)
        if not match:
            continue
        fixed = "FE" + match.group(1)
        if fixed in defined:
            fork.attrib["functional-event"] = fixed
            rewritten.append((raw, fixed))
    return rewritten


def _cache_paths(cache_dir: Path, xml_path: Path, cache_key: str) -> tuple[Path, Path]:
    stem = xml_path.stem
    return (
        cache_dir / f"{stem}.{cache_key}.xml",
        cache_dir / f"{stem}.{cache_key}.json",
    )


def patch_generic_pwr(
    xml_path: Path | str,
    *,
    cache_dir: Path | None = None,
    jsinp_path: Path | None = None,
    csv_path: Path | None = None,
) -> PatchResult:
    """Rewrite ``xml_path`` into a self-contained MEF file. Original is unchanged."""
    xml_path = Path(xml_path)
    jsinp_path = Path(jsinp_path) if jsinp_path else matching_jsinp(xml_path)
    csv_path = Path(csv_path) if csv_path else be_csv_path(xml_path)
    sources = {
        "xml": sha256_file(xml_path),
        "jsinp": sha256_file(jsinp_path),
        "csv": sha256_file(csv_path),
        "shim": f"{SHIM_ID}:{SHIM_VERSION}",
    }
    cache_key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir) if cache_dir is not None else Path("/tmp/fuzzybits-pwr-compat")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached, meta_path = _cache_paths(cache_dir, xml_path, cache_key)
    if cached.is_file() and meta_path.is_file():
        extra = json.loads(meta_path.read_text(encoding="utf-8"))
        return PatchResult(
            path=cached,
            original=xml_path,
            injected_gates=tuple(extra.get("injected_gates") or ()),
            injected_events=tuple(extra.get("injected_events") or ()),
            rewritten_functional_events=tuple(
                tuple(pair) for pair in extra.get("rewritten_functional_events") or ()
            ),
            sources=sources,
            unchanged=bool(extra.get("unchanged")),
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()
    jsinp = _load_jsinp(jsinp_path)
    gates_by_ft, events = _index_jsinp(jsinp)
    csv = _load_be_csv(csv_path)
    ft_root_gate: dict[int, int] = {}
    for tree_js in jsinp.get("faulttreelist") or []:
        header = tree_js.get("ftheader") or {}
        ft_root_gate[int(header["ftid"])] = int(header["gtid"])

    rewritten_fe = _rewrite_functional_events(root)

    defined_be = {
        el.attrib["name"] for el in root.iter("define-basic-event") if "name" in el.attrib
    }
    injected_gates: list[str] = []
    injected_events: list[str] = []

    def lookup_gate(ftid: int | None, gid: int) -> dict | None:
        if ftid is not None and gid in gates_by_ft.get(ftid, {}):
            return gates_by_ft[ftid][gid]
        for table in gates_by_ft.values():
            if gid in table:
                return table[gid]
        return None

    def ensure_event(eid: int) -> None:
        name = f"BE{eid}"
        if name in defined_be:
            return
        model_data = root.find("model-data")
        if model_data is None:
            model_data = ET.SubElement(root, "model-data")
        model_data.append(_define_basic_event(eid, events, csv))
        defined_be.add(name)
        injected_events.append(name)

    def ensure_gate(ft_el: ET.Element, ftid: int | None, gid: int, stack: set[int]) -> None:
        name = f"G{gid}"
        local = {el.attrib["name"] for el in ft_el.findall("define-gate") if "name" in el.attrib}
        if name in local:
            return
        if ftid is not None and ft_root_gate.get(ftid) == gid:
            # ftheader.gtid is the FT top (FT{n}.TOP); do not invent G{gtid}.
            return
        if gid in stack:
            return
        spec = lookup_gate(ftid, gid)
        if spec is None:
            return
        stack.add(gid)
        for child in spec.get("gateinput") or []:
            ensure_gate(ft_el, ftid, int(child), stack)
        for eid in spec.get("eventinput") or []:
            ensure_event(int(eid))
        ft_el.append(_define_gate(spec))
        injected_gates.append(f"{ft_el.attrib.get('name', '?')}.{name}")
        stack.remove(gid)

    for ft_el in root.findall("define-fault-tree"):
        ft_name = ft_el.attrib.get("name") or ""
        ftid = _ft_id(ft_name)
        local = {el.attrib["name"] for el in ft_el.findall("define-gate") if "name" in el.attrib}
        refs: list[int] = []
        for ref in ft_el.iter("gate"):
            name = ref.attrib.get("name") or ""
            if "." in name:
                continue
            gid = _gate_id(name)
            if gid is None or name in local:
                continue
            refs.append(gid)
        for gid in dict.fromkeys(refs):
            ensure_gate(ft_el, ftid, gid, set())

    unchanged = not injected_gates and not injected_events and not rewritten_fe
    comment = (
        f" generic_pwr shim v{SHIM_VERSION}; "
        f"xml={sources['xml'][:12]} jsinp={sources['jsinp'][:12]} csv={sources['csv'][:12]} "
    )
    root.insert(0, ET.Comment(comment))
    ET.indent(root, space="    ")
    xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8")
    cached.write_bytes(xml_bytes)
    meta = {
        "injected_gates": injected_gates,
        "injected_events": injected_events,
        "rewritten_functional_events": rewritten_fe,
        "unchanged": unchanged,
        "sources": sources,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return PatchResult(
        path=cached,
        original=xml_path,
        injected_gates=tuple(injected_gates),
        injected_events=tuple(injected_events),
        rewritten_functional_events=tuple(rewritten_fe),
        sources=sources,
        unchanged=unchanged,
    )
