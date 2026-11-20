import type { InkPalette } from "../../../../../components/plot/plotTokens";
import { withAlpha } from "../../../../../components/plot/plotTokens";
import type {
  ExecutionPartitionDescription,
  ExecutionPlanPayload,
} from "../../../../../api/faultTreeApi";

/**
 * Pure layout + colour model for the DirectEval execution plan.
 *
 * The native plan is *partition-centric*: every kernel partition carries a
 * device-shaped launch geometry (logical / local / global 3D ranges) and a
 * layer index that encodes the topological execution order. This module folds
 * that into a renderable "execution landscape":
 *
 *   - X (depth) = execution layer       — the hierarchical ordering
 *   - Z (lane)  = partition within layer — siblings spread sideways
 *   - Y (rise)  = log10(launched work-items) — how much work the device runs
 *
 * Nothing here touches the DOM or MathBox, so it is trivially testable and is
 * shared between the WebGL scene and the 2D fallback map.
 */

/** Stable colour buckets — one pigment per kernel family. */
export type ExecKindKey =
  | "basic_event"
  | "and"
  | "or"
  | "atleast"
  | "xor"
  | "not"
  | "nand"
  | "nor"
  | "null"
  | "tally"
  | "gate";

export type SceneTower = {
  id: number;
  layer: number;
  lane: number;
  /** Normalised world coordinates (x,z in [0,1]/[-0.5,0.5]; height in [0,1]). */
  x: number;
  z: number;
  height: number;
  colorKey: ExecKindKey;
  partition: ExecutionPartitionDescription;
};

export type SceneEdge = {
  source: number;
  target: number;
};

export type ExecutionScene = {
  towers: SceneTower[];
  towerById: Map<number, SceneTower>;
  edges: SceneEdge[];
  layerCount: number;
  maxLanes: number;
  /** log10(max work-items) used to normalise tower height. */
  maxLogWork: number;
  /** Distinct colour buckets present, in legend order. */
  kinds: ExecKindKey[];
};

const KIND_ORDER: ExecKindKey[] = [
  "basic_event",
  "and",
  "nand",
  "or",
  "nor",
  "xor",
  "atleast",
  "not",
  "null",
  "gate",
  "tally",
];

const KIND_LABELS: Record<ExecKindKey, string> = {
  basic_event: "basic event",
  and: "AND",
  nand: "NAND",
  or: "OR",
  nor: "NOR",
  xor: "XOR",
  atleast: "VOT",
  not: "NOT",
  null: "NULL",
  gate: "gate",
  tally: "tally",
};

export function kindLabel(key: ExecKindKey): string {
  return KIND_LABELS[key] ?? key;
}

/** Resolve a partition to its stable colour bucket. */
export function partitionKind(partition: {
  kind: string;
  connective: string;
}): ExecKindKey {
  if (partition.kind === "basic_event") return "basic_event";
  if (partition.kind === "tally") return "tally";
  const connective = partition.connective as ExecKindKey;
  if (KIND_ORDER.includes(connective)) return connective;
  return "gate";
}

/**
 * Map a colour bucket to the archive pigment register. Towers reuse the same
 * botanical hues the WebGL DAG plate uses for gates, so the three pipeline
 * canvases read as one family.
 */
export function kindColor(key: ExecKindKey, palette: InkPalette): string {
  switch (key) {
    case "basic_event":
      return palette.ochre;
    case "and":
    case "nand":
      return palette.indigo;
    case "or":
    case "nor":
      return palette.madder;
    case "xor":
      return palette.sage;
    case "atleast":
      return palette.indigoDeep;
    case "not":
      return palette.rose;
    case "null":
      return palette.inkFaint;
    case "tally":
      return palette.ink;
    case "gate":
    default:
      return palette.paynes;
  }
}

function safeLog10(value: number): number {
  return value > 0 ? Math.log10(value + 1) : 0;
}

/** Fold the partition list into normalised tower + edge geometry. */
export function buildExecutionScene(plan: ExecutionPlanPayload): ExecutionScene {
  const partitions = [...(plan.partitions ?? [])].sort((a, b) => {
    if (a.layer !== b.layer) return a.layer - b.layer;
    return a.id - b.id;
  });

  const layers = Array.from(new Set(partitions.map((p) => p.layer))).sort((a, b) => a - b);
  const layerIndex = new Map<number, number>(layers.map((layer, index) => [layer, index]));
  const layerCount = Math.max(1, layers.length);

  const laneCursor = new Map<number, number>();
  const lanesPerLayer = new Map<number, number>();
  for (const partition of partitions) {
    lanesPerLayer.set(partition.layer, (lanesPerLayer.get(partition.layer) ?? 0) + 1);
  }
  const maxLanes = Math.max(1, ...Array.from(lanesPerLayer.values()));

  const maxLogWork = Math.max(
    1e-6,
    ...partitions.map((p) => safeLog10(p.work_item_count)),
  );

  const laneStep = maxLanes > 1 ? 1 / maxLanes : 0;

  const towers: SceneTower[] = partitions.map((partition) => {
    const li = layerIndex.get(partition.layer) ?? 0;
    const lane = laneCursor.get(partition.layer) ?? 0;
    laneCursor.set(partition.layer, lane + 1);
    const lanesHere = lanesPerLayer.get(partition.layer) ?? 1;

    const x = layerCount > 1 ? li / (layerCount - 1) : 0.5;
    const z = (lane - (lanesHere - 1) / 2) * laneStep;
    const height = safeLog10(partition.work_item_count) / maxLogWork;

    return {
      id: partition.id,
      layer: partition.layer,
      lane,
      x,
      z,
      height: Math.max(height, 0.012),
      colorKey: partitionKind(partition),
      partition,
    };
  });

  const towerById = new Map(towers.map((tower) => [tower.id, tower]));

  const edges: SceneEdge[] = (plan.dependency_edges ?? []).filter(
    (edge) => towerById.has(edge.source_partition) && towerById.has(edge.target_partition),
  ).map((edge) => ({ source: edge.source_partition, target: edge.target_partition }));

  const presentKinds = new Set(towers.map((tower) => tower.colorKey));
  const kinds = KIND_ORDER.filter((key) => presentKinds.has(key));

  return { towers, towerById, edges, layerCount, maxLanes, maxLogWork, kinds };
}

/** Translucent fill paired with each pigment for the tower body. */
export function towerFill(key: ExecKindKey, palette: InkPalette): string {
  return withAlpha(kindColor(key, palette), 0.32);
}

// ---------------------------------------------------------------------------
//  Formatting helpers (execution-plan specific)
// ---------------------------------------------------------------------------

/** Compact magnitude formatting for launch counts (22,492,160 -> "22.5M"). */
export function compactCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs < 1000) return String(Math.round(value));
  const units: Array<[number, string]> = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [scale, suffix] of units) {
    if (abs >= scale) {
      const scaled = value / scale;
      const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
      return `${scaled.toFixed(digits)}${suffix}`;
    }
  }
  return String(Math.round(value));
}

export type Range3 = { x: number; y: number; z: number; product?: number };

/** Render a 3D launch range as "x x y x z" with thin spaces. */
export function formatRange3(range: Range3 | undefined): string {
  if (!range) return "n/a";
  return `${compactCount(range.x)} x ${compactCount(range.y)} x ${compactCount(range.z)}`;
}

/** Fraction of launched work-items that are masked-off padding. */
export function overprovisionRatio(partition: ExecutionPartitionDescription): number {
  if (!partition.work_item_count) return 0;
  return partition.overprovisioned_work_items / partition.work_item_count;
}
