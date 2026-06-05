/**
 * Top-level SYCL execution model.
 *
 * Folds a hydrated `ExecutionPlanPayload` into a device + an ordered set of
 * `KernelModel`s and an explicit execution schedule. The schedule is the
 * topological layering the native planner already computes: kernels in the same
 * `layer` are mutually independent and "run together"; layers run in order and
 * edges only ever point to earlier layers. This is the pure description the
 * renderer animates — it never decides ordering itself.
 */
import type {
  ExecutionPartitionDescription,
  ExecutionPlanPayload,
} from "../../../../api/faultTreeApi";
import { DEFAULT_DEVICE, type DeviceModel, deviceFromDescription } from "./device";
import { type KernelModel, type TileShape, buildKernelModel } from "./kernel";

/** One stage of the execution schedule: a layer and the kernels it launches. */
export type ExecutionStep = {
  index: number;
  layer: number;
  kernelIds: number[];
};

export type ExecutionModel = {
  device: DeviceModel;
  bitsPerBitpack: number;
  kernels: KernelModel[];
  byId: Map<number, KernelModel>;
  /** Distinct layers in ascending (execution) order. */
  layers: number[];
  /** Execution schedule, one step per layer. */
  steps: ExecutionStep[];
  /** Kernel the scene opens on: the first basic-event kernel, else the first. */
  defaultKernelId: number;
};

/**
 * The maximum single work-group footprint (in bitpack tiles) across all
 * kernels. The world-fixed bitpack field is sized to this so any layer's
 * work-group overlay fits within it; smaller work-groups occupy a sub-region.
 */
export function maxWorkGroupShape(model: ExecutionModel): TileShape {
  let rows = 1;
  let cols = 1;
  for (const kernel of model.kernels) {
    rows = Math.max(rows, kernel.workGroupShape.rows);
    cols = Math.max(cols, kernel.workGroupShape.cols);
  }
  return { rows, cols };
}

/** Representative kernel for a step: first basic-event, else first kernel. */
export function representativeKernelId(model: ExecutionModel, step: ExecutionStep): number {
  const basic = step.kernelIds.find((id) => model.byId.get(id)?.kind === "basic_event");
  return basic ?? step.kernelIds[0] ?? model.defaultKernelId;
}

/** Step index that launches a given kernel (by its layer). */
export function stepIndexOfKernel(model: ExecutionModel, kernelId: number): number {
  const kernel = model.byId.get(kernelId);
  if (!kernel) return 0;
  const index = model.steps.findIndex((step) => step.layer === kernel.layer);
  return index >= 0 ? index : 0;
}

function squareBits(bits: number): number {
  const side = Math.sqrt(bits);
  return Number.isInteger(side) && side > 0 ? bits : 64;
}

function buildSteps(kernels: KernelModel[]): { layers: number[]; steps: ExecutionStep[] } {
  const byLayer = new Map<number, number[]>();
  for (const kernel of kernels) {
    const bucket = byLayer.get(kernel.layer);
    if (bucket) bucket.push(kernel.id);
    else byLayer.set(kernel.layer, [kernel.id]);
  }
  const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
  const steps = layers.map((layer, index) => ({
    index,
    layer,
    kernelIds: (byLayer.get(layer) ?? []).slice().sort((a, b) => a - b),
  }));
  return { layers, steps };
}

function pickDefaultKernel(kernels: KernelModel[]): number {
  if (kernels.length === 0) return -1;
  const basic = kernels.find((k) => k.kind === "basic_event");
  if (basic) return basic.id;
  const ordered = kernels
    .slice()
    .sort((a, b) => (a.layer !== b.layer ? a.layer - b.layer : a.id - b.id));
  return ordered[0].id;
}

function assembleModel(
  device: DeviceModel,
  bitsPerBitpack: number,
  partitions: ExecutionPartitionDescription[],
): ExecutionModel {
  const bits = squareBits(bitsPerBitpack);
  const kernels = partitions.map((partition) => buildKernelModel(partition, device, bits));
  const byId = new Map(kernels.map((kernel) => [kernel.id, kernel]));
  const { layers, steps } = buildSteps(kernels);
  return {
    device,
    bitsPerBitpack: bits,
    kernels,
    byId,
    layers,
    steps,
    defaultKernelId: pickDefaultKernel(kernels),
  };
}

export function buildExecutionModel(plan: ExecutionPlanPayload): ExecutionModel {
  const device = plan.device ? deviceFromDescription(plan.device) : DEFAULT_DEVICE;
  return assembleModel(device, plan.bits_per_bitpack ?? 64, plan.partitions ?? []);
}

// ---------------------------------------------------------------------------
//  Representative fallback — the `nus9601` basic-event partition from the note.
// ---------------------------------------------------------------------------

const DEFAULT_BASIC_EVENT_PARTITION: ExecutionPartitionDescription = {
  id: 0,
  layer: 0,
  kind: "basic_event",
  connective: "sample",
  operation: "sample",
  node_indices: [],
  dependency_partition_ids: [],
  node_count: 1567,
  logical_range: { x: 1567, y: 1, z: 184512, product: 289130304 },
  local_range: { x: 64, y: 1, z: 16, product: 1024 },
  global_range: { x: 1600, y: 1, z: 184512, product: 295219200 },
  work_group_count: 288300,
  work_item_count: 295219200,
  logical_work_item_count: 289130304,
  overprovisioned_work_items: 6088896,
  bitpacks_per_node: 184512,
  bits_per_node: 11808768,
  bytes_per_node: 1476096,
  resident_bytes: 2313042432,
  min_fan_in: 0,
  max_fan_in: 0,
  total_fan_in: 0,
  positive_input_count: 0,
  negated_input_count: 0,
};

/**
 * A geometrically-correct stand-in used before any plan is hydrated, so the
 * MathBox scene renders the documented `nus9601` basic-event work-group window
 * instead of an arbitrary placeholder.
 */
export function defaultExecutionModel(): ExecutionModel {
  return assembleModel(DEFAULT_DEVICE, 64, [DEFAULT_BASIC_EVENT_PARTITION]);
}

/** Build from a plan when present, else the representative default. */
export function executionModelFromPlan(plan: ExecutionPlanPayload | null | undefined): ExecutionModel {
  if (plan && Array.isArray(plan.partitions) && plan.partitions.length > 0) {
    return buildExecutionModel(plan);
  }
  return defaultExecutionModel();
}
