/**
 * Kernel model: the bridge from a server execution-plan partition to addressable
 * SYCL units (work-item / sub-group / work-group) and the bitpack tile windows
 * they own.
 *
 * Data-layer convention (see `sycl_range_mapping.md`): the bitpack grid is a 2D
 * lattice of `bitpack_t` tiles. A row is one event/node; a column is one sample
 * (bitpack) in that node's per-pass buffer. One tile == one work-item output ==
 * one `bitpack_t`, drawn as a `bitsPerSide x bitsPerSide` grid of trial bits.
 *
 * The mapping from a 3D launch coordinate `(x, y, z)` to a `(row, col)` cell is
 * uniform across kernel kinds because every kernel writes one bitpack per
 * `(node, sample)`:
 *
 *   row = x                                   // event / node index
 *   col = y * bitpacksPerBatch + z            // flattened sample index
 *
 * with `bitpacksPerBatch = logical_range.z`. For the sample-shaper's
 * `batch_size == 1` plans (`logical_range.y == 1`) a contiguous run of
 * work-items maps to a contiguous run of columns, so sub-groups and work-groups
 * are exact axis-aligned tile rectangles.
 */
import type { ExecutionPartitionDescription } from "../../../../api/faultTreeApi";
import type { DeviceModel } from "./device";
import {
  type Dim3,
  type Span,
  type TileWindow,
  ceilDiv,
  clampSpanToLimit,
  delinearizeLocal,
  fromRange3,
  span,
  volumeOf,
  workGroupGrid,
} from "./geometry";

/** A square bitpack: `bitsPerBitpack` trial bits drawn as `bitsPerSide^2`. */
export type BitpackGeometry = {
  bitsPerBitpack: number;
  bitsPerSide: number;
};

export function bitpackGeometry(bitsPerBitpack: number): BitpackGeometry {
  const bitsPerSide = Math.sqrt(bitsPerBitpack);
  if (!Number.isInteger(bitsPerSide) || bitsPerSide <= 0) {
    throw new Error(`bitpack must be square, got ${bitsPerBitpack} bits`);
  }
  return { bitsPerBitpack, bitsPerSide };
}

/** A 2D extent measured in whole bitpack tiles. */
export type TileShape = { rows: number; cols: number };

/**
 * Maps launch coordinates to bitpack data-layer cells. Constructed per kernel
 * from its logical range so the `bitpacksPerBatch` stride is exact.
 */
export type CellMapping = {
  bitpacksPerBatch: number;
  /** Global launch coordinate -> data-layer `(row, col)` tile cell. */
  cellOf(global: Dim3): { row: number; col: number };
  /**
   * Tile shape covered by a single block of `local` work-items rooted at the
   * origin (work-group footprint). Exact for `batch_size == 1`.
   */
  blockShape(local: Dim3): TileShape;
  /**
   * Tile bounding box of the first `count` work-items of a block in SYCL
   * linear order (z fastest). Used for sub-group / warp footprints.
   */
  linearRunShape(local: Dim3, count: number): TileShape;
};

export function cellMapping(bitpacksPerBatch: number): CellMapping {
  const bpb = Math.max(1, Math.floor(bitpacksPerBatch));
  const cellOf = (global: Dim3) => ({ row: global.x, col: global.y * bpb + global.z });
  return {
    bitpacksPerBatch: bpb,
    cellOf,
    blockShape(local: Dim3): TileShape {
      return { rows: local.x, cols: local.y * local.z };
    },
    linearRunShape(local: Dim3, count: number): TileShape {
      const n = Math.min(Math.max(0, count), volumeOf(local));
      if (n <= 0) return { rows: 0, cols: 0 };
      let rowMin = Infinity;
      let rowMax = -Infinity;
      let colMin = Infinity;
      let colMax = -Infinity;
      for (let linear = 0; linear < n; linear += 1) {
        const lid = delinearizeLocal(local, linear);
        const { row, col } = cellOf(lid);
        rowMin = Math.min(rowMin, row);
        rowMax = Math.max(rowMax, row);
        colMin = Math.min(colMin, col);
        colMax = Math.max(colMax, col);
      }
      return { rows: rowMax - rowMin + 1, cols: colMax - colMin + 1 };
    },
  };
}

export type NdRange = {
  logical: Dim3;
  local: Dim3;
  global: Dim3;
};

/**
 * Fully-derived geometry for one kernel partition. Everything downstream
 * (window selection, overlays, sequencing) reads from this; nothing here knows
 * about rendering.
 */
export type KernelModel = {
  id: number;
  layer: number;
  kind: string;
  connective: string;
  operation: string;
  nodeCount: number;
  dependencyPartitionIds: number[];

  ndRange: NdRange;
  bitpack: BitpackGeometry;
  device: DeviceModel;
  mapping: CellMapping;

  /** Logical (real) tile grid: rows = nodes, cols = samples. */
  dataLayer: TileShape;
  /** Launched (padded) tile grid. */
  launchedLayer: TileShape;
  /** Padding tiles (launched - logical) per axis; masked-off work-items. */
  padding: TileShape;

  workGroupGrid: Dim3;
  workGroupCount: number;
  workGroupShape: TileShape;

  subGroupSize: number;
  subGroupsPerGroup: number;
  subGroupShape: TileShape;

  workItemCount: number;
  logicalWorkItemCount: number;
  overprovisionedWorkItems: number;
};

export function buildKernelModel(
  partition: ExecutionPartitionDescription,
  device: DeviceModel,
  bitsPerBitpack: number,
): KernelModel {
  const logical = fromRange3(partition.logical_range);
  const local = fromRange3(partition.local_range);
  const global = fromRange3(partition.global_range);

  const bitpack = bitpackGeometry(bitsPerBitpack);
  const mapping = cellMapping(logical.z);

  const dataLayer: TileShape = { rows: logical.x, cols: logical.y * logical.z };
  const launchedLayer: TileShape = { rows: global.x, cols: global.y * global.z };
  const padding: TileShape = {
    rows: Math.max(0, launchedLayer.rows - dataLayer.rows),
    cols: Math.max(0, launchedLayer.cols - dataLayer.cols),
  };

  const grid = workGroupGrid(global, local);
  const subGroupSize = device.subGroupSize;
  const localVolume = volumeOf(local);

  return {
    id: partition.id,
    layer: partition.layer,
    kind: partition.kind,
    connective: partition.connective,
    operation: partition.operation,
    nodeCount: partition.node_count,
    dependencyPartitionIds: partition.dependency_partition_ids ?? [],

    ndRange: { logical, local, global },
    bitpack,
    device,
    mapping,

    dataLayer,
    launchedLayer,
    padding,

    workGroupGrid: grid,
    workGroupCount: grid.x * grid.y * grid.z,
    workGroupShape: mapping.blockShape(local),

    subGroupSize,
    subGroupsPerGroup: Math.max(1, Math.round(localVolume / Math.max(1, subGroupSize))),
    subGroupShape: mapping.linearRunShape(local, subGroupSize),

    workItemCount: partition.work_item_count,
    logicalWorkItemCount: partition.logical_work_item_count,
    overprovisionedWorkItems: partition.overprovisioned_work_items,
  };
}

// ---------------------------------------------------------------------------
//  Window selection — choose a small tile crop of the data layer to render.
// ---------------------------------------------------------------------------

/** A work-group address in the per-axis work-group grid. */
export type WorkGroupCoord = Dim3;

/**
 * Tile window owned by one work-group, in *launched* (possibly padded)
 * coordinates. Columns are contiguous when `batch_size == 1`.
 */
export function workGroupWindow(kernel: KernelModel, group: WorkGroupCoord): TileWindow {
  const { local } = kernel.ndRange;
  const rowStart = group.x * local.x;
  const colStart = (group.y * local.y) * kernel.mapping.bitpacksPerBatch + group.z * local.z;
  return {
    row: span(rowStart, kernel.workGroupShape.rows),
    col: span(colStart, kernel.workGroupShape.cols),
  };
}

export type ClampedWindow = {
  /** Window clamped to the launched grid (what we render). */
  window: TileWindow;
  /** In-bounds (real data) portion of the window. */
  real: TileWindow;
  /** Padding tiles inside the window per axis. */
  padRows: number;
  padCols: number;
};

/** Clamp a window to the real data layer, reporting the padded remainder. */
export function clampWindow(kernel: KernelModel, window: TileWindow): ClampedWindow {
  const realRow = clampSpanToLimit(window.row, kernel.dataLayer.rows);
  const realCol = clampSpanToLimit(window.col, kernel.dataLayer.cols);
  return {
    window,
    real: { row: realRow, col: realCol },
    padRows: window.row.count - realRow.count,
    padCols: window.col.count - realCol.count,
  };
}

export type WindowBudget = {
  /** Hard cap on rendered tiles to bound point count / upload size. */
  maxTiles?: number;
  maxRows?: number;
  maxCols?: number;
};

function shrinkSpan(s: Span, max: number | undefined): Span {
  if (!max || s.count <= max) return s;
  return span(s.start, max);
}

/**
 * The recommended first focus from the design note: the first work-group window
 * `(0, 0, 0)`, optionally shrunk to a tile budget so the initial view stays
 * light. Returns both the (possibly shrunk) window and its work-group address.
 */
export function defaultWindow(
  kernel: KernelModel,
  budget: WindowBudget = {},
): { window: TileWindow; group: WorkGroupCoord } {
  const group: WorkGroupCoord = { x: 0, y: 0, z: 0 };
  let window = workGroupWindow(kernel, group);

  window = {
    row: shrinkSpan(window.row, budget.maxRows),
    col: shrinkSpan(window.col, budget.maxCols),
  };

  const maxTiles = budget.maxTiles;
  if (maxTiles && window.row.count * window.col.count > maxTiles) {
    // Preserve the column run (cache-line shaped) and trim rows first.
    const cols = Math.min(window.col.count, Math.max(1, kernel.workGroupShape.cols));
    const rows = Math.max(1, Math.floor(maxTiles / cols));
    window = { row: span(window.row.start, rows), col: span(window.col.start, cols) };
  }

  return { window, group };
}

// ---------------------------------------------------------------------------
//  Sub-divisions of a window into sub-group / work-item tiles (for overlays).
// ---------------------------------------------------------------------------

/** Tile a window into sub-group-sized blocks (row-major, columns fastest). */
export function subGroupWindows(kernel: KernelModel, window: TileWindow): TileWindow[] {
  const shape = kernel.subGroupShape;
  if (shape.rows <= 0 || shape.cols <= 0) return [];
  const windows: TileWindow[] = [];
  for (let r = window.row.start; r < window.row.start + window.row.count; r += shape.rows) {
    const rowCount = Math.min(shape.rows, window.row.start + window.row.count - r);
    for (let c = window.col.start; c < window.col.start + window.col.count; c += shape.cols) {
      const colCount = Math.min(shape.cols, window.col.start + window.col.count - c);
      windows.push({ row: span(r, rowCount), col: span(c, colCount) });
    }
  }
  return windows;
}

/** Number of sub-group blocks tiling a window (including partial edge blocks). */
export function subGroupBlockCount(kernel: KernelModel, window: TileWindow): number {
  const shape = kernel.subGroupShape;
  if (shape.rows <= 0 || shape.cols <= 0) return 0;
  return ceilDiv(window.row.count, shape.rows) * ceilDiv(window.col.count, shape.cols);
}
