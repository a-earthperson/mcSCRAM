/**
 * Kernel scene assembler — the boundary between the pure SYCL model and MathBox.
 *
 * Given a {@link KernelModel} and a palette, this builds the full renderable
 * subtree for one kernel: a bitpack field cropped to a small work-group window
 * plus the work-item / sub-group / work-group / padding overlays. Everything is
 * parented under a single `<group>` so the controller can show/hide a kernel in
 * O(1) without rebuilding GPU resources.
 *
 * The execution layer never reaches past this module: it deals in
 * `KernelModel`s and tile windows; this module is the only place that turns
 * those into shaders, world coordinates, and draw calls.
 */
import type { InkPalette } from "../../../../../components/plotTokens";
import {
  type ClampedWindow,
  type KernelModel,
  type TileWindow,
  type WindowBudget,
  clampWindow,
  defaultWindow,
  subGroupWindows,
} from "../sycl";
import { partitionKind } from "./executionScene";
import {
  type BitpackFieldConfig,
  type FieldExtent,
  type FieldFrame,
  type WorldRect,
  buildBitpackField,
} from "./bitpackField";
import { proceduralBitSource } from "./bitSource";
import { type FieldStyle, fieldStyle } from "./fieldStyle";
import { buildRegionOverlay, buildWorkItemGrid } from "./overlays";
import type { MathBoxView } from "./types";

export type KernelSceneOptions = {
  palette: InkPalette;
  /** Cap on rendered tiles / rows / cols for the opening window. */
  windowBudget?: WindowBudget;
  showWorkItems?: boolean;
  showSubGroups?: boolean;
  showWorkGroup?: boolean;
  showPadding?: boolean;
  bitPitch?: number;
  tilePitch?: number;
  pointSize?: number;
  vectorWidth?: number;
  /** Procedural bit-pattern seed; defaults to the kernel id (0 == historical). */
  bitSeed?: number;
  /** Initial visibility (controller toggles this for the active kernel). */
  visible?: boolean;
};

export type KernelSceneHandles = {
  kernelId: number;
  group: MathBoxView;
  frame: FieldFrame;
  extent: FieldExtent;
  /** Absolute (launched-coordinate) tile window rendered. */
  window: TileWindow;
  clamped: ClampedWindow;
  style: FieldStyle;
  setVisible: (visible: boolean) => void;
};

const DEFAULT_WINDOW_BUDGET: WindowBudget = { maxTiles: 64 * 32 };

/** Padded (masked-off) tile rectangles within the rendered window, field-local. */
function paddingRects(frame: FieldFrame, window: TileWindow, clamped: ClampedWindow): WorldRect[] {
  const rects: WorldRect[] = [];
  const realRows = clamped.real.row.count;
  const realCols = clamped.real.col.count;
  const padRows = window.row.count - realRows;
  const padCols = window.col.count - realCols;

  if (padRows > 0) {
    // Full-width strip of padded rows at the high edge.
    rects.push(frame.regionRect(0, window.col.count, realRows, padRows));
  }
  if (padCols > 0) {
    // Padded columns over the real rows only (avoids double-covering the corner).
    rects.push(frame.regionRect(realCols, padCols, 0, realRows));
  }
  return rects;
}

/**
 * Build the complete MathBox subtree for one kernel. The subtree is appended to
 * `view` (typically the cartesian view) under its own group.
 */
export function buildKernelScene(
  view: MathBoxView,
  kernel: KernelModel,
  options: KernelSceneOptions,
): KernelSceneHandles {
  const {
    palette,
    windowBudget = DEFAULT_WINDOW_BUDGET,
    showWorkItems = true,
    showSubGroups = true,
    showWorkGroup = true,
    showPadding = true,
    visible = true,
  } = options;

  const kind = partitionKind({ kind: kernel.kind, connective: kernel.connective });
  const style = fieldStyle(palette, kind, {
    pointSize: options.pointSize,
    vectorWidth: options.vectorWidth,
  });

  const { window } = defaultWindow(kernel, windowBudget);
  const clamped = clampWindow(kernel, window);

  const columns = Math.max(1, window.col.count);
  const rows = Math.max(1, window.row.count);
  const seed = options.bitSeed ?? kernel.id;

  const group: MathBoxView = view.group({ id: `kernel-${kernel.id}`, visible });

  const fieldConfig: BitpackFieldConfig = {
    columns,
    rows,
    bitsPerSide: kernel.bitpack.bitsPerSide,
    bitPitch: options.bitPitch ?? 0.02,
    tilePitch: options.tilePitch ?? 0.18,
    pointSize: style.pointSize,
    vectorWidth: style.vectorWidth,
    setBitZDisplacement: style.setBitZDisplacement,
    unsetBitZDisplacement: style.unsetBitZDisplacement,
    colors: style.bits,
    bitSource: proceduralBitSource(seed),
    idPrefix: `k${kernel.id}`,
  };

  const field = buildBitpackField(group, fieldConfig);
  const { frame } = field;

  if (showWorkItems) {
    buildWorkItemGrid(group, frame, style.workItem, `k${kernel.id}-`);
  }

  if (showSubGroups) {
    const subGroups = subGroupWindows(kernel, window);
    const rects = subGroups.map((sw) =>
      frame.regionRect(
        sw.col.start - window.col.start,
        sw.col.count,
        sw.row.start - window.row.start,
        sw.row.count,
      ),
    );
    buildRegionOverlay(group, rects, style.subGroup, `k${kernel.id}-sg`);
  }

  if (showWorkGroup) {
    buildRegionOverlay(group, [frame.fullRect()], style.workGroup, `k${kernel.id}-wg`);
  }

  if (showPadding) {
    const rects = paddingRects(frame, window, clamped);
    if (rects.length > 0) {
      buildRegionOverlay(group, rects, style.padding, `k${kernel.id}-pad`);
    }
  }

  return {
    kernelId: kernel.id,
    group,
    frame,
    extent: field.extent,
    window,
    clamped,
    style,
    setVisible: (next: boolean) => {
      try {
        group.set("visible", next);
      } catch {
        /* selection flavour without set — non-fatal */
      }
    },
  };
}
