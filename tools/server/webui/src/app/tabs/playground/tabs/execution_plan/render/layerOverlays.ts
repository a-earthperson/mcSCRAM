/**
 * Per-layer SYCL launch-geometry overlays, registered onto the shared field.
 *
 * The bitpack field is world-fixed and shared across all layers; only these
 * overlays change when the layer cursor moves. For a given kernel we draw, in
 * field-local tile coordinates anchored at the field *core* origin (past the
 * faded padding):
 *
 *   - the work-item lattice over the kernel's work-group footprint,
 *   - the sub-group (warp) rectangles tiling that footprint,
 *   - the work-group rectangle itself.
 *
 * Each kernel's overlays live under their own `<transform>` + `<group>`: the
 * transform supplies the Z travel used by layer transitions, the group the
 * visibility toggle, and `setOpacity` scales every draw node's opacity for the
 * cross-fade. The controller owns the tween; this module just exposes the knobs.
 */
import type { InkPalette } from "../../../../../components/plot/plotTokens";
import { type KernelModel, span, subGroupWindows } from "../sycl";
import { partitionKind } from "../scene/executionScene";
import type { FieldFrame } from "./bitpackField";
import { type FieldStyle, fieldStyle } from "./fieldStyle";
import { buildRegionOverlay, buildWorkItemGrid } from "./overlays";
import type { MathBoxView } from "./types";

export type LayerOverlayOptions = {
  palette: InkPalette;
  showWorkItems?: boolean;
  showSubGroups?: boolean;
  showWorkGroup?: boolean;
  visible?: boolean;
  /** Core tile offset (field padding) so overlays anchor on the core. */
  origin?: { col: number; row: number };
};

export type LayerOverlayHandles = {
  kernelId: number;
  style: FieldStyle;
  /** Work-group footprint actually drawn (clamped to the field), in tiles. */
  footprint: { rows: number; cols: number };
  setVisible: (visible: boolean) => void;
  /** Scale every draw node's opacity by `alpha` (relative to its base). */
  setOpacity: (alpha: number) => void;
  /** Translate the overlay group along Z (the transition travel axis). */
  setTravel: (z: number) => void;
  /** Dispose the overlay subtree. */
  remove: () => void;
};

/** A draw node paired with the opacity it shows at full strength. */
type FadeNode = { node: MathBoxView; base: number };

function safeSet(node: MathBoxView, key: string, value: unknown): void {
  try {
    node.set(key, value);
  } catch {
    /* selection flavour without set — non-fatal */
  }
}

export function buildLayerOverlays(
  view: MathBoxView,
  frame: FieldFrame,
  kernel: KernelModel,
  options: LayerOverlayOptions,
): LayerOverlayHandles {
  const {
    palette,
    showWorkItems = true,
    showSubGroups = true,
    showWorkGroup = true,
    visible = true,
    origin = { col: 0, row: 0 },
  } = options;

  const kind = partitionKind({ kind: kernel.kind, connective: kernel.connective });
  const style = fieldStyle(palette, kind);

  const cols = Math.max(1, Math.min(kernel.workGroupShape.cols, frame.columns));
  const rows = Math.max(1, Math.min(kernel.workGroupShape.rows, frame.rows));
  const footprint = { rows, cols };
  const oc = origin.col;
  const or = origin.row;

  const transform: MathBoxView = view.transform({ id: `layer-xform-${kernel.id}`, position: [0, 0, 0] });
  const group: MathBoxView = transform.group({ id: `overlay-${kernel.id}`, visible });
  const prefix = `o${kernel.id}-`;
  const fades: FadeNode[] = [];

  if (showWorkItems) {
    const grid = buildWorkItemGrid(
      group,
      frame,
      { colStart: oc, colCount: cols, rowStart: or, rowCount: rows },
      style.workItem,
      prefix,
    );
    fades.push({ node: grid, base: style.workItem.lineOpacity });
  }

  if (showSubGroups) {
    const wgWindow = { row: span(0, rows), col: span(0, cols) };
    const rects = subGroupWindows(kernel, wgWindow).map((sw) =>
      frame.regionRect(oc + sw.col.start, sw.col.count, or + sw.row.start, sw.row.count),
    );
    const handles = buildRegionOverlay(group, rects, style.subGroup, `${prefix}sg`);
    if (handles.fill) fades.push({ node: handles.fill, base: style.subGroup.fillOpacity });
    if (handles.outline) fades.push({ node: handles.outline, base: style.subGroup.lineOpacity });
  }

  if (showWorkGroup) {
    const handles = buildRegionOverlay(group, [frame.regionRect(oc, cols, or, rows)], style.workGroup, `${prefix}wg`);
    if (handles.fill) fades.push({ node: handles.fill, base: style.workGroup.fillOpacity });
    if (handles.outline) fades.push({ node: handles.outline, base: style.workGroup.lineOpacity });
  }

  return {
    kernelId: kernel.id,
    style,
    footprint,
    setVisible: (next) => safeSet(group, "visible", next),
    setOpacity: (alpha) => {
      for (const fade of fades) safeSet(fade.node, "opacity", Math.max(0, fade.base * alpha));
    },
    setTravel: (z) => safeSet(transform, "position", [0, 0, z]),
    remove: () => {
      try {
        transform.remove();
      } catch {
        /* already detached */
      }
    },
  };
}
