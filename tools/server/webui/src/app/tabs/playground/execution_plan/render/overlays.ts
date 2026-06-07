/**
 * SYCL geometry overlays drawn over a bitpack field.
 *
 * Each overlay tier corresponds to one execution unit from
 * `sycl_range_mapping.md`:
 *
 *   - work-item  -> one lattice cell per bitpack tile (a `<grid>`).
 *   - sub-group  -> warp-shaped tile rectangles (e.g. 2x16 on NVIDIA).
 *   - work-group -> the whole rendered window rectangle (the CTA footprint).
 *   - padding    -> masked-off rows/cols beyond the real data layer.
 *
 * Geometry comes entirely from a {@link FieldFrame} (world coordinates) and
 * pre-computed *field-local* tile rectangles, so overlays line up with the field
 * to the pixel. Rectangles are batched into a single draw call per tier to keep
 * the overlay cost negligible relative to the field.
 */
import * as THREE from "three";
import type { MathBoxView, Point3 } from "./types";
import { type FieldFrame, type WorldRect } from "./bitpackField";
import { type OverlayStyle, overlayFill, overlayLine } from "./fieldStyle";

/** Z planes for overlay tiers; small offsets keep them above the bit points. */
const Z_GRID = 0.0;
const Z_FILL = 0.0;
const Z_OUTLINE = 0.0;

/** Expand a world rectangle by `padding` on all sides. */
function padWorldRect(rect: WorldRect, padding: number): WorldRect {
  if (padding === 0) return rect;
  return {
    x0: rect.x0 - padding,
    y0: rect.y0 - padding,
    x1: rect.x1 + padding,
    y1: rect.y1 + padding,
  };
}

function rectCorners(rect: WorldRect, z: number): [Point3, Point3, Point3, Point3] {
  return [
    [rect.x0, rect.y0, z],
    [rect.x1, rect.y0, z],
    [rect.x1, rect.y1, z],
    [rect.x0, rect.y1, z],
  ];
}

/** Flatten rectangles into batched line segments (4 per rect, 2 endpoints each). */
function outlineSegments(rects: WorldRect[], z: number): Point3[] {
  const data: Point3[] = [];
  for (const rect of rects) {
    const [a, b, c, d] = rectCorners(rect, z);
    data.push(a, b, b, c, c, d, d, a);
  }
  return data;
}

/** Flatten rectangles into batched quads (4 corners per rect). */
function quadCorners(rects: WorldRect[], z: number): Point3[] {
  const data: Point3[] = [];
  for (const rect of rects) {
    const [a, b, c, d] = rectCorners(rect, z);
    data.push(a, b, c, d);
  }
  return data;
}

/** A rectangular block of bitpack tiles in field-local indices. */
export type GridRegion = { colStart: number; colCount: number; rowStart: number; rowCount: number };

/**
 * Work-item lattice: one cell per bitpack tile over a tile sub-region of the
 * field. Drawn as a single `<grid>` whose divisions land exactly on tile-cell
 * boundaries. A kernel's work-group occupies a sub-region of the (larger,
 * shared) field, so the lattice is scoped to that region rather than the field.
 */
export function buildWorkItemGrid(
  view: MathBoxView,
  frame: FieldFrame,
  region: GridRegion,
  style: OverlayStyle,
  idPrefix = "",
): MathBoxView {
  const rect = padWorldRect(
    frame.regionRect(region.colStart, region.colCount, region.rowStart, region.rowCount),
    style.padding,
  );
  return view.grid({
    id: `${idPrefix}wi-grid`,
    axes: [1, 2],
    rangeX: [rect.x0, rect.x1],
    rangeY: [rect.y0, rect.y1],
    divideX: Math.max(1, region.colCount),
    divideY: Math.max(1, region.rowCount),
    niceX: false,
    niceY: false,
    // Keep true: MathBox drops any tick exactly at 0 when false, and the field
    // is centred on the origin so a lattice line often lands on x=0 or y=0.
    zeroX: true,
    zeroY: true,
    width: style.width,
    color: overlayLine(style),
    opacity: style.lineOpacity,
    origin: [0, 0, Z_GRID, 0],
    zBias: 1,
    zWrite: false,
  });
}

export type RegionOverlayHandles = {
  fill?: MathBoxView;
  outline?: MathBoxView;
};

/**
 * Draw a set of field-local tile rectangles as a filled + outlined overlay tier.
 * `rects` are world rectangles already resolved through the field frame.
 */
export function buildRegionOverlay(
  view: MathBoxView,
  rects: WorldRect[],
  style: OverlayStyle,
  id: string,
): RegionOverlayHandles {
  if (rects.length === 0) return {};
  const handles: RegionOverlayHandles = {};
  const padded = style.padding === 0 ? rects : rects.map((rect) => padWorldRect(rect, style.padding));

  if (style.fillOpacity > 0) {
    handles.fill = view
      .array({ width: padded.length, items: 4, channels: 3, data: quadCorners(padded, Z_FILL), live: false })
      .face({
        id: `${id}-fill`,
        color: overlayFill(style),
        opacity: style.fillOpacity,
        fill: true,
        line: true,
        lineBias: 3,
        blending: THREE.AdditiveBlending,
        zBias: 2,
        zWrite: false,
      });
  }

  if (style.lineOpacity > 0) {
    handles.outline = view
      .array({ width: padded.length * 4, items: 2, channels: 3, data: outlineSegments(padded, Z_OUTLINE), live: false })
      .vector({
        id: `${id}-outline`,
        color: overlayLine(style),
        width: style.width,
        opacity: style.lineOpacity,
        start: false,
        end: false,
        blending: THREE.AdditiveBlending,
        zBias: 2,
        zWrite: false,
      });
  }

  return handles;
}

/** Convenience: resolve field-local tile windows to world rectangles. */
export function windowsToRects(
  frame: FieldFrame,
  windows: Array<{ col: { start: number; count: number }; row: { start: number; count: number } }>,
): WorldRect[] {
  return windows.map((w) => frame.regionRect(w.col.start, w.col.count, w.row.start, w.row.count));
}
