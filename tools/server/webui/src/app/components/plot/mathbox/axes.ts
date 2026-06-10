/**
 * Native MathBox axis decoration: grid, axis spines, and tick labels.
 *
 * Replaces the previous SVG overlay entirely. Each axis is the canonical
 * MathBox pipeline from `references/context.html` + `references/format.html`:
 *
 *     scale  -> tick positions along an axis (auto-regenerated on range change)
 *     ticks  -> short tick marks sampling the scale
 *     format -> per-tick text via a numeric `expr`
 *     label  -> GL text sprites placed at the scale positions
 *
 * Axes are pinned to the plot box edges with `origin` (data coordinates): the
 * X axis/scale sit at y = yMin (bottom), the Y axis/scale at x = xMin (left), so
 * tick labels render in the reserved margins regardless of where 0 falls.
 */

import { type AxisFormat, type MathBoxNode, type TrajectoryAxesSpec, type Vec2 } from "./types";
import { splitColorAlpha } from "./color";

export type AxesUpdate = {
  xRange: Vec2;
  yRange: Vec2;
  spec: TrajectoryAxesSpec;
};

export type AxesHandle = {
  update: (next: AxesUpdate) => void;
  dispose: () => void;
};

const DEFAULT_DIVIDE = 6;
const TICK_SIZE = 6;
const LABEL_SIZE = 12;
const X_LABEL_OFFSET: Vec2 = [0, -16];
const Y_LABEL_OFFSET: Vec2 = [-18, 0];
// Visual hierarchy: a quiet grid backdrop, present-but-restrained spines, and
// full-ink ticks + labels carrying the actual information.
const GRID_OPACITY = 0.14;
const SPINE_OPACITY = 0.7;

/** A log axis is signalled by a defined `base` in the spec (linear leaves it
 * undefined). Log scales must use `nice:false` so ticks land on the exact
 * integer plot-space powers the host transform produced; linear scales keep
 * `nice:true` to round arbitrary data ranges to friendly steps. */
function isLog(base: number | undefined): boolean {
  return base !== undefined;
}

/** Wrap a formatter as a MathBox `format` expr (value is the 1st arg). */
function exprOf(format: AxisFormat | undefined): (value: number) => string {
  const fmt = format ?? ((value: number) => String(value));
  return (value: number) => fmt(value);
}

/**
 * Build grid + both axis pipelines onto `view`. The subtree is grouped so it
 * disposes in one call and never collides with series ids.
 */
export function createAxes(view: MathBoxNode, init: AxesUpdate): AxesHandle {
  // THREE.Color drops alpha, so a translucent ink (e.g. --rule) is split into
  // an opaque color + an alpha factor folded into each node's opacity.
  const { color, alpha } = splitColorAlpha(init.spec.color ?? "#000000");
  const group: MathBoxNode = view.group({ id: "axes" });

  const xDivide = init.spec.xDivide ?? DEFAULT_DIVIDE;
  const yDivide = init.spec.yDivide ?? DEFAULT_DIVIDE;
  const [xMin] = init.xRange;
  const [, yTop] = init.yRange;
  const yMin = init.yRange[0];

  // --- grid (behind everything) -------------------------------------------
  const grid: MathBoxNode = group.grid({
    id: "axes-grid",
    axes: [1, 2],
    divideX: xDivide,
    divideY: yDivide,
    width: 1,
    opacity: GRID_OPACITY * alpha,
    color,
    zBias: -5,
    zWrite: false,
    visible: init.spec.grid !== false,
  });

  // --- X axis: spine + scale -> ticks -> format -> label ------------------
  const xAxis: MathBoxNode = group.axis({
    id: "axes-x-axis",
    axis: 1,
    origin: [0, yMin, 0, 0],
    width: 2,
    opacity: SPINE_OPACITY * alpha,
    color,
    end: false,
    zBias: -1,
    zWrite: false,
  });
  const xScale: MathBoxNode = group.scale({
    id: "axes-x-scale",
    axis: 1,
    divide: xDivide,
    base: init.spec.xBase ?? 10,
    origin: [0, yMin, 0, 0],
    nice: !isLog(init.spec.xBase),
    zero: false,
  });
  group.ticks({
    id: "axes-x-ticks",
    points: "#axes-x-scale",
    size: TICK_SIZE,
    width: 1,
    color,
    opacity: alpha,
    zBias: -1,
    zWrite: false,
  });
  const xFormat: MathBoxNode = group.format({
    id: "axes-x-format",
    source: "#axes-x-scale",
    expr: exprOf(init.spec.xFormat),
  });
  group.label({
    id: "axes-x-label",
    points: "#axes-x-scale",
    text: "#axes-x-format",
    size: LABEL_SIZE,
    color,
    opacity: alpha,
    offset: X_LABEL_OFFSET,
    outline: 0,
    zIndex: 1,
    zWrite: false,
  });

  // --- Y axis: spine + scale -> ticks -> format -> label ------------------
  const yAxis: MathBoxNode = group.axis({
    id: "axes-y-axis",
    axis: 2,
    origin: [xMin, 0, 0, 0],
    width: 2,
    opacity: SPINE_OPACITY * alpha,
    color,
    end: false,
    zBias: -1,
    zWrite: false,
  });
  const yScale: MathBoxNode = group.scale({
    id: "axes-y-scale",
    axis: 2,
    divide: yDivide,
    base: init.spec.yBase ?? 10,
    origin: [xMin, 0, 0, 0],
    nice: !isLog(init.spec.yBase),
    zero: false,
  });
  group.ticks({
    id: "axes-y-ticks",
    points: "#axes-y-scale",
    size: TICK_SIZE,
    width: 1,
    color,
    opacity: alpha,
    zBias: -1,
    zWrite: false,
  });
  const yFormat: MathBoxNode = group.format({
    id: "axes-y-format",
    source: "#axes-y-scale",
    expr: exprOf(init.spec.yFormat),
  });
  group.label({
    id: "axes-y-label",
    points: "#axes-y-scale",
    text: "#axes-y-format",
    size: LABEL_SIZE,
    color,
    opacity: alpha,
    offset: Y_LABEL_OFFSET,
    outline: 0,
    zIndex: 1,
    zWrite: false,
  });

  // Track values so we only push changed props (avoids needless GL rebuilds).
  let prev = init;

  return {
    update(next) {
      // Dirty-checks compare the raw spec strings; the split (opaque color +
      // alpha-scaled opacity) is only computed when a node is actually set.
      const rawColor = next.spec.color ?? "#000000";
      const prevRawColor = prev.spec.color ?? "#000000";
      const colorChanged = rawColor !== prevRawColor;
      const nextInk = splitColorAlpha(rawColor);
      const nx = next.spec.xDivide ?? DEFAULT_DIVIDE;
      const ny = next.spec.yDivide ?? DEFAULT_DIVIDE;
      const nxMin = next.xRange[0];
      const nyMin = next.yRange[0];

      const nxBase = next.spec.xBase ?? 10;
      const nyBase = next.spec.yBase ?? 10;

      // The view range itself is set by the controller; scales/axes listen to
      // span.range and regenerate ticks for free. We only push the moving
      // origin (which edge the axis is pinned to) and any styling/format/divide/
      // base (the latter two change when an axis toggles linear<->log).
      if (
        nyMin !== prev.yRange[0] ||
        nx !== (prev.spec.xDivide ?? DEFAULT_DIVIDE) ||
        next.spec.xBase !== prev.spec.xBase ||
        colorChanged
      ) {
        xAxis.set({ origin: [0, nyMin, 0, 0], color: nextInk.color, opacity: SPINE_OPACITY * nextInk.alpha });
        xScale.set({ origin: [0, nyMin, 0, 0], divide: nx, base: nxBase, nice: !isLog(next.spec.xBase) });
      }
      if (
        nxMin !== prev.xRange[0] ||
        ny !== (prev.spec.yDivide ?? DEFAULT_DIVIDE) ||
        next.spec.yBase !== prev.spec.yBase ||
        colorChanged
      ) {
        yAxis.set({ origin: [nxMin, 0, 0, 0], color: nextInk.color, opacity: SPINE_OPACITY * nextInk.alpha });
        yScale.set({ origin: [nxMin, 0, 0, 0], divide: ny, base: nyBase, nice: !isLog(next.spec.yBase) });
      }
      if (next.spec.xFormat !== prev.spec.xFormat) {
        xFormat.set("expr", exprOf(next.spec.xFormat));
      }
      if (next.spec.yFormat !== prev.spec.yFormat) {
        yFormat.set("expr", exprOf(next.spec.yFormat));
      }
      if ((next.spec.grid !== false) !== (prev.spec.grid !== false) || colorChanged) {
        grid.set({ visible: next.spec.grid !== false, color: nextInk.color, opacity: GRID_OPACITY * nextInk.alpha });
      }
      void yTop;
      prev = next;
    },
    dispose() {
      group.remove();
    },
  };
}
