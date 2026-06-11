/**
 * Shared types for the MathBox 2D trajectory plotter.
 *
 * The plotter is split into small, single-responsibility modules that each
 * compose imperatively onto a MathBox cartesian `view` node:
 *
 *   - {@link ./trajectoryPlot}  bootstrap + camera + view + lifecycle (the handle)
 *   - {@link ./axes}            grid, axis spines, and the scale->ticks->format->label
 *                               pipeline (native GL axis decoration)
 *   - {@link ./series}          data sources (`array`/`area`) and their draw
 *                               primitives (`line`/`point`/`surface`)
 *   - {@link ./axis}            pure number formatters used as `format` exprs
 */

/** Untyped MathBox selection (the meta-API ships no usable types). */
export type MathBoxNode = any;

export type Vec2 = readonly [number, number];

export type TrajectoryStroke = "solid" | "dashed" | "dotted";

/** Maps a MathBox stroke style name onto the primitive's `stroke` prop. */
export const STROKE_TO_MB: Record<TrajectoryStroke, string> = {
  solid: "solid",
  dashed: "dashed",
  dotted: "dotted",
};

/** A numeric -> text formatter used to label a scale's ticks. */
export type AxisFormat = (value: number) => string;

/** Axis mapping mode. `log` plots `log_base(value)` (base sets the unit:
 * 2 = shannon/bit, e = nat, 10 = hartley/dex). */
export type AxisScaleType = "linear" | "log";

/** Per-axis scale configuration (plot-level). */
export type AxisScale = {
  type: AxisScaleType;
  /** Log base; only meaningful when `type === "log"`. Defaults to 10. */
  base?: number;
};

/**
 * Pre-allocated GPU buffer width per series. MathBox reallocates the data
 * texture whenever the requested width exceeds the buffer width, so we reserve
 * a generous, fixed buffer once and merely *refill* it as the streamed point
 * count grows -- keeping the per-frame update on the fast path (no realloc).
 * See `references/partial-data.html`: reserve `bufferWidth`, vary `width`.
 */
export const BUFFER_WIDTH = 4096;

/** Default GL line width (px) when a series omits `width`. */
export const DEFAULT_LINE_WIDTH = 2;

/** A polyline (optionally with point markers) over interleaved [x,y,...] data. */
export type TrajectoryLineSeries = {
  id: string;
  kind: "line";
  /** Interleaved [x0, y0, x1, y1, ...]; the first `count` points are drawn. */
  data: Float32Array;
  count: number;
  color: string;
  width?: number;
  opacity?: number;
  stroke?: TrajectoryStroke;
  /** Marker size in px; omit/0 for a bare line. */
  markerSize?: number;
  markerColor?: string;
  /** Marker alpha (MathBox `point.opacity`); defaults to `opacity` then 1. */
  markerOpacity?: number;
  /** 2D stacking order; higher draws on top. */
  zIndex?: number;
};

/** A filled ribbon between `lower(x)` and `upper(x)`. */
export type TrajectoryBandSeries = {
  id: string;
  kind: "band";
  x: Float32Array;
  lower: Float32Array;
  upper: Float32Array;
  count: number;
  color: string;
  opacity?: number;
  zIndex?: number;
};

export type TrajectorySeries = TrajectoryLineSeries | TrajectoryBandSeries;

/** A constant-coordinate guide line (e.g. a zero baseline or an oracle p). */
export type TrajectoryReference = {
  id: string;
  /** "y" -> horizontal line at `value`; "x" -> vertical line at `value`. */
  axis: "x" | "y";
  value: number;
  color: string;
  width?: number;
  opacity?: number;
  stroke?: TrajectoryStroke;
};

/** Native axis decoration spec (grid + spines + ticks + numeric labels).
 *
 * Coordinates handed to the controller are always in *plot space* (already
 * `log_base`-transformed for log axes), so this spec only needs the per-axis
 * `base` (which makes MathBox's nice-tick algorithm land steps on whole
 * base-powers) and a `format` that maps a plot-space tick back to a label. */
export type TrajectoryAxesSpec = {
  xFormat?: AxisFormat;
  yFormat?: AxisFormat;
  /** Target tick counts (forwarded to each `scale.divide`). */
  xDivide?: number;
  yDivide?: number;
  /** Power base for the nice-tick step rounding (log axes). */
  xBase?: number;
  yBase?: number;
  grid?: boolean;
  color?: string;
  /** CSS font-family stack for GL tick labels (MathBox `format.font`). */
  tickFont?: string;
};

export type TrajectoryFrame = {
  xRange: Vec2;
  yRange: Vec2;
  series: readonly TrajectorySeries[];
  references?: readonly TrajectoryReference[];
  axes?: TrajectoryAxesSpec;
};

/** Plot-area inset (px) within the GL canvas, reserved for tick labels. */
export type TrajectoryMargin = { left: number; right: number; top: number; bottom: number };

/** Matches |Y_LABEL_OFFSET.x| in {@link ./axes} so tick labels sit in [0, left] px. */
export const Y_TICK_LABEL_OFFSET_LEFT = 72;

export const DEFAULT_MARGIN: TrajectoryMargin = {
  left: Y_TICK_LABEL_OFFSET_LEFT,
  right: 18,
  top: 14,
  bottom: 26,
};

export type TrajectoryPlotController = {
  render: (frame: TrajectoryFrame) => void;
  resize: () => void;
  dispose: () => void;
};
