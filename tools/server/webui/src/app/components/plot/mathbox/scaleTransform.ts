/**
 * Per-axis linear/log coordinate transforms for the trajectory plotter.
 *
 * MathBox's cartesian view cannot map logarithmically, and its native log-tick
 * generator throws ("Log ticks not yet implemented"). So a log axis is realised
 * entirely host-side: data and the view range are mapped into *plot space*
 * `p = log_base(value)`, the view stays linear, and the `scale` primitive's
 * `base` makes its (linear) nice-tick algorithm round steps to whole powers of
 * the base -- i.e. one tick per base-power. Tick labels invert the transform
 * (`value = base^p`).
 *
 * The base sets the information unit of one decade: base 2 = shannon (bit),
 * base e = nat, base 10 = hartley (dex).
 */

import { type AxisScale } from "./types";

export type AxisTransform = {
  type: "linear" | "log";
  base: number;
  /** value -> plot-space coordinate. */
  forward: (value: number) => number;
  /** plot-space coordinate -> value. */
  inverse: (plot: number) => number;
};

const IDENTITY: AxisTransform = {
  type: "linear",
  base: 10,
  forward: (value) => value,
  inverse: (plot) => plot,
};

/** Build a transform from a per-axis scale config (undefined -> linear). */
export function makeAxisTransform(scale?: AxisScale): AxisTransform {
  if (!scale || scale.type !== "log") return IDENTITY;
  // Base must be > 0 and != 1 for log to be well-defined; fall back to 10.
  const base = scale.base && scale.base > 0 && scale.base !== 1 ? scale.base : 10;
  const lnBase = Math.log(base);
  return {
    type: "log",
    base,
    forward: (value) => (value > 0 ? Math.log(value) / lnBase : Number.NaN),
    inverse: (plot) => base ** plot,
  };
}

/** Running min / min-positive / max accumulator over a value-space axis. */
export type Extent = { min: number; minPositive: number; max: number };

export function emptyExtent(): Extent {
  return { min: Number.POSITIVE_INFINITY, minPositive: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY };
}

export function accumulate(extent: Extent, value: number): void {
  if (!Number.isFinite(value)) return;
  if (value < extent.min) extent.min = value;
  if (value > extent.max) extent.max = value;
  if (value > 0 && value < extent.minPositive) extent.minPositive = value;
}

/**
 * Resolve a plot-space `[lo, hi]` range for one axis.
 *
 * - linear: forward the caller-supplied value-space range (identity).
 * - log: bracket the positive data extent to whole base-powers so ticks land
 *   exactly on each power; falls back to the supplied range when it is already
 *   positive, and to a unit decade when there is no positive data.
 */
export function resolveRange(
  transform: AxisTransform,
  suppliedRange: readonly [number, number],
  extent: Extent,
): [number, number] {
  if (transform.type === "linear") {
    return [suppliedRange[0], suppliedRange[1]];
  }

  const lowCandidate = suppliedRange[0] > 0 ? Math.min(suppliedRange[0], extent.minPositive) : extent.minPositive;
  const highCandidate = Math.max(suppliedRange[1] > 0 ? suppliedRange[1] : 0, extent.max);

  if (!Number.isFinite(lowCandidate) || !Number.isFinite(highCandidate) || highCandidate <= 0) {
    return [0, 1];
  }

  const lo = Math.floor(transform.forward(lowCandidate));
  let hi = Math.ceil(transform.forward(highCandidate));
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

/** Whole base-powers spanned by a plot-space range (the natural tick count). */
export function powerSpan(range: readonly [number, number]): number {
  return Math.max(1, Math.round(range[1] - range[0]));
}
