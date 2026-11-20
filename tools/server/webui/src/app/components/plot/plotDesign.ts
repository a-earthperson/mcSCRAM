import { PLOT_TRANSPARENT, withAlpha } from "./plotTokens";

export { PLOT_TRANSPARENT, withAlpha };

export type PlotTrace = Record<string, unknown>;

function isPositiveFinite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

export function logAxisRange(
  values: Array<number | null | undefined>,
  options: { pad?: number; floor?: number; ceiling?: number; include?: Array<number | null | undefined> } = {},
): [number, number] | undefined {
  const pad = options.pad ?? 0.08;
  const present = [...values, ...(options.include ?? [])]
    .filter(isPositiveFinite)
    .map((value) => {
      let next = value;
      if (options.floor !== undefined) next = Math.max(next, options.floor);
      if (options.ceiling !== undefined) next = Math.min(next, options.ceiling);
      return next;
    })
    .filter(isPositiveFinite);
  if (!present.length) return undefined;
  let min = Math.min(...present);
  let max = Math.max(...present);
  if (min <= 0 || max <= 0) return undefined;
  if (min > max) [min, max] = [max, min];
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  const span = hi - lo;
  return [lo - span * pad, hi + span * pad];
}
