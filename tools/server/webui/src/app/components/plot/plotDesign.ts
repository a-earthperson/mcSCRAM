import {
  getInkPalette,
  getMethodStyle,
  PLOT_TRANSPARENT,
  withAlpha,
  type MethodStyle,
} from "./plotTokens";

export type { MethodStyle };
export { getMethodStyle, withAlpha, PLOT_TRANSPARENT };

export type PlotTrace = Record<string, unknown>;
export type PlotShape = Record<string, unknown>;
export type PlotAnnotation = Record<string, unknown>;

export type ReferenceLayer = {
  shapes: PlotShape[];
  annotations: PlotAnnotation[];
};

export type LogFitResult = {
  slope: number;
  intercept: number;
  r2: number;
  n: number;
  xMin: number;
  xMax: number;
};

const flagSymbols: Record<string, string> = {
  C0: "circle",
  C1: "circle-open",
  C2: "square",
  C3: "square-open",
  C4: "diamond",
  C8: "triangle-up",
};

export function compileFlagSymbol(flag: string | null | undefined): string {
  return flag ? flagSymbols[flag] ?? "circle" : "circle";
}

export function horizontalReference(value: number | null | undefined, label: string, options: { color?: string; dash?: string; width?: number; yshift?: number } = {}): ReferenceLayer {
  if (!isPositiveFinite(value)) return emptyReference();
  const palette = getInkPalette();
  const color = options.color ?? palette.ink;
  return {
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: value,
        y1: value,
        line: { color, width: options.width ?? 1.4, dash: options.dash ?? "dash" },
      },
    ],
    annotations: [
      {
        xref: "paper",
        x: 1,
        xanchor: "right",
        yref: "y",
        y: value,
        yanchor: "bottom",
        yshift: options.yshift ?? 4,
        text: label,
        showarrow: false,
        font: { color, size: 11 },
        bgcolor: palette.annotationBg,
        borderpad: 2,
      },
    ],
  };
}

export function verticalReference(value: number | null | undefined, label: string, options: { color?: string; dash?: string; width?: number; xshift?: number } = {}): ReferenceLayer {
  if (!isPositiveFinite(value)) return emptyReference();
  const palette = getInkPalette();
  const color = options.color ?? palette.ink;
  return {
    shapes: [
      {
        type: "line",
        xref: "x",
        x0: value,
        x1: value,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color, width: options.width ?? 1.2, dash: options.dash ?? "dot" },
      },
    ],
    annotations: [
      {
        xref: "x",
        x: value,
        xanchor: "left",
        xshift: options.xshift ?? 4,
        yref: "paper",
        y: 1,
        yanchor: "top",
        text: label,
        showarrow: false,
        font: { color, size: 11 },
        bgcolor: palette.annotationBg,
        borderpad: 2,
      },
    ],
  };
}

export function mergeReferenceLayers(...layers: ReferenceLayer[]): ReferenceLayer {
  return {
    shapes: layers.flatMap((layer) => layer.shapes),
    annotations: layers.flatMap((layer) => layer.annotations),
  };
}

export function confidenceBandTrace({
  name,
  x,
  lower,
  upper,
  color,
  opacity,
  legendgroup,
  showlegend = true,
}: {
  name: string;
  x: Array<number | null | undefined>;
  lower: Array<number | null | undefined>;
  upper: Array<number | null | undefined>;
  color: string;
  opacity: number;
  legendgroup?: string;
  showlegend?: boolean;
}): PlotTrace | null {
  const points = x
    .map((value, index) => ({ x: value, lo: lower[index], hi: upper[index] }))
    .filter((point): point is { x: number; lo: number; hi: number } => isPositiveFinite(point.x) && isPositiveFinite(point.lo) && isPositiveFinite(point.hi));
  if (points.length < 2) return null;
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: [...points.map((point) => point.x), ...points.map((point) => point.x).reverse()],
    y: [...points.map((point) => point.hi), ...points.map((point) => point.lo).reverse()],
    fill: "toself",
    fillcolor: withAlpha(color, opacity),
    line: { color: PLOT_TRANSPARENT, width: 0 },
    hoverinfo: "skip",
    legendgroup,
    showlegend,
  };
}

export function logFit(xs: Array<number | null | undefined>, ys: Array<number | null | undefined>): LogFitResult | null {
  const points = xs
    .map((x, index) => ({ x, y: ys[index] }))
    .filter((point): point is { x: number; y: number } => isPositiveFinite(point.x) && isPositiveFinite(point.y));
  if (points.length < 3) return null;

  const lx = points.map((point) => Math.log10(point.x));
  const ly = points.map((point) => Math.log10(point.y));
  const xMean = mean(lx);
  const yMean = mean(ly);
  const denominator = lx.reduce((sum, value) => sum + (value - xMean) ** 2, 0);
  if (denominator === 0) return null;

  const slope = lx.reduce((sum, value, index) => sum + (value - xMean) * (ly[index] - yMean), 0) / denominator;
  const intercept = yMean - slope * xMean;
  const residual = ly.reduce((sum, value, index) => {
    const fitted = slope * lx[index] + intercept;
    return sum + (value - fitted) ** 2;
  }, 0);
  const total = ly.reduce((sum, value) => sum + (value - yMean) ** 2, 0);
  const xValues = points.map((point) => point.x);

  return {
    slope,
    intercept,
    r2: total === 0 ? 1 : 1 - residual / total,
    n: points.length,
    xMin: Math.min(...xValues),
    xMax: Math.max(...xValues),
  };
}

export function logFitTrace(fit: LogFitResult | null, name: string, options: { color?: string; dash?: string; width?: number } = {}): PlotTrace | null {
  if (!fit) return null;
  const methodStyle = getMethodStyle();
  const xs = logspace(fit.xMin, fit.xMax, 96);
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: xs,
    y: xs.map((x) => 10 ** (fit.slope * Math.log10(x) + fit.intercept)),
    line: { color: options.color ?? methodStyle.fit.color, dash: options.dash ?? methodStyle.fit.dash, width: options.width ?? 2 },
    hovertemplate: `${name}<br>slope=${fit.slope.toFixed(2)}<br>R^2=${fit.r2.toFixed(2)}<br>n=${fit.n}<extra>log-log fit</extra>`,
  };
}

export function percentile(values: Array<number | null | undefined>, p: number): number | null {
  const present = values.filter(isFiniteNumber).sort((a, b) => a - b);
  if (!present.length) return null;
  const rank = (present.length - 1) * p;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return present[lo];
  return present[lo] + (present[hi] - present[lo]) * (rank - lo);
}

export function median(values: Array<number | null | undefined>): number | null {
  return percentile(values, 0.5);
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

export function isFiniteNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}

export function isPositiveFinite(value: number | null | undefined): value is number {
  return isFiniteNumber(value) && value > 0;
}

function emptyReference(): ReferenceLayer {
  return { shapes: [], annotations: [] };
}

function logspace(min: number, max: number, steps: number): number[] {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return Array.from({ length: steps }, (_, index) => 10 ** (logMin + ((logMax - logMin) * index) / (steps - 1)));
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
