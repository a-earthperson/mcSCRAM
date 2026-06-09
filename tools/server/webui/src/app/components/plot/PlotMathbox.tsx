import { useEffect, useMemo, useRef, useState } from "react";
import {
  createTrajectoryPlot,
  type AxisFormat,
  type AxisScale,
  type TrajectoryFrame,
  type TrajectoryPlotController,
  type TrajectoryReference,
  type TrajectorySeries,
  type TrajectoryStroke,
} from "./mathbox/trajectoryPlot";
import { formatDefaultLinear } from "./mathbox/axis";
import {
  accumulate,
  emptyExtent,
  makeAxisTransform,
  powerSpan,
  resolveRange,
  type AxisTransform,
  type Extent,
} from "./mathbox/scaleTransform";

/**
 * Declarative React facade over the MathBox 2D plotter.
 *
 * Everything inside the plot frame -- grid, axis spines, ticks, numeric tick
 * labels, traces, bands, and reference lines -- is rendered natively in WebGL
 * by the imperative controller (`./mathbox/*`). The only DOM chrome here is the
 * textual surround that is not part of the plot geometry: the rotated Y-axis
 * title, the X-axis title, and the legend. The GL canvas fills the region
 * between those bands; the controller insets the data box for the tick labels.
 */

export type PlotMathboxAxis = {
  range: [number, number];
  title?: string;
  format?: AxisFormat;
  tickCount?: number;
  color?: string;
  /** Linear (default) or logarithmic with a configurable base (2/e/10). */
  scale?: AxisScale;
};

export type PlotMathboxSeries =
  | {
      id: string;
      kind: "line";
      x: ArrayLike<number>;
      y: ArrayLike<number>;
      color: string;
      width?: number;
      opacity?: number;
      stroke?: TrajectoryStroke;
      markerSize?: number;
      markerColor?: string;
      markerOpacity?: number;
      zIndex?: number;
    }
  | {
      id: string;
      kind: "band";
      x: ArrayLike<number>;
      lower: ArrayLike<number>;
      upper: ArrayLike<number>;
      color: string;
      opacity?: number;
      zIndex?: number;
    };

export type PlotMathboxReference = {
  /** Stable handle id; defaults to `ref-${axis}-${index}` when omitted. */
  id?: string;
  axis: "x" | "y";
  value: number;
  color: string;
  stroke?: TrajectoryStroke;
  width?: number;
  opacity?: number;
  /** Reserved for future GL annotation; not currently drawn. */
  label?: string;
  labelColor?: string;
};

export type PlotMathboxLegendItem = {
  label: string;
  color: string;
  stroke?: TrajectoryStroke;
  marker?: boolean;
  fill?: boolean;
};

export type PlotMathboxProps = {
  x: PlotMathboxAxis;
  y: PlotMathboxAxis;
  series: PlotMathboxSeries[];
  references?: PlotMathboxReference[];
  legend?: PlotMathboxLegendItem[];
  className?: string;
};

const Y_TITLE_WIDTH = 20;
const X_TITLE_HEIGHT = 18;
const LEGEND_MIN_HEIGHT = 26;

const DASH_ARRAY: Record<TrajectoryStroke, string | undefined> = {
  solid: undefined,
  dashed: "6 4",
  dotted: "2 3",
};

// Build helpers map value space -> plot space (identity for linear axes, and
// `log_base` for log axes), drop samples that don't survive the transform
// (non-finite, or non-positive on a log axis), and accumulate value-space
// extents so a log range can be bracketed to whole base-powers.

function buildLineData(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xT: AxisTransform,
  yT: AxisTransform,
  xExt: Extent,
  yExt: Extent,
): { data: Float32Array; count: number } {
  const n = Math.min(x.length, y.length);
  const out = new Float32Array(n * 2);
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    accumulate(xExt, x[i]);
    accumulate(yExt, y[i]);
    const px = xT.forward(x[i]);
    const py = yT.forward(y[i]);
    if (Number.isFinite(px) && Number.isFinite(py)) {
      out[count * 2] = px;
      out[count * 2 + 1] = py;
      count += 1;
    }
  }
  return { data: out, count };
}

function buildBandData(
  x: ArrayLike<number>,
  lower: ArrayLike<number>,
  upper: ArrayLike<number>,
  xT: AxisTransform,
  yT: AxisTransform,
  xExt: Extent,
  yExt: Extent,
): { x: Float32Array; lower: Float32Array; upper: Float32Array; count: number } {
  const n = Math.min(x.length, lower.length, upper.length);
  const xs = new Float32Array(n);
  const los = new Float32Array(n);
  const his = new Float32Array(n);
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    accumulate(xExt, x[i]);
    accumulate(yExt, lower[i]);
    accumulate(yExt, upper[i]);
    const px = xT.forward(x[i]);
    const plo = yT.forward(lower[i]);
    const phi = yT.forward(upper[i]);
    if (Number.isFinite(px) && Number.isFinite(plo) && Number.isFinite(phi)) {
      xs[count] = px;
      los[count] = plo;
      his[count] = phi;
      count += 1;
    }
  }
  return { x: xs, lower: los, upper: his, count };
}

/** Wrap a value-space formatter so it labels a plot-space tick (inverts log). */
function wrapFormat(transform: AxisTransform, format: AxisFormat | undefined): AxisFormat | undefined {
  if (transform.type === "linear") return format;
  const fmt = format ?? ((value: number) => String(value));
  return (plot: number) => fmt(transform.inverse(plot));
}

export function PlotMathbox({ x, y, series, references, legend, className }: PlotMathboxProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const glRef = useRef<HTMLDivElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<TrajectoryPlotController | null>(null);
  const [failed, setFailed] = useState(false);
  const hasLegend = Boolean(legend && legend.length);
  // Measure the legend so the GL inset reserves exactly enough room however many
  // rows it wraps to -- a fixed reserve clipped large legends (e.g. 6-9 items).
  const [legendHeight, setLegendHeight] = useState(LEGEND_MIN_HEIGHT);

  const glInset = useMemo(
    () => ({
      left: y.title ? Y_TITLE_WIDTH : 0,
      bottom: (x.title ? X_TITLE_HEIGHT : 0) + (hasLegend ? legendHeight : 0),
    }),
    [x.title, y.title, hasLegend, legendHeight],
  );

  // Translate declarative props into a controller frame (plot-space typed
  // arrays + axis spec). Recomputed whenever inputs change; the controller
  // diffs by id. All coordinates are mapped into plot space here so the
  // controller, axes, and series modules stay scale-agnostic.
  const frame = useMemo<TrajectoryFrame>(() => {
    const xT = makeAxisTransform(x.scale);
    const yT = makeAxisTransform(y.scale);
    const xExt = emptyExtent();
    const yExt = emptyExtent();

    const resolved: TrajectorySeries[] = series.map((entry, index) => {
      if (entry.kind === "band") {
        const band = buildBandData(entry.x, entry.lower, entry.upper, xT, yT, xExt, yExt);
        return {
          id: entry.id,
          kind: "band",
          x: band.x,
          lower: band.lower,
          upper: band.upper,
          count: band.count,
          color: entry.color,
          opacity: entry.opacity,
          zIndex: entry.zIndex ?? index,
        };
      }
      const line = buildLineData(entry.x, entry.y, xT, yT, xExt, yExt);
      return {
        id: entry.id,
        kind: "line",
        data: line.data,
        count: line.count,
        color: entry.color,
        width: entry.width,
        opacity: entry.opacity,
        stroke: entry.stroke,
        markerSize: entry.markerSize,
        markerColor: entry.markerColor,
        markerOpacity: entry.markerOpacity,
        zIndex: entry.zIndex ?? index + 10,
      };
    });

    // Reference values are mapped through their axis transform; a value that
    // doesn't survive (e.g. y<=0 on a log axis) is dropped.
    const refs: TrajectoryReference[] = [];
    (references ?? []).forEach((ref, index) => {
      const transform = ref.axis === "y" ? yT : xT;
      const value = transform.forward(ref.value);
      if (!Number.isFinite(value)) return;
      refs.push({
        id: ref.id ?? `ref-${ref.axis}-${index}`,
        axis: ref.axis,
        value,
        color: ref.color,
        stroke: ref.stroke ?? "dashed",
        width: ref.width,
        opacity: ref.opacity,
      });
    });

    const xRange = resolveRange(xT, x.range, xExt);
    const yRange = resolveRange(yT, y.range, yExt);

    return {
      xRange,
      yRange,
      series: resolved,
      references: refs,
      axes: {
        xFormat: wrapFormat(xT, x.format),
        yFormat: wrapFormat(yT, y.format ?? (yT.type === "linear" ? formatDefaultLinear : undefined)),
        xDivide: xT.type === "log" ? powerSpan(xRange) : x.tickCount,
        yDivide: yT.type === "log" ? powerSpan(yRange) : y.tickCount,
        xBase: xT.type === "log" ? xT.base : undefined,
        yBase: yT.type === "log" ? yT.base : undefined,
        color: x.color ?? y.color,
        grid: true,
      },
    };
  }, [series, references, x.range, y.range, x.format, y.format, x.tickCount, y.tickCount, x.color, y.color, x.scale, y.scale]);

  // Create the controller once; tear it down on unmount.
  useEffect(() => {
    const container = glRef.current;
    if (!container) return undefined;
    const controller = createTrajectoryPlot(container);
    controllerRef.current = controller;
    setFailed(controller === null);
    return () => {
      controllerRef.current = null;
      controller?.dispose();
    };
  }, []);

  // Keep the GL viewport synced to the canvas region (resizes on layout change).
  useEffect(() => {
    const container = glRef.current;
    if (!container) return undefined;
    const observer = new ResizeObserver(() => controllerRef.current?.resize());
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Track the legend's rendered height (it wraps to N rows by content/width) so
  // the reserved GL inset stays exact. Width is independent of the inset, so
  // there is no measurement feedback loop.
  useEffect(() => {
    const node = legendRef.current;
    if (!node) return undefined;
    const measure = () => setLegendHeight(Math.max(LEGEND_MIN_HEIGHT, Math.ceil(node.offsetHeight)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [hasLegend, legend]);

  // Push the latest frame to the GPU.
  useEffect(() => {
    if (failed) return;
    controllerRef.current?.render(frame);
  }, [frame, failed]);

  const rootClass = ["plot-mathbox", className].filter(Boolean).join(" ");

  return (
    <div ref={rootRef} className={rootClass}>
      <div
        ref={glRef}
        className="plot-mathbox__gl"
        style={{ left: glInset.left, top: 0, right: 0, bottom: glInset.bottom }}
        hidden={failed}
        aria-hidden="true"
      />

      {y.title ? (
        <div className="plot-mathbox__y-title" style={{ width: Y_TITLE_WIDTH, bottom: glInset.bottom }}>
          <span style={{ color: y.color }}>{y.title}</span>
        </div>
      ) : null}

      {x.title ? (
        <div
          className="plot-mathbox__x-title"
          style={{ left: glInset.left, height: X_TITLE_HEIGHT, bottom: hasLegend ? legendHeight : 0, color: x.color }}
        >
          {x.title}
        </div>
      ) : null}

      {hasLegend ? (
        <div className="plot-mathbox__legend" ref={legendRef}>
          {legend!.map((item, index) => (
            <span className="plot-mathbox__legend-item" key={`legend-${index}`}>
              <svg width={22} height={10} aria-hidden="true">
                {item.fill ? (
                  <rect x={0} y={1} width={22} height={8} fill={item.color} opacity={0.4} />
                ) : (
                  <line
                    x1={0}
                    x2={22}
                    y1={5}
                    y2={5}
                    stroke={item.color}
                    strokeWidth={2}
                    strokeDasharray={DASH_ARRAY[item.stroke ?? "solid"]}
                  />
                )}
                {item.marker ? <circle cx={11} cy={5} r={2.5} fill={item.color} /> : null}
              </svg>
              <span className="plot-mathbox__legend-label">{item.label}</span>
            </span>
          ))}
        </div>
      ) : null}

      {failed ? (
        <div className="plot-mathbox__fallback">WebGL unavailable; trajectory plot cannot render.</div>
      ) : null}
    </div>
  );
}
