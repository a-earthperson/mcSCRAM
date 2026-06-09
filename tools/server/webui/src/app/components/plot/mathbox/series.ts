/**
 * Drawable series for the trajectory plotter.
 *
 * Each factory composes a small MathBox subtree onto a parent `view` node and
 * returns an imperative {@link SeriesNode} handle that can be updated in place
 * or disposed. This mirrors the reference examples:
 *
 *   - a data source (`array` / `area`) feeds one or more draw primitives
 *     (`line` + `point`, or `surface`) -- data is decoupled from drawing
 *     (`references/curve.html`, `references/scatter.html`);
 *   - sources reserve `bufferWidth` once and vary `width` to reveal more samples
 *     without reallocating GPU buffers (`references/partial-data.html`);
 *   - a constant guide line is just an `axis` pinned to an `origin`
 *     (`references/context.html`).
 */

import {
  BUFFER_WIDTH,
  DEFAULT_LINE_WIDTH,
  STROKE_TO_MB,
  type MathBoxNode,
  type TrajectoryBandSeries,
  type TrajectoryLineSeries,
  type TrajectoryReference,
} from "./types";

/** An imperatively updatable drawable bound to one source subtree. */
export type SeriesNode<TSpec> = {
  readonly kind: TSpec extends { kind: infer K } ? K : string;
  update: (spec: TSpec) => void;
  dispose: () => void;
};

export type AnySeriesNode =
  | SeriesNode<TrajectoryLineSeries>
  | SeriesNode<TrajectoryReference>
  | (SeriesNode<TrajectoryBandSeries> & { kind: "band" });

/** Polyline (+ optional point markers) over interleaved [x,y,...] samples. */
export function createLineSeries(view: MathBoxNode, spec: TrajectoryLineSeries): SeriesNode<TrajectoryLineSeries> {
  const group: MathBoxNode = view.group({ id: `series-${spec.id}` });
  const source: MathBoxNode = group.array({
    id: `series-${spec.id}-data`,
    width: Math.max(1, spec.count),
    bufferWidth: BUFFER_WIDTH,
    channels: 2,
    items: 1,
    live: false,
    data: spec.data,
  });

  let line: MathBoxNode = null;
  let point: MathBoxNode = null;
  let current = spec;

  const buildDraw = (s: TrajectoryLineSeries) => {
    const lineWidth = s.width ?? DEFAULT_LINE_WIDTH;
    if (lineWidth > 0) {
      line = group.line({
        id: `series-${s.id}-line`,
        color: s.color,
        width: lineWidth,
        opacity: s.opacity ?? 1,
        stroke: STROKE_TO_MB[s.stroke ?? "solid"],
        start: false,
        end: false,
        zIndex: s.zIndex ?? 1,
        zWrite: false,
      });
    }
    if (s.markerSize && s.markerSize > 0) {
      point = group.point({
        id: `series-${s.id}-point`,
        color: s.markerColor ?? s.color,
        size: s.markerSize,
        opacity: s.opacity ?? 1,
        zIndex: (s.zIndex ?? 1) + 1,
        zWrite: false,
      });
    }
  };

  buildDraw(spec);

  const hasMarkers = (s: TrajectoryLineSeries) => Boolean(s.markerSize && s.markerSize > 0);

  return {
    kind: "line",
    update(next) {
      source.set({ width: Math.max(1, next.count), data: next.data });
      // Marker presence is structural (the `point` node either exists or not),
      // so toggling it rebuilds the draw subtree; everything else is a prop set.
      if (hasMarkers(next) !== hasMarkers(current)) {
        line?.remove();
        point?.remove();
        line = point = null;
        buildDraw(next);
      } else {
        line?.set({
          color: next.color,
          opacity: next.opacity ?? 1,
          width: next.width ?? DEFAULT_LINE_WIDTH,
          stroke: STROKE_TO_MB[next.stroke ?? "solid"],
        });
        point?.set({
          color: next.markerColor ?? next.color,
          size: next.markerSize ?? 0,
          opacity: next.opacity ?? 1,
        });
      }
      current = next;
    },
    dispose() {
      group.remove();
    },
  };
}

/** Filled ribbon between lower(x) and upper(x), drawn as a 2xN `surface`. */
export function createBandSeries(view: MathBoxNode, spec: TrajectoryBandSeries): SeriesNode<TrajectoryBandSeries> {
  // Mutable closure the `area` expr reads each frame; swap arrays on update and
  // bump `width` to re-stream, never reallocating (bufferWidth is reserved).
  const state = { x: spec.x, lower: spec.lower, upper: spec.upper, count: spec.count };

  const group: MathBoxNode = view.group({ id: `series-${spec.id}` });
  const source: MathBoxNode = group.area({
    id: `series-${spec.id}-area`,
    width: Math.max(2, state.count),
    height: 2,
    bufferWidth: BUFFER_WIDTH,
    bufferHeight: 2,
    channels: 2,
    items: 1,
    live: true,
    // Row j=0 -> lower envelope, j=1 -> upper envelope; column i -> sample i.
    expr: (emit: (x: number, y: number) => void, _x: number, _y: number, i: number, j: number) => {
      const idx = i < state.count ? i : Math.max(0, state.count - 1);
      emit(state.x[idx], j === 0 ? state.lower[idx] : state.upper[idx]);
    },
  });
  const surface: MathBoxNode = group.surface({
    id: `series-${spec.id}-fill`,
    color: spec.color,
    opacity: spec.opacity ?? 0.12,
    fill: true,
    shaded: false,
    lineX: false,
    lineY: false,
    zIndex: spec.zIndex ?? 0,
    zWrite: false,
  });

  return {
    kind: "band",
    update(next) {
      state.x = next.x;
      state.lower = next.lower;
      state.upper = next.upper;
      state.count = next.count;
      source.set("width", Math.max(2, next.count));
      surface.set({ color: next.color, opacity: next.opacity ?? 0.12 });
    },
    dispose() {
      group.remove();
    },
  };
}

/**
 * Constant-coordinate guide line. Implemented as an `axis` pinned by `origin`
 * so it spans the full range and tracks range changes for free: a horizontal
 * line at y=v is axis 1 with origin y=v; a vertical line at x=v is axis 2 with
 * origin x=v.
 */
export function createReference(view: MathBoxNode, spec: TrajectoryReference): SeriesNode<TrajectoryReference> {
  const axis = spec.axis === "y" ? 1 : 2;
  const origin = spec.axis === "y" ? [0, spec.value, 0, 0] : [spec.value, 0, 0, 0];
  const node: MathBoxNode = view.axis({
    id: `ref-${spec.id}`,
    axis,
    origin,
    width: spec.width ?? 1.2,
    color: spec.color,
    opacity: spec.opacity ?? 1,
    stroke: STROKE_TO_MB[spec.stroke ?? "dashed"],
    start: false,
    end: false,
    zBias: -1,
    zWrite: false,
  });

  return {
    kind: spec.axis === "y" ? "y" : "x",
    update(next) {
      node.set({
        origin: next.axis === "y" ? [0, next.value, 0, 0] : [next.value, 0, 0, 0],
        color: next.color,
        opacity: next.opacity ?? 1,
        width: next.width ?? 1.2,
        stroke: STROKE_TO_MB[next.stroke ?? "dashed"],
      });
    },
    dispose() {
      node.remove();
    },
  };
}
