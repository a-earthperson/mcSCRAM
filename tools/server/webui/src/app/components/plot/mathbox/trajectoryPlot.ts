import * as THREE from "three";
import { mathBox } from "mathbox";

import {
  DEFAULT_MARGIN,
  type MathBoxNode,
  type TrajectoryFrame,
  type TrajectoryMargin,
  type TrajectoryPlotController,
} from "./types";
import { createAxes, type AxesHandle } from "./axes";
import { type AnySeriesNode, createBandSeries, createLineSeries, createReference } from "./series";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export type {
  AxisFormat,
  AxisScale,
  AxisScaleType,
  TrajectoryAxesSpec,
  TrajectoryBandSeries,
  TrajectoryFrame,
  TrajectoryLineSeries,
  TrajectoryMargin,
  TrajectoryPlotController,
  TrajectoryReference,
  TrajectorySeries,
  TrajectoryStroke,
  Vec2,
} from "./types";

/**
 * High-performance 2D plotter built on MathBox (WebGL).
 *
 * This module is just the *plot handle*: it bootstraps the MathBox context,
 * frames a head-on 2D camera, owns the cartesian `view`, and drives lifecycle
 * (resize / dispose). All decoration and data are delegated to single-purpose
 * modules that compose imperatively onto the `view`:
 *
 *   - `./axes`   grid + axis spines + scale->ticks->format->label (native GL)
 *   - `./series` line / point / band draw primitives over data sources
 *
 * Framing trick (from the upstream `ortho.html`): all geometry lives on z=0 and
 * is viewed head-on, so a perspective camera yields zero distortion while
 * MathBox's pixel calibration stays valid. The cartesian *data box* is inset
 * within the canvas via `position` + `scale` so the native tick labels render
 * in the reserved margins (see {@link TrajectoryMargin}).
 */

/** Vertical FOV for the head-on camera; D solves D·tan(fov/2)=1 so the visible
 * half-height at z=0 is exactly 1 world unit. */
const CAMERA_FOV_DEG = 45;
const CAMERA_DISTANCE = 1 / Math.tan((CAMERA_FOV_DEG / 2) * (Math.PI / 180));

export type TrajectoryPlotOptions = { margin?: Partial<TrajectoryMargin> };

function clearContainer(container: HTMLElement): void {
  try {
    for (const canvas of container.querySelectorAll("canvas")) {
      for (const id of ["webgl2", "webgl", "experimental-webgl"] as const) {
        const gl = canvas.getContext(id) as WebGLRenderingContext | null;
        gl?.getExtension("WEBGL_lose_context")?.loseContext();
      }
    }
    while (container.firstChild) container.removeChild(container.firstChild);
  } catch {
    /* ignore */
  }
}

export function createTrajectoryPlot(
  container: HTMLElement,
  options: TrajectoryPlotOptions = {},
): TrajectoryPlotController | null {
  const margin: TrajectoryMargin = { ...DEFAULT_MARGIN, ...options.margin };

  let mathbox: MathBoxNode = null;
  try {
    // Listing "mathbox" registers the meta-API before bootstrap so the
    // fullscreen "splash" loader is never auto-installed over the panel.
    // "controls" plugin for camera orbiting.
    mathbox = mathBox({
      element: container,
      plugins: ["core", "controls", "cursor", "mathbox"],
      controls: { klass: OrbitControls },
      camera: { fov: CAMERA_FOV_DEG, near: 0.01, far: CAMERA_DISTANCE * 16 },
      renderer: {
        parameters: {
          alpha: true,
          antialias: true,
          premultipliedAlpha: true,
          preserveDrawingBuffer: true,
          depth: true,
          stencil: false,
        },
      },
      mathbox: { init: true, inspect: false, warmup: 0, splash: true },
    });
  } catch {
    clearContainer(container);
    return null;
  }

  // WebGL missing -> threestrap returns a fallback without the meta-API.
  if (!mathbox || mathbox.fallback || typeof mathbox.cartesian !== "function") {
    try {
      mathbox?.three?.renderer?.forceContextLoss?.();
      mathbox?.three?.renderer?.dispose?.();
      mathbox?.three?.destroy?.();
    } catch {
      /* ignore */
    }
    clearContainer(container);
    return null;
  }

  const three = mathbox.three;

  try {
    three.renderer.setClearColor(new THREE.Color(0x000000), 0);
    three.renderer.setClearAlpha?.(0);
  } catch {
    /* some renderer flavours lack setClearColor */
  }

  // Unit calibration for the head-on camera. `scale` stays at its default
  // (null -> use the live viewport height), so line widths / point sizes / tick
  // sizes / label offsets are all in true device pixels at any panel size.
  mathbox.set({ focus: CAMERA_DISTANCE, fov: CAMERA_FOV_DEG });

  try {
    const camera = three.camera;
    camera.position?.set?.(0, 0, CAMERA_DISTANCE);
    camera.up?.set?.(0, 1, 0);
    camera.lookAt?.(new THREE.Vector3(0, 0, 0));
    if ("fov" in camera) camera.fov = CAMERA_FOV_DEG;
    if ("near" in camera) camera.near = 0.01;
    if ("far" in camera) camera.far = CAMERA_DISTANCE * 16;
    camera.updateProjectionMatrix?.();
  } catch {
    /* camera flavour differences are non-fatal */
  }

  // The view's range maps to a box centered at `position` with half-extents
  // `scale`. We inset that box inside the canvas so margins remain for labels.
  const view: MathBoxNode = mathbox.cartesian({
    range: [
      [0, 1],
      [0, 1],
      [-1, 1],
    ],
    position: [0, 0, 0],
    scale: [1, 1, 1],
  });

  let axes: AxesHandle | null = null;
  const handles = new Map<string, AnySeriesNode>();
  let lastFrame: TrajectoryFrame | null = null;
  let disposed = false;

  /**
   * Inset the data box so the plot area = canvas minus `margin` (px). The
   * visible world region at z=0 is x in [-A, A], y in [-1, 1] with A = w/h, so
   * 1px = (2/h) world units uniformly. We translate margins (px) into the box
   * center + half-extents.
   */
  const applyFraming = (xRange: readonly [number, number], yRange: readonly [number, number]) => {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    const A = w / h;
    const px = 2 / h; // world units per pixel
    const hx = Math.max(1e-3, A - ((margin.left + margin.right) * px) / 2);
    const hy = Math.max(1e-3, 1 - ((margin.top + margin.bottom) * px) / 2);
    const cx = ((margin.left - margin.right) * px) / 2;
    const cy = ((margin.bottom - margin.top) * px) / 2;

    const [x0, x1] = xRange;
    const [y0, y1] = yRange;
    const xLo = Number.isFinite(x0) ? x0 : 0;
    const xHi = Number.isFinite(x1) && x1 > xLo ? x1 : xLo + 1;
    const yLo = Number.isFinite(y0) ? y0 : 0;
    const yHi = Number.isFinite(y1) && y1 > yLo ? y1 : yLo + 1;

    view.set({
      range: [
        [xLo, xHi],
        [yLo, yHi],
        [-1, 1],
      ],
      position: [cx, cy, 0],
      scale: [hx, hy, 1],
    });
  };

  const render = (frame: TrajectoryFrame) => {
    if (disposed) return;
    applyFraming(frame.xRange, frame.yRange);

    const axesUpdate = { xRange: frame.xRange, yRange: frame.yRange, spec: frame.axes ?? {} };
    if (axes) axes.update(axesUpdate);
    else axes = createAxes(view, axesUpdate);

    const seen = new Set<string>();

    for (const series of frame.series) {
      seen.add(series.id);
      const existing = handles.get(series.id);
      if (existing && existing.kind === series.kind) {
        // Types align by construction (kind matches); cast through the union.
        (existing.update as (s: typeof series) => void)(series);
        continue;
      }
      existing?.dispose();
      handles.set(
        series.id,
        series.kind === "line" ? createLineSeries(view, series) : createBandSeries(view, series),
      );
    }

    for (const ref of frame.references ?? []) {
      seen.add(ref.id);
      const existing = handles.get(ref.id);
      const refKind = ref.axis;
      if (existing && existing.kind === refKind) {
        (existing.update as (r: typeof ref) => void)(ref);
        continue;
      }
      existing?.dispose();
      handles.set(ref.id, createReference(view, ref));
    }

    for (const [id, handle] of handles) {
      if (!seen.has(id)) {
        handle.dispose();
        handles.delete(id);
      }
    }

    lastFrame = frame;
  };

  const resize = () => {
    if (disposed) return;
    const w = container.clientWidth || 0;
    const h = container.clientHeight || 0;
    if (w < 1 || h < 1) return;
    if (lastFrame) applyFraming(lastFrame.xRange, lastFrame.yRange);
    // Drive threestrap's own resize pass. Its `size` plugin binds a DOM "resize"
    // listener on `three.element` (the container); dispatching that event is
    // exactly what a real window resize does -- the plugin re-measures the
    // container next frame and emits the bootstrap "resize" event, which the
    // renderer, camera, AND MathBox context all consume. Only that path rebuilds
    // the viewport-dependent line/tick geometry; a bare renderer.setSize() does
    // not, which is why traces stayed invisible until a manual window resize.
    try {
      container.dispatchEvent(new Event("resize"));
    } catch {
      /* environments without Event constructor: best-effort no-op */
    }
  };

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    for (const handle of handles.values()) handle.dispose();
    handles.clear();
    axes?.dispose();
    axes = null;
    try {
      mathbox.select("*").remove();
    } catch {
      /* ignore */
    }
    // forceContextLoss() actually frees the WebGL context; dispose() alone does
    // not, so without it contexts leak across StrictMode/HMR remounts until the
    // browser drops the oldest ("too many active WebGL contexts").
    try {
      three.renderer?.forceContextLoss?.();
    } catch {
      /* ignore */
    }
    try {
      three.renderer?.dispose?.();
      three.destroy?.();
    } catch {
      /* ignore */
    }
    clearContainer(container);
  };

  // The panel mounts this gated on data (container starts at 0x0), so force one
  // measurement pass on the next frame to build geometry at the real size.
  if (typeof requestAnimationFrame !== "undefined") {
    requestAnimationFrame(() => resize());
  }

  return { render, resize, dispose };
}
