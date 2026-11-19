import type { CanvasSize } from "./types";

export type CanvasViewportConfig = {
  scale: {
    min: number;
    max: number;
  };
  wheel: {
    /** Wheel delta is mapped as `exp(-deltaY * sensitivity)`. */
    sensitivity: number;
  };
  button: {
    zoomInFactor: number;
    zoomOutFactor: number;
    panStepPx: number;
  };
  fit: {
    scaleMargin: number;
    /** World-space padding on each side when fitting axis-aligned bounds (WebGL DAG). */
    boundsPaddingWorld?: number;
    /** Min screen-space inset when fitting a layout width/height box (2D fault tree). */
    edgePaddingPx?: number;
    /** Upper cap on scale from fit-to-content (2D fault tree). */
    maxFitScale?: number;
  };
  reset: {
    scale: number;
    offsetX: number | "center";
    offsetY: number;
  };
  layout: {
    defaultSize: CanvasSize;
    minWidth: number;
    minHeight: number;
  };
};

const SHARED_CANVAS_VIEWPORT = {
  wheel: { sensitivity: 0.0012 },
  button: { zoomInFactor: 1.25, zoomOutFactor: 0.8, panStepPx: 90 },
  fit: { scaleMargin: 0.92 },
  layout: {
    defaultSize: { width: 900, height: 620 },
    minWidth: 320,
    minHeight: 420,
  },
} as const satisfies Pick<CanvasViewportConfig, "wheel" | "button" | "fit" | "layout">;

/** WebGL DAG: world-space bounds, wide scale range, centered default X. */
export const DAG_CANVAS_VIEWPORT: CanvasViewportConfig = {
  ...SHARED_CANVAS_VIEWPORT,
  scale: { min: 1e-6, max: 512 },
  fit: { ...SHARED_CANVAS_VIEWPORT.fit, boundsPaddingWorld: 1.2 },
  reset: { scale: 64, offsetX: "center", offsetY: 80 },
};

/** 2D fault tree: layout box in screen space, tighter zoom range, fixed reset offsets. */
export const FAULT_TREE_CANVAS_VIEWPORT: CanvasViewportConfig = {
  ...SHARED_CANVAS_VIEWPORT,
  scale: { min: 0.001, max: 2.5 },
  fit: { ...SHARED_CANVAS_VIEWPORT.fit, edgePaddingPx: 24, maxFitScale: 1.4 },
  reset: { scale: 1, offsetX: 72, offsetY: 72 },
};
