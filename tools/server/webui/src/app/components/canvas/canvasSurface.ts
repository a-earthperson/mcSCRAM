import { useEffect, useState, type RefObject } from "react";
import type { CanvasViewportConfig } from "./canvasViewport";
import type { CanvasSize } from "./types";

export type CanvasSurfaceAttributes = {
  size: CanvasSize;
  dpr: number;
  bufferWidth: number;
  bufferHeight: number;
};

const lastApplied = new WeakMap<HTMLCanvasElement, CanvasSurfaceAttributes>();

export function getDevicePixelRatio(): number {
  return window.devicePixelRatio || 1;
}

/** Laid-out CSS pixel size of the canvas element (flex-controlled display box). */
export function readCanvasClientSize(canvas: HTMLCanvasElement): CanvasSize | null {
  const width = Math.round(canvas.clientWidth);
  const height = Math.round(canvas.clientHeight);
  if (width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

/** Prefer live canvas client size; fall back to the last measured value. */
export function resolveCanvasFitSize(
  canvas: HTMLCanvasElement | null,
  fallback: CanvasSize,
): CanvasSize {
  if (canvas) {
    const live = readCanvasClientSize(canvas);
    if (live) {
      return live;
    }
  }
  return fallback;
}

/** Logical CSS-pixel size and device backing-store dimensions for a canvas surface. */
export function computeCanvasSurfaceAttributes(
  size: CanvasSize,
  dpr = getDevicePixelRatio(),
): CanvasSurfaceAttributes {
  return {
    size,
    dpr,
    bufferWidth: Math.max(1, Math.round(size.width * dpr)),
    bufferHeight: Math.max(1, Math.round(size.height * dpr)),
  };
}

function applyCanvasSurfaceAttributes(
  canvas: HTMLCanvasElement,
  attrs: CanvasSurfaceAttributes,
  previous: CanvasSurfaceAttributes | undefined,
): boolean {
  if (
    previous &&
    previous.bufferWidth === attrs.bufferWidth &&
    previous.bufferHeight === attrs.bufferHeight
  ) {
    return false;
  }

  if (canvas.width !== attrs.bufferWidth) {
    canvas.width = attrs.bufferWidth;
  }
  if (canvas.height !== attrs.bufferHeight) {
    canvas.height = attrs.bufferHeight;
  }
  return true;
}

/**
 * Match backing-store resolution to the canvas display box.
 * Display size comes from CSS layout; only the bitmap is written here.
 */
export function syncCanvasSurface(
  canvas: HTMLCanvasElement,
  size?: CanvasSize | null,
): { attrs: CanvasSurfaceAttributes; changed: boolean } {
  const resolved =
    size && size.width > 0 && size.height > 0 ? size : readCanvasClientSize(canvas);
  if (!resolved) {
    const empty = computeCanvasSurfaceAttributes({ width: 1, height: 1 });
    return { attrs: empty, changed: false };
  }

  const attrs = computeCanvasSurfaceAttributes(resolved);
  const previous = lastApplied.get(canvas);
  const changed = applyCanvasSurfaceAttributes(canvas, attrs, previous);
  if (changed) {
    lastApplied.set(canvas, attrs);
  }
  return { attrs, changed };
}

function sameCanvasSize(a: CanvasSize, b: CanvasSize): boolean {
  return a.width === b.width && a.height === b.height;
}

/** Observe the drawable canvas and expose its laid-out CSS pixel size. */
export function useCanvasSurfaceSize(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  _layout: CanvasViewportConfig["layout"],
): { size: CanvasSize; hasMeasured: boolean } {
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });
  const [hasMeasured, setHasMeasured] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const reportSize = () => {
      const nextSize = readCanvasClientSize(canvas);
      if (!nextSize) return;
      setHasMeasured(true);
      setSize((prev) => (sameCanvasSize(prev, nextSize) ? prev : nextSize));
    };

    reportSize();
    const observer = new ResizeObserver(reportSize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [canvasRef]);

  return { size, hasMeasured };
}
