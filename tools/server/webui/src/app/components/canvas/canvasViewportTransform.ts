import type { CanvasViewportConfig } from "./canvasViewport";
import type { CanvasSize, DagBounds, ViewTransform } from "./types";

export function clampViewScale(
  scale: number,
  config: CanvasViewportConfig,
): number {
  return Math.max(config.scale.min, Math.min(config.scale.max, scale));
}

export function wheelZoomFactor(
  deltaY: number,
  config: CanvasViewportConfig,
): number {
  return Math.exp(-deltaY * config.wheel.sensitivity);
}

export function panViewTransform(
  view: ViewTransform,
  dx: number,
  dy: number,
): ViewTransform {
  return { ...view, offsetX: view.offsetX + dx, offsetY: view.offsetY + dy };
}

export function wheelZoomAtCanvasPoint(
  view: ViewTransform,
  canvas: HTMLCanvasElement,
  event: WheelEvent,
  config: CanvasViewportConfig,
): ViewTransform {
  const rect = canvas.getBoundingClientRect();
  return zoomAtPoint(
    view,
    event.clientX - rect.left,
    event.clientY - rect.top,
    wheelZoomFactor(event.deltaY, config),
    config,
  );
}

export function pinchZoomAtCanvasMidpoint(
  view: ViewTransform,
  canvas: HTMLCanvasElement,
  clientX0: number,
  clientY0: number,
  clientX1: number,
  clientY1: number,
  distanceRatio: number,
  config: CanvasViewportConfig,
): ViewTransform {
  const rect = canvas.getBoundingClientRect();
  const px = (clientX0 + clientX1) / 2 - rect.left;
  const py = (clientY0 + clientY1) / 2 - rect.top;
  return zoomAtPoint(view, px, py, distanceRatio, config);
}

export function zoomAtPoint(
  view: ViewTransform,
  px: number,
  py: number,
  factor: number,
  config: CanvasViewportConfig,
): ViewTransform {
  const nextScale = clampViewScale(view.scale * factor, config);
  const worldX = (px - view.offsetX) / view.scale;
  const worldY = (py - view.offsetY) / view.scale;
  return {
    scale: nextScale,
    offsetX: px - worldX * nextScale,
    offsetY: py - worldY * nextScale,
  };
}

export function resetViewTransform(
  size: CanvasSize,
  config: CanvasViewportConfig,
): ViewTransform {
  const offsetX = config.reset.offsetX === "center" ? size.width / 2 : config.reset.offsetX;
  return {
    scale: config.reset.scale,
    offsetX,
    offsetY: config.reset.offsetY,
  };
}

function fitContentScale(
  size: CanvasSize,
  contentWidth: number,
  contentHeight: number,
  config: CanvasViewportConfig,
): number {
  const scaleX = size.width / Math.max(contentWidth, 1);
  const scaleY = size.height / Math.max(contentHeight, 1);
  return clampViewScale(Math.min(scaleX, scaleY) * config.fit.scaleMargin, config);
}

export function fitBoundsView(
  bounds: DagBounds,
  size: CanvasSize,
  config: CanvasViewportConfig,
): ViewTransform {
  const paddingWorld = config.fit.boundsPaddingWorld ?? 0;
  const paddedWidth = Math.max(bounds.width + paddingWorld * 2, 1);
  const paddedHeight = Math.max(bounds.height + paddingWorld * 2, 1);
  const scale = fitContentScale(size, paddedWidth, paddedHeight, config);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  return {
    scale,
    offsetX: size.width / 2 - centerX * scale,
    offsetY: size.height / 2 - centerY * scale,
  };
}

export function fitLayoutView(
  layoutWidth: number,
  layoutHeight: number,
  size: CanvasSize,
  config: CanvasViewportConfig,
): ViewTransform {
  const edgePaddingPx = config.fit.edgePaddingPx ?? 0;
  let scale = fitContentScale(size, layoutWidth, layoutHeight, config);
  if (config.fit.maxFitScale !== undefined) {
    scale = Math.min(scale, config.fit.maxFitScale);
  }
  return {
    scale,
    offsetX: Math.max(edgePaddingPx, (size.width - layoutWidth * scale) / 2),
    offsetY: Math.max(edgePaddingPx, (size.height - layoutHeight * scale) / 2),
  };
}
