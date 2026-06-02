import {
  drawFaultTreeEvent,
  drawFaultTreeGate,
  faultTreeDetailLabel,
  isFaultTreeNullGate,
} from "./faultTreeGlyphs";
import { readFaultTreeCanvasTheme, type FaultTreeCanvasTheme } from "./readFaultTreeCanvasTheme";
import type { CanvasSize, FaultTreeLayout, RenderNode, ViewTransform } from "./types";

export function drawFaultTreeCanvas(
  canvas: HTMLCanvasElement,
  layout: FaultTreeLayout,
  view: ViewTransform,
  size: CanvasSize,
  themeRoot: HTMLElement,
): void {
  const theme = readFaultTreeCanvasTheme(themeRoot);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(size.width * dpr);
  canvas.height = Math.round(size.height * dpr);
  canvas.style.width = `${size.width}px`;
  canvas.style.height = `${size.height}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size.width, size.height);

  const { scale, offsetX, offsetY } = view;
  const pad = theme.cullPad;
  const wx0 = (-offsetX / scale) - pad;
  const wy0 = (-offsetY / scale) - pad;
  const wx1 = ((size.width - offsetX) / scale) + pad;
  const wy1 = ((size.height - offsetY) / scale) + pad;
  const toScreenX = (x: number) => x * scale + offsetX;
  const toScreenY = (y: number) => y * scale + offsetY;
  const visible = (node: RenderNode) => node.x >= wx0 && node.x <= wx1 && node.y >= wy0 && node.y <= wy1;

  ctx.lineWidth = Math.max(theme.edgeStrokeWidthMin, scale * theme.edgeStrokeWidthScale);
  ctx.strokeStyle = theme.stroke;
  for (const edge of layout.edges) {
    const source = layout.nodes[edge.source];
    const target = layout.nodes[edge.target];
    if (!source || !target) continue;
    if (!visible(source) && !visible(target)) continue;
    const sourceAnchor = isFaultTreeNullGate(source.node) ? 0 : theme.edgeSourceAnchor;
    const targetAnchor = isFaultTreeNullGate(target.node) ? 0 : theme.edgeTargetAnchor;
    const sx = toScreenX(source.x);
    const sy = toScreenY(source.y + sourceAnchor);
    const tx = toScreenX(target.x);
    const ty = toScreenY(target.y - targetAnchor);
    const midY = (sy + ty) / 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.bezierCurveTo(sx, midY, tx, midY, tx, ty);
    ctx.stroke();
  }

  const labelVisible = scale > theme.labelMinScale;
  for (const node of layout.nodes) {
    if (!visible(node)) continue;
    drawFaultTreeNode(ctx, node, view, labelVisible, theme);
  }
}

function scaledPx(base: number, min: number, scale: number): number {
  return Math.max(min, base * scale);
}

function drawFaultTreeNode(
  ctx: CanvasRenderingContext2D,
  renderNode: RenderNode,
  view: ViewTransform,
  labelVisible: boolean,
  theme: FaultTreeCanvasTheme,
): void {
  const { node } = renderNode;
  const { scale } = view;
  const x = renderNode.x * scale + view.offsetX;
  const y = renderNode.y * scale + view.offsetY;
  const gateW = theme.gateWidth * scale;
  const gateH = theme.gateHeight * scale;
  const radius = theme.basicEventRadius * scale;
  const cornerRadius = Math.max(theme.gateCornerRadiusMin, theme.gateCornerRadiusScale * scale);

  ctx.lineWidth = Math.max(theme.nodeStrokeWidthMin, scale * theme.nodeStrokeWidthScale);
  ctx.strokeStyle = theme.stroke;
  if (node.kind === "gate") {
    drawFaultTreeGate(ctx, node, x, y, gateW, gateH, cornerRadius, theme);
  } else {
    drawFaultTreeEvent(ctx, node, x, y, radius, theme);
  }

  if (!labelVisible) return;
  const maxWidth = scaledPx(theme.labelMaxWidthScale, theme.labelMaxWidthMin, scale);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = theme.labelColor;
  ctx.font = `${scaledPx(theme.labelFontSize, theme.labelFontSizeMin, scale)}px ${theme.labelFontFamily}`;
  ctx.fillText(node.label, x, y + theme.labelOffsetY * scale, maxWidth);

  const detailLabel = faultTreeDetailLabel(node);
  if (!detailLabel) return;
  ctx.fillStyle = theme.connectiveColor;
  ctx.font = `${theme.connectiveFontWeight} ${scaledPx(theme.connectiveFontSize, theme.connectiveFontSizeMin, scale)}px ${theme.connectiveFontFamily}`;
  ctx.fillText(detailLabel, x, y + theme.connectiveOffsetY * scale, maxWidth);
}
