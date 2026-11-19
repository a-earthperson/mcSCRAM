import type { FaultTreeCanvasTheme } from "./readFaultTreeCanvasTheme";
import type { FaultTreeNode } from "./types";

export type FaultTreeGateShape =
  | "upright_triangle"
  | "upright_triangle_inscribed"
  | "upsidedown_triangle"
  | "upsidedown_triangle_inscribed"
  | "circle_cross"
  | "circle_cross_inscribed"
  | "pentagon"
  | "circle_in_circle"
  | "diamond"
  | "square"
  | "round_rect";

type GateVisual = {
  shape: FaultTreeGateShape;
  fill: (theme: FaultTreeCanvasTheme) => string;
};

const CONNECTIVE_VISUALS: Record<string, GateVisual> = {
  and: { shape: "upright_triangle", fill: (theme) => theme.fillAnd },
  nand: { shape: "upright_triangle_inscribed", fill: (theme) => theme.fillNand },
  or: { shape: "upsidedown_triangle", fill: (theme) => theme.fillOr },
  nor: { shape: "upsidedown_triangle_inscribed", fill: (theme) => theme.fillNor },
  xor: { shape: "circle_cross", fill: (theme) => theme.fillXor },
  xnor: { shape: "circle_cross_inscribed", fill: (theme) => theme.fillXnor },
  atleast: { shape: "pentagon", fill: (theme) => theme.fillAtleast },
  cardinality: { shape: "pentagon", fill: (theme) => theme.fillAtleast },
  not: { shape: "circle_in_circle", fill: (theme) => theme.fillNot },
  iff: { shape: "diamond", fill: (theme) => theme.fillXor },
  imply: { shape: "diamond", fill: (theme) => theme.fillOr },
  null: { shape: "square", fill: (theme) => theme.fillGateDefault },
};

const DEFAULT_GATE_VISUAL: GateVisual = {
  shape: "round_rect",
  fill: (theme) => theme.fillGateDefault,
};

/** Connectives that keep a caption; shape/color carry the rest. */
const LABELED_CONNECTIVES = new Set(["cardinality", "iff", "imply"]);

export function faultTreeGateVisual(node: FaultTreeNode): GateVisual {
  if (node.kind !== "gate") {
    return DEFAULT_GATE_VISUAL;
  }
  const connective = node.connective?.toLowerCase() ?? "unknown";
  return CONNECTIVE_VISUALS[connective] ?? DEFAULT_GATE_VISUAL;
}

export function isFaultTreeNullGate(node: FaultTreeNode): boolean {
  return node.kind === "gate" && node.connective?.toLowerCase() === "null";
}

/** Gate caption for types where shape alone is ambiguous; `null` otherwise. */
export function faultTreeConnectiveLabel(node: FaultTreeNode): string | null {
  if (node.kind !== "gate") return null;
  const connective = node.connective?.toLowerCase() ?? "";
  if (!LABELED_CONNECTIVES.has(connective)) return null;
  if (connective === "cardinality") {
    if (node.min && node.max) return `${node.min}-${node.max}`;
    if (node.min) return `≥${node.min}`;
    if (node.max) return `≤${node.max}`;
    return "CARD";
  }
  return connective.toUpperCase();
}

export function faultTreeDetailLabel(node: FaultTreeNode): string | null {
  if (node.kind === "gate") return faultTreeConnectiveLabel(node);
  if (node.kind === "house_event" && node.value) return node.value.toUpperCase();
  return null;
}

export function drawFaultTreeEvent(
  ctx: CanvasRenderingContext2D,
  node: FaultTreeNode,
  x: number,
  y: number,
  radius: number,
  theme: FaultTreeCanvasTheme,
): void {
  if (node.kind === "house_event") {
    ctx.fillStyle = theme.fillHouseEvent;
    fillStrokePath(ctx, () => appendHouse(ctx, x, y, radius));
    return;
  }

  ctx.fillStyle = theme.fillBasicEvent;
  if (node.flavor?.toLowerCase() === "undeveloped") {
    fillStrokePath(ctx, () => appendDiamond(ctx, x, y, radius * 2, radius * 2));
    return;
  }

  fillStrokePath(ctx, () => appendCircle(ctx, x, y, radius * 2, radius * 2));
}

export function drawFaultTreeGate(
  ctx: CanvasRenderingContext2D,
  node: FaultTreeNode,
  x: number,
  y: number,
  width: number,
  height: number,
  cornerRadius: number,
  theme: FaultTreeCanvasTheme,
): void {
  const visual = faultTreeGateVisual(node);
  ctx.fillStyle = visual.fill(theme);

  switch (visual.shape) {
    case "upright_triangle_inscribed":
      fillStrokePath(ctx, () => appendUprightTriangle(ctx, x, y, width, height));
      strokePath(ctx, () => appendInscribedTriangle(ctx, x, y, width, height, 1));
      return;
    case "upsidedown_triangle_inscribed":
      fillStrokePath(ctx, () => appendUpsidedownTriangle(ctx, x, y, width, height));
      strokePath(ctx, () => appendInscribedTriangle(ctx, x, y, width, height, -1));
      return;
    case "circle_cross":
      fillStrokePath(ctx, () => appendCircle(ctx, x, y, width, height));
      strokePath(ctx, () => appendCross(ctx, x, y, width, height));
      return;
    case "circle_cross_inscribed":
      fillStrokePath(ctx, () => appendCircle(ctx, x, y, width, height));
      strokePath(ctx, () => appendCross(ctx, x, y, width, height));
      strokePath(ctx, () => appendInnerCircle(ctx, x, y, width, height, 0.42));
      return;
    case "circle_in_circle":
      fillStrokePath(ctx, () => appendCircle(ctx, x, y, width, height));
      strokePath(ctx, () => appendInnerCircle(ctx, x, y, width, height, 0.42));
      return;
    default:
      fillStrokePath(ctx, () => appendGatePath(ctx, visual.shape, x, y, width, height, cornerRadius));
  }
}

function fillStrokePath(ctx: CanvasRenderingContext2D, build: () => void): void {
  ctx.beginPath();
  build();
  ctx.fill();
  ctx.stroke();
}

function strokePath(ctx: CanvasRenderingContext2D, build: () => void): void {
  ctx.beginPath();
  build();
  ctx.stroke();
}

function appendGatePath(
  ctx: CanvasRenderingContext2D,
  shape: FaultTreeGateShape,
  x: number,
  y: number,
  width: number,
  height: number,
  cornerRadius: number,
): void {
  switch (shape) {
    case "upright_triangle":
      appendUprightTriangle(ctx, x, y, width, height);
      break;
    case "upsidedown_triangle":
      appendUpsidedownTriangle(ctx, x, y, width, height);
      break;
    case "pentagon":
      appendRegularPolygon(ctx, x, y, Math.min(width, height) / 2, 5, -Math.PI / 2);
      break;
    case "diamond":
      appendDiamond(ctx, x, y, width, height);
      break;
    case "square":
      appendSquare(ctx, x, y, Math.min(width, height) * 0.78);
      break;
    case "round_rect":
    default:
      ctx.roundRect(x - width / 2, y - height / 2, width, height, cornerRadius);
      break;
  }
}

function appendUprightTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const halfW = width / 2;
  const halfH = height / 2;
  ctx.moveTo(x, y - halfH);
  ctx.lineTo(x + halfW, y + halfH);
  ctx.lineTo(x - halfW, y + halfH);
  ctx.closePath();
}

function appendUpsidedownTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const halfW = width / 2;
  const halfH = height / 2;
  ctx.moveTo(x, y + halfH);
  ctx.lineTo(x + halfW, y - halfH);
  ctx.lineTo(x - halfW, y - halfH);
  ctx.closePath();
}

function appendInscribedTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  direction: 1 | -1,
): void {
  const scale = 0.58;
  const halfW = (width * scale) / 2;
  const halfH = (height * scale) / 2;
  if (direction > 0) {
    ctx.moveTo(x, y - halfH);
    ctx.lineTo(x + halfW, y + halfH);
    ctx.lineTo(x - halfW, y + halfH);
  } else {
    ctx.moveTo(x, y + halfH);
    ctx.lineTo(x + halfW, y - halfH);
    ctx.lineTo(x - halfW, y - halfH);
  }
  ctx.closePath();
}

function appendCircle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  ctx.arc(x, y, Math.min(width, height) / 2, 0, Math.PI * 2);
}

function appendInnerCircle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  scale: number,
): void {
  ctx.arc(x, y, (Math.min(width, height) / 2) * scale, 0, Math.PI * 2);
}

function appendCross(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const arm = Math.min(width, height) * 0.36;
  ctx.moveTo(x - arm, y - arm);
  ctx.lineTo(x + arm, y + arm);
  ctx.moveTo(x + arm, y - arm);
  ctx.lineTo(x - arm, y + arm);
}

function appendDiamond(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const halfW = width / 2;
  const halfH = height / 2;
  ctx.moveTo(x, y - halfH);
  ctx.lineTo(x + halfW, y);
  ctx.lineTo(x, y + halfH);
  ctx.lineTo(x - halfW, y);
  ctx.closePath();
}

function appendSquare(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void {
  const half = size / 2;
  ctx.rect(x - half, y - half, size, size);
}

function appendHouse(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number): void {
  ctx.moveTo(x - radius * 0.82, y - radius * 0.05);
  ctx.lineTo(x, y - radius * 0.9);
  ctx.lineTo(x + radius * 0.82, y - radius * 0.05);
  ctx.lineTo(x + radius * 0.62, y + radius * 0.82);
  ctx.lineTo(x - radius * 0.62, y + radius * 0.82);
  ctx.closePath();
}

function appendRegularPolygon(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  sides: number,
  rotation: number,
): void {
  for (let index = 0; index < sides; index += 1) {
    const angle = rotation + (index * 2 * Math.PI) / sides;
    const px = x + radius * Math.cos(angle);
    const py = y + radius * Math.sin(angle);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}
