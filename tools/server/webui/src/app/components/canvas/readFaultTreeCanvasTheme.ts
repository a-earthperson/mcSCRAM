/** Resolved canvas paint tokens from `.fault-tree-frame` custom properties. */
export type FaultTreeCanvasTheme = {
  cullPad: number;
  labelMinScale: number;
  stroke: string;
  edgeStrokeWidthMin: number;
  edgeStrokeWidthScale: number;
  edgeSourceAnchor: number;
  edgeTargetAnchor: number;
  nodeStrokeWidthMin: number;
  nodeStrokeWidthScale: number;
  gateWidth: number;
  gateHeight: number;
  basicEventRadius: number;
  gateCornerRadiusMin: number;
  gateCornerRadiusScale: number;
  fillBasicEvent: string;
  fillAnd: string;
  fillNand: string;
  fillOr: string;
  fillNor: string;
  fillXor: string;
  fillXnor: string;
  fillAtleast: string;
  fillNot: string;
  fillHouseEvent: string;
  fillGateDefault: string;
  labelColor: string;
  labelFontFamily: string;
  labelFontSize: number;
  labelFontSizeMin: number;
  labelOffsetY: number;
  labelMaxWidthMin: number;
  labelMaxWidthScale: number;
  connectiveColor: string;
  connectiveFontFamily: string;
  connectiveFontWeight: number;
  connectiveFontSize: number;
  connectiveFontSizeMin: number;
  connectiveOffsetY: number;
};

let cachedRoot: HTMLElement | null = null;
let cachedTheme: FaultTreeCanvasTheme | null = null;

function readCssNumber(style: CSSStyleDeclaration, name: string, fallback: number): number {
  const raw = style.getPropertyValue(name).trim();
  if (!raw) return fallback;
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : fallback;
}

function readCssString(style: CSSStyleDeclaration, name: string, fallback: string): string {
  const raw = style.getPropertyValue(name).trim();
  return raw || fallback;
}

/** Read canvas tokens from `themeRoot` (expected: `.fault-tree-frame`). */
export function readFaultTreeCanvasTheme(themeRoot: HTMLElement): FaultTreeCanvasTheme {
  if (cachedRoot === themeRoot && cachedTheme) return cachedTheme;

  const style = getComputedStyle(themeRoot);
  const theme: FaultTreeCanvasTheme = {
    cullPad: readCssNumber(style, "--ft-cull-pad", 120),
    labelMinScale: readCssNumber(style, "--ft-label-min-scale", 0.38),
    stroke: readCssString(style, "--ft-stroke", "#22201e"),
    edgeStrokeWidthMin: readCssNumber(style, "--ft-edge-stroke-width-min", 0.5),
    edgeStrokeWidthScale: readCssNumber(style, "--ft-edge-stroke-width-scale", 1.1),
    edgeSourceAnchor: readCssNumber(style, "--ft-edge-source-anchor", 32),
    edgeTargetAnchor: readCssNumber(style, "--ft-edge-target-anchor", 34),
    nodeStrokeWidthMin: readCssNumber(style, "--ft-node-stroke-width-min", 0.8),
    nodeStrokeWidthScale: readCssNumber(style, "--ft-node-stroke-width-scale", 1.1),
    gateWidth: readCssNumber(style, "--ft-gate-width", 84),
    gateHeight: readCssNumber(style, "--ft-gate-height", 56),
    basicEventRadius: readCssNumber(style, "--ft-basic-event-radius", 28),
    gateCornerRadiusMin: readCssNumber(style, "--ft-gate-corner-radius-min", 2),
    gateCornerRadiusScale: readCssNumber(style, "--ft-gate-corner-radius-scale", 3),
    fillBasicEvent: readCssString(style, "--ft-fill-basic-event", "rgba(179, 152, 102, 0.18)"),
    fillAnd: readCssString(style, "--ft-fill-and", "rgba(107, 127, 150, 0.16)"),
    fillNand: readCssString(style, "--ft-fill-nand", "rgba(107, 127, 150, 0.66)"),
    fillOr: readCssString(style, "--ft-fill-or", "rgba(168, 128, 122, 0.16)"),
    fillNor: readCssString(style, "--ft-fill-nor", "rgba(168, 128, 122, 0.66)"),
    fillXor: readCssString(style, "--ft-fill-xor", "rgba(139, 161, 137, 0.16)"),
    fillXnor: readCssString(style, "--ft-fill-xnor", "rgba(139, 161, 137, 0.66)"),
    fillAtleast: readCssString(style, "--ft-fill-atleast", "rgba(139, 161, 137, 0.18)"),
    fillNot: readCssString(style, "--ft-fill-not", "rgba(139, 161, 137, 0.66)"),
    fillHouseEvent: readCssString(style, "--ft-fill-house-event", "rgba(179, 152, 102, 0.42)"),
    fillGateDefault: readCssString(style, "--ft-fill-gate-default", "rgba(250, 247, 243, 0.72)"),
    labelColor: readCssString(style, "--ft-label-color", "#22201e"),
    labelFontFamily: readCssString(style, "--ft-label-font-family", "IBM Plex Mono, ui-monospace, monospace"),
    labelFontSize: readCssNumber(style, "--ft-label-font-size", 12),
    labelFontSizeMin: readCssNumber(style, "--ft-label-font-size-min", 8),
    labelOffsetY: readCssNumber(style, "--ft-label-offset-y", -5),
    labelMaxWidthMin: readCssNumber(style, "--ft-label-max-width-min", 42),
    labelMaxWidthScale: readCssNumber(style, "--ft-label-max-width-scale", 72),
    connectiveColor: readCssString(style, "--ft-connective-color", "#58524d"),
    connectiveFontFamily: readCssString(style, "--ft-connective-font-family", "Inter Tight, sans-serif"),
    connectiveFontWeight: readCssNumber(style, "--ft-connective-font-weight", 700),
    connectiveFontSize: readCssNumber(style, "--ft-connective-font-size", 9),
    connectiveFontSizeMin: readCssNumber(style, "--ft-connective-font-size-min", 7),
    connectiveOffsetY: readCssNumber(style, "--ft-connective-offset-y", 13),
  };

  cachedRoot = themeRoot;
  cachedTheme = theme;
  return theme;
}

export function invalidateFaultTreeCanvasThemeCache(): void {
  cachedRoot = null;
  cachedTheme = null;
}
