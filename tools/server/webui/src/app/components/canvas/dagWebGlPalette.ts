export type Rgba = [number, number, number, number];

export enum DagWebGlShape {
  circle = "circle",
  upright_triangle = "upright_triangle",
  upsidedown_triangle = "upsidedown_triangle",
  circle_with_cross = "circle_with_cross",
  pentagon = "pentagon",
  circle_inside_circle = "circle_inside_circle",
  upright_triangle_inscribed = "upright_triangle_inscribed",
  upsidedown_triangle_inscribed = "upsidedown_triangle_inscribed",
  circle_with_cross_inscribed = "circle_with_cross_inscribed",
  house = "house",
  square = "square",
}

/*
 * Palette for the WebGL DAG plate - authored in OKLCH, baked to sRGB.
 * ------------------------------------------------------------------
 * Colors are written straight to the framebuffer (no in-shader
 * linearization), so the floats below are sRGB-encoded 0..1.
 *
 * This is the design language pushed into a fluorescent specimen-ink
 * register: botanical / mineral pigments under UV instead of quiet
 * watercolor wash. The hues still orbit the archive palette (ochre,
 * indigo, madder, sage, paynes, amber), but the fill is only a thin
 * translucent stain. The fluorescent stroke carries the signal, so the
 * glyphs read as neon specimen outlines instead of solid colored icons.
 *
 *   role           fill oklch / hex           stroke oklch / hex        fluorescent pigment
 *   basic_event    0.820 0.155 108 / #cecb40  0.700 0.150 108 / #a8a405  acid yellow ochre
 *   and / nand     0.720 0.155 238 / #19b0f9  0.680 0.150 238 / #05a3e9  cyanotype indigo
 *   or  / nor      0.700 0.180 18  / #fa6773  0.650 0.180 18  / #e85664  rhodamine madder
 *   xor / xnor     0.760 0.170 148 / #57ce70  0.680 0.170 148 / #39b457  electric viridian
 *   atleast        0.700 0.180 305 / #b97df7  0.650 0.180 305 / #a96de6  ultraviolet paynes
 *   not            0.780 0.160 68  / #f9a129  0.700 0.150 68  / #da8a11  amber resin
 *   constant_event 0.780 0.120 332 / #e49ada  0.660 0.130 332 / #c072b5  fluorescent fuchsia (hot rose)
 *   edge           0.620 0.095 220 /rgb(34, 63, 71)  -                         blue-green graph line
 *   edge_compl     0.700 0.150 68  / #da8a11  -                         amber complement (signed arg)
 */

export const DAG_WEBGL_NODE_RADIUS = 0.333 * (window?.devicePixelRatio ?? 1);

/** World-space glyph shift (× node radius). Graph (x, y) and edges stay put. */
export const DAG_WEBGL_NODE_DISPLAY_OFFSET: Record<DagWebGlShape, { x: number; y: number }> = {
  [DagWebGlShape.upright_triangle] : {x: 0, y: DAG_WEBGL_NODE_RADIUS * -0.25},
  [DagWebGlShape.upright_triangle_inscribed]: {x: 0, y: DAG_WEBGL_NODE_RADIUS * -0.25},
  [DagWebGlShape.upsidedown_triangle]: {x: 0, y: DAG_WEBGL_NODE_RADIUS * 0.333},
  [DagWebGlShape.upsidedown_triangle_inscribed]: {x: 0, y: DAG_WEBGL_NODE_RADIUS * 0.333},
  [DagWebGlShape.circle]  : {x: 0, y: DAG_WEBGL_NODE_RADIUS *0},
  [DagWebGlShape.circle_inside_circle]: {x: 0, y: DAG_WEBGL_NODE_RADIUS *0},
  [DagWebGlShape.circle_with_cross]: {x: 0, y: DAG_WEBGL_NODE_RADIUS *0},
  [DagWebGlShape.circle_with_cross_inscribed]: {x: 0, y: DAG_WEBGL_NODE_RADIUS *0},
  [DagWebGlShape.pentagon]: {x: 0, y: DAG_WEBGL_NODE_RADIUS *0.0},
  [DagWebGlShape.house]: {x: 0, y: DAG_WEBGL_NODE_RADIUS * -0.125},
  [DagWebGlShape.square]: {x: 0, y: DAG_WEBGL_NODE_RADIUS *0.0},
}

export const DAG_WEBGL_EDGE_COLOR: Rgba = [0.303, 0.3235, 0.321568, 0.85];

/** Signed PDAG argument edges (screen-space quads, thicker than regular edges). */
export const DAG_WEBGL_COMPLEMENTED_EDGE_COLOR: Rgba = [0.856043, 0.2482, 0.67185, 0.88];

/** Half-width in screen pixels for complemented edge quads. */
export const DAG_WEBGL_COMPLEMENTED_EDGE_WIDTH_PX = 2.75;

export const DAG_WEBGL_STROKE_NONE: Rgba = [0.599156, 0.595092, 0.810668, 0.9];

export const DAG_WEBGL_FILL_NONE: Rgba = [1.0, 0, 0, 1.0];

export const DAG_WEBGL_FILLS = {
  basic_event: [0.808844, 0.795639, 0.249941, 0.20] as Rgba,
  and_gate: [0.096146, 0.691099, 0.977711, 0.2] as Rgba,
  nand_gate: [0.096146, 0.691099, 0.977711, 0.2] as Rgba,
  or_gate: [0.979533, 0.402418, 0.450999, 0.2] as Rgba,
  nor_gate: [0.979533, 0.402418, 0.450999, 0.2] as Rgba,
  xor_gate: [0.340127, 0.806187, 0.438209, 0.2] as Rgba,
  xnor_gate: [0.340127, 0.806187, 0.438209, 0.2] as Rgba,
  atleast_gate: [0.724892, 0.488624, 0.967278, 0.2] as Rgba,
  not_gate: [0.977485, 0.631504, 0.160365, 0.2] as Rgba,
  constant_event: [0.895947, 0.604909, 0.353011, 0.20] as Rgba,
} as const;

export const DAG_WEBGL_STROKES = {
  basic_event: [0.659408, 0.644099, 0.019619, 1.0] as Rgba,
  and_gate: [0.01793, 0.640085, 0.913893, 1.0] as Rgba,
  nand_gate: [0.01793, 0.640085, 0.913893, 1.0] as Rgba,
  or_gate: [0.909049, 0.338076, 0.393436, 1.0] as Rgba,
  nor_gate: [0.909049, 0.338076, 0.393436, 1.0] as Rgba,
  xor_gate: [0.222174, 0.704073, 0.341718, 1.0] as Rgba,
  xnor_gate: [0.222174, 0.704073, 0.341718, 1.0] as Rgba,
  atleast_gate: [0.663468, 0.427058, 0.90006, 1.0] as Rgba,
  not_gate: [0.856043, 0.539825, 0.067185, 1.0] as Rgba,
  constant_event: [0.752398, 0.448841, 0.311252, 1.0] as Rgba,
} as const;

export type DagWebGlFillKey = keyof typeof DAG_WEBGL_FILLS | "none";
export type DagWebGlStrokeKey = keyof typeof DAG_WEBGL_STROKES | "none";

export function dagWebGlFill(key: DagWebGlFillKey): Rgba {
  if (key === "none") return DAG_WEBGL_FILL_NONE;
  return DAG_WEBGL_FILLS[key];
}

export function dagWebGlStroke(key: DagWebGlStrokeKey): Rgba {
  if (key === "none") return DAG_WEBGL_STROKE_NONE;
  return DAG_WEBGL_STROKES[key];
}
