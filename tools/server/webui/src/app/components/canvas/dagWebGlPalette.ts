export type Rgba = [number, number, number, number];

export const DAG_WEBGL_NODE_RADIUS = 0.666;

export const DAG_WEBGL_EDGE_COLOR: Rgba = [0.345098, 0.321569, 0.301961, 0.8];

export const DAG_WEBGL_STROKE_COLOR: Rgba = [0.133333, 0.12549, 0.117647, 0.42];

export const DAG_WEBGL_FILL_NONE: Rgba = [1.0, 0, 0, 1.0];

export const DAG_WEBGL_FILLS = {
  basic_event: [0.701961, 0.596078, 0.4, 1.0] as Rgba,
  and_gate: [0.419608, 0.498039, 0.588235, 1.0] as Rgba,
  nand_gate: [0.419608, 0.498039, 0.588235, 1.0] as Rgba,
  or_gate: [0.658824, 0.501961, 0.478431, 1.0] as Rgba,
  nor_gate: [0.658824, 0.501961, 0.478431, 1.0] as Rgba,
  xor_gate: [0.545098, 0.631373, 0.537255, 1.0] as Rgba,
  xnor_gate: [0.545098, 0.631373, 0.537255, 1.0] as Rgba,
  atleast_gate: [0.545098, 0.631373, 0.537255, 1.0] as Rgba,
  not_gate: [0.545098, 0.631373, 0.537255, 1.0] as Rgba,
} as const;

export type DagWebGlFillKey = keyof typeof DAG_WEBGL_FILLS | "none";

export function dagWebGlFill(key: DagWebGlFillKey): Rgba {
  if (key === "none") return DAG_WEBGL_FILL_NONE;
  return DAG_WEBGL_FILLS[key];
}
