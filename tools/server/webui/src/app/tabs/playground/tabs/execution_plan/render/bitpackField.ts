/**
 * Bitpack point field — the base data layer for the execution scene.
 *
 * One `bitpack_t` is drawn as a `bitsPerSide x bitsPerSide` grid of points (64
 * bits -> 8x8); one such tile is one work-item output. The field renders a small
 * `columns x rows` *window* of that lattice. The heavy lifting stays on the GPU:
 * a single seed point is resampled into a `bitsPerSide^2` block, `repeat`-ed
 * across the window, and a pluggable {@link BitSource} assigns each point its
 * trial bit. Bits are lifted to a signed Z displacement and drawn as points plus
 * displacement vectors.
 *
 * This module owns *all* MathBox/GLSL semantics for the data layer. Callers pass
 * a plain {@link BitpackFieldConfig} (counts + pitches + colours + bit source)
 * and receive {@link BitpackFieldHandles} — ids to reference and a
 * {@link FieldFrame} describing the field's world geometry, which is the single
 * source of truth the overlays reuse so everything lines up exactly.
 */
import * as THREE from "three";
import type { MathBoxView } from "./types";
import { type BitColors } from "./fieldStyle";
import { type BitSource, proceduralBitSource } from "./bitSource";

export const BITPACK_SIZE = 64;

const DEFAULT_BIT_POINT_SIZE = 10;
const DEFAULT_BIT_VECTOR_WIDTH = 2;
const DEFAULT_BIT_PITCH = 0.02;
const DEFAULT_TILE_PITCH = 0.18;

const DEFAULT_BIT_COLORS: BitColors = {
  base: "#000",
  set: "#ff4633",
  unset: "#fff14f",
};

export type BitpackFieldConfig = {
  /** Tiles along X (samples / bitpack columns). */
  columns: number;
  /** Tiles along Y (events / node rows). */
  rows: number;
  /** sqrt(bitsPerBitpack); 64-bit packs are 8. */
  bitsPerSide: number;
  /** Point-to-point spacing within one tile, in field units. */
  bitPitch: number;
  /** Distance between adjacent tile origins, in field units. */
  tilePitch: number;
  /** Screen-space bit point diameter in CSS pixels. */
  pointSize?: number;
  /** Screen-space displacement-vector stroke width in CSS pixels. */
  vectorWidth?: number;
  setBitZDisplacement: number;
  unsetBitZDisplacement: number;
  colors?: BitColors;
  /** Bit assignment strategy; defaults to the procedural hash (seed 0). */
  bitSource?: BitSource;
  /** Draw the per-bit displacement vectors. Disable for a flat/empty field. */
  showDisplacementVectors?: boolean;
  /** Padding tile columns per side; these fade to transparent toward the edge. */
  paddingCols?: number;
  /** Padding tile rows per side; these fade to transparent toward the edge. */
  paddingRows?: number;
  /** Exponential decay constant (per padding tile) along +X/-X. */
  paddingFalloffX?: number;
  /** Exponential decay constant (per padding tile) along +Y/-Y. */
  paddingFalloffY?: number;
  /**
   * @deprecated Use `paddingFalloffX`/`paddingFalloffY`.
   * Shared exponential decay constant applied to both axes.
   */
  paddingFalloff?: number;
  /** Namespace for node ids so multiple fields can coexist in one view. */
  idPrefix?: string;
};

export type FieldExtent = {
  width: number;
  height: number;
  maxSpan: number;
};

export type BitpackFieldHandles = {
  frame: FieldFrame;
  extent: FieldExtent;
  /** Id of the laid-out surface (points geometry source). */
  surfaceId: string;
  /** Id of the per-point colour source. */
  colorsId: string;
};

// ---------------------------------------------------------------------------
//  FieldFrame — world geometry, mirroring BITPACK_TILE_SHADER exactly.
// ---------------------------------------------------------------------------

function axisSpan(tiles: number, bitsPerSide: number, bitPitch: number, tilePitch: number): number {
  return Math.max(bitPitch, (tiles - 1) * tilePitch + (bitsPerSide - 1) * bitPitch);
}

/** Half-open world interval. */
export type WorldRect = { x0: number; x1: number; y0: number; y1: number };

/**
 * World-space geometry of a bitpack field. All tile indices are *field-local*
 * (`0..columns-1`, `0..rows-1`); callers translate absolute data-layer windows
 * into field-local space before asking for rectangles.
 *
 * The field is centred on the origin (matching the tile shader's `point - span/2`),
 * tiles are spaced by `tilePitch`, and a tile's lattice cell is `tilePitch` wide
 * so adjacent cells tile seamlessly into a work-item lattice.
 */
export class FieldFrame {
  readonly columns: number;
  readonly rows: number;
  readonly bitsPerSide: number;
  readonly bitPitch: number;
  readonly tilePitch: number;
  readonly spanX: number;
  readonly spanY: number;

  constructor(columns: number, rows: number, bitsPerSide: number, bitPitch: number, tilePitch: number) {
    this.columns = columns;
    this.rows = rows;
    this.bitsPerSide = bitsPerSide;
    this.bitPitch = bitPitch;
    this.tilePitch = tilePitch;
    this.spanX = axisSpan(columns, bitsPerSide, bitPitch, tilePitch);
    this.spanY = axisSpan(rows, bitsPerSide, bitPitch, tilePitch);
  }

  /** World X of the centre of tile column `col`. */
  tileCenterX(col: number): number {
    return col * this.tilePitch + ((this.bitsPerSide - 1) * this.bitPitch) / 2 - this.spanX / 2;
  }

  /** World Y of the centre of tile row `row`. */
  tileCenterY(row: number): number {
    return row * this.tilePitch + ((this.bitsPerSide - 1) * this.bitPitch) / 2 - this.spanY / 2;
  }

  private get cellHalf(): number {
    return this.tilePitch / 2;
  }

  /** World rectangle enclosing a contiguous block of tiles (field-local indices). */
  regionRect(colStart: number, colCount: number, rowStart: number, rowCount: number): WorldRect {
    const cEnd = colStart + Math.max(1, colCount) - 1;
    const rEnd = rowStart + Math.max(1, rowCount) - 1;
    return {
      x0: this.tileCenterX(colStart) - this.cellHalf,
      x1: this.tileCenterX(cEnd) + this.cellHalf,
      y0: this.tileCenterY(rowStart) - this.cellHalf,
      y1: this.tileCenterY(rEnd) + this.cellHalf,
    };
  }

  /** World rectangle of the whole field. */
  fullRect(): WorldRect {
    return this.regionRect(0, this.columns, 0, this.rows);
  }

  /** Cell-boundary range for a `divideX = columns` lattice grid. */
  gridRangeX(): [number, number] {
    return [this.tileCenterX(0) - this.cellHalf, this.tileCenterX(this.columns - 1) + this.cellHalf];
  }

  gridRangeY(): [number, number] {
    return [this.tileCenterY(0) - this.cellHalf, this.tileCenterY(this.rows - 1) + this.cellHalf];
  }

  extent(): FieldExtent {
    return { width: this.spanX, height: this.spanY, maxSpan: Math.max(this.spanX, this.spanY) };
  }
}

export function bitpackFieldFrame(config: Pick<BitpackFieldConfig, "columns" | "rows" | "bitsPerSide" | "bitPitch" | "tilePitch">): FieldFrame {
  return new FieldFrame(
    positiveInteger(config.columns),
    positiveInteger(config.rows),
    positiveInteger(config.bitsPerSide),
    positiveFinite(config.bitPitch, DEFAULT_BIT_PITCH),
    positiveFinite(config.tilePitch, DEFAULT_TILE_PITCH),
  );
}

export function bitpackFieldWorldExtent(config: BitpackFieldConfig): FieldExtent {
  return bitpackFieldFrame(config).extent();
}

// ---------------------------------------------------------------------------
//  Shader chain (unchanged semantics; the bit-assignment slot is pluggable).
// ---------------------------------------------------------------------------

function colorLiteral(color: string): string {
  const c = new THREE.Color(color);
  return `vec4(${c.r.toFixed(8)}, ${c.g.toFixed(8)}, ${c.b.toFixed(8)}, 1.0)`;
}

function bitZColorShader(colors: BitColors): string {
  return `
uniform float setBitZDisplacement;
uniform float unsetBitZDisplacement;

vec4 getSample(vec2 xy);

vec4 getFramesSample(vec2 xy) {
  vec4 c = getSample(xy);
  float setWeight = clamp(max(c.z, 0.0) / max(abs(setBitZDisplacement), 0.000001), 0.0, 1.0);
  float unsetWeight = clamp(max(-c.z, 0.0) / max(abs(unsetBitZDisplacement), 0.000001), 0.0, 1.0);
  float baseWeight = 1.0 - max(setWeight, unsetWeight);

  // c.w carries the per-point padding fade (1 in the core, ->0 at the edges).
  float fade = clamp(c.w, 0.0, 1.0);

  vec4 color = ${colorLiteral(colors.base)} * baseWeight
    + ${colorLiteral(colors.unset)} * unsetWeight
    + ${colorLiteral(colors.set)} * setWeight;

  return vec4(color.rgb, color.a * fade);
}
`;
}

const BIT_DISPLACEMENT_VECTOR_SHADER = `
vec4 getSample(vec4 xyzw);

vec4 getFramesSample(vec4 xyzw) {
  vec4 c = getSample(vec4(xyzw.xyz, 0.0));
  float endpoint = step(0.5, xyzw.w);
  return vec4(c.xy, c.z * endpoint, c.a);
}
`;

function bitDisplacementVectorColorShader(colors: BitColors): string {
  return `
vec4 getSample(vec4 xyzw);

vec4 getFramesSample(vec4 xyzw) {
  vec4 c = getSample(vec4(xyzw.xyz, 0.0));
  float bit = step(0.0, c.z);
  float endpoint = step(0.5, xyzw.w);
  vec4 bitColor = mix(${colorLiteral(colors.unset)}, ${colorLiteral(colors.set)}, bit);
  return mix(${colorLiteral(colors.base)}, bitColor, endpoint);
}
`;
}

const BITPACK_8X8_SHADER = `
vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  return vec4(floor(xyz.x + 0.5), floor(xyz.y + 0.5), 0.0, 0.0);
}
`;

const BIT_Z_DISPLACEMENT_SHADER = `
uniform float setBitZDisplacement;
uniform float unsetBitZDisplacement;

vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  vec4 c = getSample(xyz);
  float bit = step(0.5, c.z);
  float z = mix(unsetBitZDisplacement, setBitZDisplacement, bit);
  return vec4(c.xy, z, c.a);
}
`;

const BITPACK_TILE_SHADER = `
uniform vec4 fieldShape;
uniform vec2 fieldSpacing;
uniform vec4 fieldPadding; // [padCols, padRows, falloffX, falloffY]

vec4 getSample(vec3 xyz);

float paddingAttenuation(vec2 centered, vec2 coreHalfSpan, vec2 falloffPerTile, float tilePitch) {
  // Distance outside the core perimeter (0 inside), measured in tile units.
  vec2 outsideTiles = max(abs(centered) - coreHalfSpan, vec2(0.0)) / max(tilePitch, 0.000001);
  // Anisotropic exponential decay using hypot-like combination at corners.
  return length(outsideTiles * max(falloffPerTile, vec2(0.0)));
}

vec4 getFramesSample(vec3 xyz) {
  float bitsPerSide = fieldShape.z;
  float bitPitch = fieldSpacing.x;
  float tilePitch = fieldSpacing.y;

  vec2 repeatedIndex = floor(xyz.xy + vec2(0.5));
  vec2 tile = floor(repeatedIndex / bitsPerSide);
  vec4 sample = getSample(xyz);
  vec2 local = sample.xy;

  vec2 point = vec2(
    tile.x * tilePitch + local.x * bitPitch,
    tile.y * tilePitch + local.y * bitPitch
  );
  vec2 span = vec2(
    max(bitPitch, (fieldShape.x - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch),
    max(bitPitch, (fieldShape.y - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch)
  );
  vec2 centered = point - span * 0.5;

  vec2 coreHalfSpan = max((fieldShape.xy - 2.0 * fieldPadding.xy) * (0.5 * tilePitch), vec2(0.0));
  float attenuation = paddingAttenuation(centered, coreHalfSpan, fieldPadding.zw, tilePitch);
  float fade = exp(-attenuation);

  float z = sample.z;
  return vec4(centered.x, centered.y, z, fade);
}
`;

// ---------------------------------------------------------------------------
//  Builder
// ---------------------------------------------------------------------------

export function buildBitpackField(view: MathBoxView, config: BitpackFieldConfig): BitpackFieldHandles {
  const columns = positiveInteger(config.columns);
  const rows = positiveInteger(config.rows);
  const bitsPerSide = positiveInteger(config.bitsPerSide);
  const bitPitch = positiveFinite(config.bitPitch, DEFAULT_BIT_PITCH);
  const tilePitch = positiveFinite(config.tilePitch, DEFAULT_TILE_PITCH);
  const setBitZDisplacement = finiteNumber(config.setBitZDisplacement, 0.0001);
  const unsetBitZDisplacement = finiteNumber(config.unsetBitZDisplacement, -0.0001);
  const pointSize = positiveFinite(config.pointSize ?? NaN, DEFAULT_BIT_POINT_SIZE);
  const vectorWidth = positiveFinite(config.vectorWidth ?? NaN, DEFAULT_BIT_VECTOR_WIDTH);
  const colors = config.colors ?? DEFAULT_BIT_COLORS;
  const bitSource = config.bitSource ?? proceduralBitSource(0);
  const showDisplacementVectors = config.showDisplacementVectors ?? true;
  const paddingCols = Math.max(0, Math.floor(config.paddingCols ?? 0));
  const paddingRows = Math.max(0, Math.floor(config.paddingRows ?? 0));
  const legacyPaddingFalloff = positiveFinite(config.paddingFalloff ?? NaN, 1);
  const paddingFalloffX = positiveFinite(config.paddingFalloffX ?? NaN, legacyPaddingFalloff);
  const paddingFalloffY = positiveFinite(config.paddingFalloffY ?? NaN, legacyPaddingFalloff);

  const prefix = config.idPrefix ? `${config.idPrefix}-` : "";
  const id = (name: string) => `${prefix}${name}`;
  const bitsId = id("bitpack-bits");
  const displacedId = id("bitpack-displaced-bits");
  const surfaceId = id("bitpack-surface");
  const colorsId = id("bitpack-colors");
  const vectorsId = id("bitpack-displacement-vectors");
  const vectorColorsId = id("bitpack-displacement-vector-colors");

  // Optional hydrated per-bit buffer, bound as an extra shader source.
  const bitShaderProps: Record<string, unknown> = {
    code: bitSource.glsl,
    ...(bitSource.uniforms ?? {}),
  };
  if (bitSource.kind === "sampled") {
    const buffer = bitSource.buffer;
    view.matrix({
      id: buffer.id,
      width: buffer.width,
      height: buffer.height,
      channels: 1,
      data: buffer.data,
      live: false,
    });
    bitShaderProps.sources = [`#${buffer.id}`];
  }

  // Data pipeline: seed -> 8x8 tile -> repeat -> bit assignment -> Z displacement
  // -> world layout (surface) -> per-point colour. Draw nodes reference the
  // surface/colour buffers by id, so they need not sit inline in the chain.
  view
    .array({ width: 1, channels: 4, items: 1, data: [[0, 0, 0, 0]], live: true })
    .shader({ code: BITPACK_8X8_SHADER })
    .resample({ width: bitsPerSide, height: bitsPerSide, indices: 3, channels: 4, sample: "absolute" })
    .repeat({ width: columns, height: rows })
    .shader(bitShaderProps)
    .resample({ id: bitsId, indices: 3, channels: 4, sample: "absolute" })
    .shader({ code: BIT_Z_DISPLACEMENT_SHADER, setBitZDisplacement, unsetBitZDisplacement })
    .resample({ source: `#${bitsId}`, id: displacedId, indices: 3, channels: 4, sample: "absolute" })
    .shader({
      code: BITPACK_TILE_SHADER,
      fieldShape: [columns, rows, bitsPerSide, 0],
      fieldSpacing: [bitPitch, tilePitch],
      fieldPadding: [paddingCols, paddingRows, paddingFalloffX, paddingFalloffY],
    })
    .resample({ source: `#${displacedId}`, id: surfaceId, indices: 3, channels: 4, sample: "absolute" })
    .shader({ code: bitZColorShader(colors), setBitZDisplacement, unsetBitZDisplacement })
    .resample({ source: `#${surfaceId}`, id: colorsId, indices: 2, channels: 4 });

  if (showDisplacementVectors) {
    view
      .shader({ code: BIT_DISPLACEMENT_VECTOR_SHADER })
      .resample({ source: `#${surfaceId}`, id: vectorsId, items: 2, indices: 4, channels: 4, sample: "absolute" })
      .shader({ code: bitDisplacementVectorColorShader(colors) })
      .resample({ source: `#${surfaceId}`, id: vectorColorsId, items: 2, indices: 4, channels: 4, sample: "absolute" });
    view.vector({
      id: id("bit-displacement-vectors"),
      points: `#${vectorsId}`,
      colors: `#${vectorColorsId}`,
      color: "#111",
      width: vectorWidth,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      start: false,
      end: false,
      zBias: -1,
      zWrite: false,
    });
  }

  view.point({
    id: id("bitpack-points"),
    points: `#${surfaceId}`,
    colors: `#${colorsId}`,
    color: "#fff",
    size: pointSize,
    opacity: 1,
    blending: THREE.AdditiveBlending,
    zBias: 0,
    zWrite: false,
  });

  const frame = new FieldFrame(columns, rows, bitsPerSide, bitPitch, tilePitch);
  return { frame, extent: frame.extent(), surfaceId, colorsId };
}

// ---------------------------------------------------------------------------
//  numeric helpers
// ---------------------------------------------------------------------------

function positiveInteger(value: number): number {
  return Math.max(1, Math.floor(value));
}

function positiveFinite(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function finiteNumber(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}
