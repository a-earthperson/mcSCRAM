import type { BasicEventBitpackBufferShape, MathBoxView } from "./types";
import * as THREE from "three";

export type BitpackPointFieldConfig = {
  bitpackPadding?: `${number} ${number} ${number} ${number}`; /* top | right | bottom | left */
  /** Number of bitpack tiles along X. For basic events this is bitpacks_per_batch. */
  bitpackColumns: number;
  /** Number of bitpack tiles along Y. For basic events this is num_events. */
  bitpackRows: number;
  /** bitpack_t is rendered as bitsPerSide x bitsPerSide points. 64-bit bitpacks are 8 x 8. */
  bitsPerSide: number;
  /** Point-to-point spacing within one bitpack, in field units. */
  bitPitch: number;
  /** Distance between adjacent bitpack tile origins, in the same field units. */
  tilePitch: number;
  /** Screen-space bit point diameter, in CSS pixels. Defaults scale from bitPitch. */
  pointSize?: number;
  /** Screen-space displacement-vector stroke width, in CSS pixels. Defaults scale from bitPitch. */
  vectorWidth?: number;
  /** Z coordinate for set bits. */
  setBitZDisplacement: number;
  /** Z coordinate for unset bits. */
  unsetBitZDisplacement: number;
};

export type BitpackPointFieldWorldExtent = {
  width: number;
  height: number;
  maxSpan: number;
};

export const BITPACK_SIZE = 64;

const DEFAULT_BIT_POINT_SIZE_PER_PITCH = 20;
const DEFAULT_BIT_VECTOR_WIDTH_PER_PITCH = 3;

export const DEFAULT_BITPACK_FIELD_CONFIG: BitpackPointFieldConfig = {
  bitpackPadding: "2 3 2 3", // /* top | right | bottom | left */
  bitpackColumns: 7,
  bitpackRows: 5,
  bitsPerSide: Math.sqrt(BITPACK_SIZE),
  bitPitch: 0.02,
  tilePitch: 0.180,
  pointSize: DEFAULT_BIT_POINT_SIZE_PER_PITCH,
  vectorWidth: DEFAULT_BIT_VECTOR_WIDTH_PER_PITCH,
  setBitZDisplacement: 0.0001,
  unsetBitZDisplacement: -0.0001,
};

const BIT_COLORS = () => {
  return {
    base: "#000",
    set: "#ff4633",
    unset: "#fff14f",
  };
};

type BitColors = ReturnType<typeof BIT_COLORS>;

function colorLiteral(color: string): string {
  const c = new THREE.Color(color);
  return `vec4(${c.r.toFixed(8)}, ${c.g.toFixed(8)}, ${c.b.toFixed(8)}, 1.0)`;
}

/** Per-texel color from signed z displacement. Palette constants are compiled once per graph. */
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

  return ${colorLiteral(colors.base)} * baseWeight
    + ${colorLiteral(colors.unset)} * unsetWeight
    + ${colorLiteral(colors.set)} * setWeight;
}
`;
}

/** Two endpoints per bit: base z=0 then displaced z. */
const BIT_DISPLACEMENT_VECTOR_SHADER = `
vec4 getSample(vec4 xyzw);

vec4 getFramesSample(vec4 xyzw) {
  vec4 c = getSample(vec4(xyzw.xyz, 0.0));
  float endpoint = step(0.5, xyzw.w);
  return vec4(c.xy, c.z * endpoint, c.a);
}
`;

/** Color gradient for displacement vectors: base color at z=0, bit-state color at displaced z. */
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

const BITPACK_BITS_ID = "bitpack-bits";
const BITPACK_DISPLACED_BITS_ID = "bitpack-displaced-bits";
const BITPACK_SURFACE_ID = "bitpack-surface";
const BITPACK_COLORS_ID = "bitpack-colors";
const BITPACK_DISPLACEMENT_VECTORS_ID = "bitpack-displacement-vectors";
const BITPACK_DISPLACEMENT_VECTOR_COLORS_ID = "bitpack-displacement-vector-colors";

const BITPACK_8X8_SHADER = `
vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  return vec4(floor(xyz.x + 0.5), floor(xyz.y + 0.5), 0.0, 0.0);
}
`;

/** Cheap low-magnitude coordinate hash; emits one deterministic 0/1 bit per repeated grid point. */
const BIT_RANDOM_BITS_SHADER = `
vec4 getSample(vec3 xyz);

float randomBit(vec2 globalCell, vec2 localCell) {
  float state = globalCell.x * 17.0 + globalCell.y * 131.0;
  state += localCell.x * 29.0 + localCell.y * 43.0 + 23.0;
  state = mod(state, 251.0);
  state = mod(state * 73.0 + 41.0, 251.0);
  state = mod(state * state + state * 37.0 + 17.0, 251.0);
  return step(125.0, state);
}

vec4 getFramesSample(vec3 xyz) {
  vec4 c = getSample(xyz);
  vec2 globalCell = floor(xyz.xy + vec2(0.5));
  vec2 localCell = floor(c.xy + vec2(0.5));
  float bit = randomBit(globalCell, localCell);
  return vec4(c.xy, bit, c.a);
}
`;

/** Map bit state in z (0/1) to signed z displacement before tiling. */
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

vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  float bitsPerSide = fieldShape.z;
  float bitPitch = fieldSpacing.x;
  float tilePitch = fieldSpacing.y;

  vec2 repeatedIndex = floor(xyz.xy + vec2(0.5));
  vec2 tile = floor(repeatedIndex / bitsPerSide);
  vec2 local = getSample(xyz).xy;

  vec2 point = vec2(
    tile.x * tilePitch + local.x * bitPitch,
    tile.y * tilePitch + local.y * bitPitch
  );
  vec2 span = vec2(
    max(bitPitch, (fieldShape.x - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch),
    max(bitPitch, (fieldShape.y - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch)
  );
  vec2 centered = point - span * 0.5;

  float z = getSample(xyz).z;
  return vec4(centered.x, centered.y, z, 0.0);
}
`;

function requireSquareBitpack(bitsPerBitpack: number): number {
  const bitsPerSide = Math.sqrt(bitsPerBitpack);
  if (!Number.isInteger(bitsPerSide) || bitsPerSide <= 0) {
    throw new Error(`bitpack point field requires a square bitpack, got ${bitsPerBitpack} bits`);
  }
  return bitsPerSide;
}

function positiveInteger(value: number): number {
  return Math.max(1, Math.floor(value));
}

function positiveFinite(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function finiteNumber(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function pointSizeForBitPitch(bitPitch: number): number {
  return clamp(bitPitch * DEFAULT_BIT_POINT_SIZE_PER_PITCH, 6, 18);
}

function vectorWidthForBitPitch(bitPitch: number): number {
  return clamp(bitPitch * DEFAULT_BIT_VECTOR_WIDTH_PER_PITCH, 1.5, 8);
}

function bitpackFieldAxisSpan(
  bitpackTiles: number,
  bitsPerSide: number,
  bitPitch: number,
  tilePitch: number,
): number {
  return Math.max(
    bitPitch,
    (bitpackTiles - 1) * tilePitch + (bitsPerSide - 1) * bitPitch,
  );
}

export function bitpackPointFieldWorldExtent(
  config: BitpackPointFieldConfig,
): BitpackPointFieldWorldExtent {
  const bitpackColumns = positiveInteger(config.bitpackColumns);
  const bitpackRows = positiveInteger(config.bitpackRows);
  const bitsPerSide = positiveInteger(config.bitsPerSide);
  const bitPitch = positiveFinite(config.bitPitch, DEFAULT_BITPACK_FIELD_CONFIG.bitPitch);
  const tilePitch = positiveFinite(config.tilePitch, DEFAULT_BITPACK_FIELD_CONFIG.tilePitch);
  const width = bitpackFieldAxisSpan(bitpackColumns, bitsPerSide, bitPitch, tilePitch);
  const height = bitpackFieldAxisSpan(bitpackRows, bitsPerSide, bitPitch, tilePitch);

  return {
    width,
    height,
    maxSpan: Math.max(width, height),
  };
}

export function buildBitpackPointField(
  view: MathBoxView,
  config: BitpackPointFieldConfig,
): void {
  const bitpackColumns = positiveInteger(config.bitpackColumns);
  const bitpackRows = positiveInteger(config.bitpackRows);
  const bitsPerSide = positiveInteger(config.bitsPerSide);
  const bitPitch = positiveFinite(config.bitPitch, DEFAULT_BITPACK_FIELD_CONFIG.bitPitch);
  const tilePitch = positiveFinite(config.tilePitch, DEFAULT_BITPACK_FIELD_CONFIG.tilePitch);
  const setBitZDisplacement = finiteNumber(
    config.setBitZDisplacement,
    DEFAULT_BITPACK_FIELD_CONFIG.setBitZDisplacement,
  );
  const unsetBitZDisplacement = finiteNumber(
    config.unsetBitZDisplacement,
    DEFAULT_BITPACK_FIELD_CONFIG.unsetBitZDisplacement,
  );
  const pointSize = positiveFinite(config.pointSize ?? NaN, pointSizeForBitPitch(bitPitch));
  const vectorWidth = positiveFinite(config.vectorWidth ?? NaN, vectorWidthForBitPitch(bitPitch));
  const bitColors = BIT_COLORS();

  view
    .array({
      width: 1,
      channels: 4,
      items: 1,
      data: [[0, 0, 0, 0]],
      live: true,
    })
    .shader({
      code: BITPACK_8X8_SHADER,
    })
    .resample({
      width: bitsPerSide,
      height: bitsPerSide,
      indices: 3,
      channels: 4,
      sample: "absolute",
    })
    .repeat({
      width: bitpackColumns,
      height: bitpackRows,
    })
    .shader({
      code: BIT_RANDOM_BITS_SHADER,
    })
    .resample({
      id: BITPACK_BITS_ID,
      indices: 3,
      channels: 4,
      sample: "absolute",
    })
    .shader({
      code: BIT_Z_DISPLACEMENT_SHADER,
      setBitZDisplacement,
      unsetBitZDisplacement,
    })
    .resample({
      source: `#${BITPACK_BITS_ID}`,
      id: BITPACK_DISPLACED_BITS_ID,
      indices: 3,
      channels: 4,
      sample: "absolute",
    })
    .shader({
      code: BITPACK_TILE_SHADER,
      fieldShape: [bitpackColumns, bitpackRows, bitsPerSide, 0],
      fieldSpacing: [bitPitch, tilePitch],
    })
    .resample({
      source: `#${BITPACK_DISPLACED_BITS_ID}`,
      id: BITPACK_SURFACE_ID,
      indices: 3,
      channels: 4,
      sample: "absolute",
    })
    .shader({
      code: bitZColorShader(bitColors),
      setBitZDisplacement,
      unsetBitZDisplacement,
    })
    .resample({
      source: `#${BITPACK_SURFACE_ID}`,
      id: BITPACK_COLORS_ID,
      indices: 2,
      channels: 4,
    })
    .shader({
      code: BIT_DISPLACEMENT_VECTOR_SHADER,
    })
    .resample({
      source: `#${BITPACK_SURFACE_ID}`,
      id: BITPACK_DISPLACEMENT_VECTORS_ID,
      items: 2,
      indices: 4,
      channels: 4,
      sample: "absolute",
    })
    .shader({
      code: bitDisplacementVectorColorShader(bitColors),
    })
    .resample({
      source: `#${BITPACK_SURFACE_ID}`,
      id: BITPACK_DISPLACEMENT_VECTOR_COLORS_ID,
      items: 2,
      indices: 4,
      channels: 4,
      sample: "absolute",
    })
    .vector({
      id: "bit-displacement-vectors",
      points: `#${BITPACK_DISPLACEMENT_VECTORS_ID}`,
      colors: `#${BITPACK_DISPLACEMENT_VECTOR_COLORS_ID}`,
      color: "#111",
      width: vectorWidth,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      start: false,
      end: false,
      zBias: -1,
      zWrite: false,
    })
    .point({
      id: "points",
      points: `#${BITPACK_SURFACE_ID}`,
      colors: `#${BITPACK_COLORS_ID}`,
      color: "#fff",
      size: pointSize,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      zBias: 0,
      zWrite: false,
    });
}

export function bitpackPointFieldFromBasicEventBitpackBufferShape(
  bufferShape: BasicEventBitpackBufferShape,
  config: Partial<Pick<BitpackPointFieldConfig, "bitPitch" | "tilePitch" | "pointSize" | "vectorWidth">> = {},
): BitpackPointFieldConfig {
  const bitsPerSide = requireSquareBitpack(
    bufferShape.bits_per_bitpack ?? BITPACK_SIZE,
  );
  const bitPitch = positiveFinite(config.bitPitch ?? NaN, DEFAULT_BITPACK_FIELD_CONFIG.bitPitch);
  const tilePitch = positiveFinite(config.tilePitch ?? NaN, DEFAULT_BITPACK_FIELD_CONFIG.tilePitch);

  return {
    ...DEFAULT_BITPACK_FIELD_CONFIG,
    bitPitch,
    tilePitch,
    pointSize: config.pointSize,
    vectorWidth: config.vectorWidth,
    bitpackRows: bufferShape.num_events,
    bitpackColumns: bufferShape.bitpacks_per_batch,
    bitsPerSide,
  };
}