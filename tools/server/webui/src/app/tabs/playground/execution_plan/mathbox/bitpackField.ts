import type { InkPalette } from "../../../../../components/plotTokens";
import type { BasicEventBitpackBufferShape, MathBoxView } from "./types";

export type BitpackPointFieldConfig = {
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
};

export const DEFAULT_BITPACK_FIELD_CONFIG: BitpackPointFieldConfig = {
  bitpackColumns: 1,
  bitpackRows: 1,
  bitsPerSide: 8,
  bitPitch: 1,
  tilePitch: 8 + 1,
};

const BITPACK_8X8_SHADER = `
vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  return vec4(floor(xyz.x + 0.5), floor(xyz.y + 0.5), 0.0, 0.0);
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
  vec2 extent = vec2(
    max(bitPitch, (fieldShape.x - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch),
    max(bitPitch, (fieldShape.y - 1.0) * tilePitch + (bitsPerSide - 1.0) * bitPitch)
  );
  vec2 centered = point - extent * 0.5;
  float fieldScale = 2.0 / max(0.000001, max(extent.x, extent.y));

  return vec4(centered.x * fieldScale, centered.y * fieldScale, 0.0, 0.0);
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

export function buildBitpackPointField(
  view: MathBoxView,
  config: BitpackPointFieldConfig,
  palette: InkPalette,
): void {
  const bitpackColumns = positiveInteger(config.bitpackColumns);
  const bitpackRows = positiveInteger(config.bitpackRows);
  const bitsPerSide = positiveInteger(config.bitsPerSide);
  const bitPitch = positiveFinite(config.bitPitch, DEFAULT_BITPACK_FIELD_CONFIG.bitPitch);
  const tilePitch = positiveFinite(config.tilePitch, DEFAULT_BITPACK_FIELD_CONFIG.tilePitch);

  view
    .array({
      width: 1,
      channels: 4,
      items: 1,
      data: [[0, 0, 0, 0]],
      live: false,
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
      code: BITPACK_TILE_SHADER,
      fieldShape: [bitpackColumns, bitpackRows, bitsPerSide, 0],
      fieldSpacing: [bitPitch, tilePitch],
    })
    .resample({
      indices: 3,
      channels: 4,
      sample: "absolute",
    })
    .point({
      color: palette.ink,
      size: 2.5,
      opacity: 0.75,
      blending: "add",
      zBias: 5,
      zWrite: false,
    });
}

export function bitpackPointFieldFromBasicEventBitpackBufferShape(
  bufferShape: BasicEventBitpackBufferShape,
  config: Pick<BitpackPointFieldConfig, "bitPitch" | "tilePitch"> = DEFAULT_BITPACK_FIELD_CONFIG,
): BitpackPointFieldConfig {
  const bitsPerSide = requireSquareBitpack(bufferShape.bits_per_bitpack);
  return {
    bitPitch: config.bitPitch,
    tilePitch: config.tilePitch,
    bitpackRows: bufferShape.num_events,
    bitpackColumns: bufferShape.bitpacks_per_batch,
    bitsPerSide,
  };
}