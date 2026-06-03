import { z } from "zod";
import { formatNumber } from "../../../../lib/format";

/** Per-node allocator slack for GPU sample shaping (`sample_shaper`, default 0.05). */
export const DEFAULT_OVERHEAD_RATIO = 1;
export const MIN_OVERHEAD_RATIO = 0.1;
export const MAX_OVERHEAD_RATIO = 100;

/** Preset ladder for increment/decrement (ascending — required by {@link stepOverheadRatio}). */
export const OVERHEAD_RATIO_PRESETS = [MIN_OVERHEAD_RATIO, 0.2, 0.5, 1, 10, MAX_OVERHEAD_RATIO] as const;

const PRESET_EPSILON = 1e-12;
const DECIMAL_PLACES = 4;

export function clampOverheadRatio(value: number): number {
  if (!Number.isFinite(value) || value < 0) return DEFAULT_OVERHEAD_RATIO;
  const clamped = Math.min(MAX_OVERHEAD_RATIO, Math.max(MIN_OVERHEAD_RATIO, value));
  return Number(clamped.toFixed(DECIMAL_PLACES));
}

export const overheadRatioSchema = z.coerce
  .number({ error: "Enter a number" })
  .refine(Number.isFinite, { message: "Enter a number" })
  .min(MIN_OVERHEAD_RATIO, { message: `Minimum r is ${formatOverheadRatioInput(MIN_OVERHEAD_RATIO)}` })
  .max(MAX_OVERHEAD_RATIO, { message: `Maximum r is ${formatOverheadRatioInput(MAX_OVERHEAD_RATIO)}` })
  .transform(clampOverheadRatio);

export type OverheadRatio = z.infer<typeof overheadRatioSchema>;

export function stepOverheadRatio(value: number, direction: 1 | -1): number {
  const current = clampOverheadRatio(value);
  if (direction > 0) {
    for (const preset of OVERHEAD_RATIO_PRESETS) {
      if (preset > current + PRESET_EPSILON) return preset;
    }
    return OVERHEAD_RATIO_PRESETS[OVERHEAD_RATIO_PRESETS.length - 1];
  }
  for (let index = OVERHEAD_RATIO_PRESETS.length - 1; index >= 0; index -= 1) {
    const preset = OVERHEAD_RATIO_PRESETS[index];
    if (preset < current - PRESET_EPSILON) return preset;
  }
  return OVERHEAD_RATIO_PRESETS[0];
}

export function formatOverheadRatioInput(ratio: number): string {
  const fixed = clampOverheadRatio(ratio).toFixed(DECIMAL_PLACES);
  return fixed.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

export function formatOverheadRatioDescription(ratio: number): string {
  const r = clampOverheadRatio(ratio);
  const percent = r * 100;
  return `r = ${formatOverheadRatioInput(r)} · effective nodes ≈ N·(1+r)`;
}

export const OVERHEAD_RATIO_FIELD_HINT = [
  "Allocator overhead ratio r passed to sample_shaper when sizing GPU batches. The engine treats each node as needing N·(1+r) worth of device memory headroom so rounded buffers stay under the limit.",
  `Use −/+ to step through ${OVERHEAD_RATIO_PRESETS.map((preset) => formatOverheadRatioInput(preset)).join(", ")}, or type any r from ${formatOverheadRatioInput(MIN_OVERHEAD_RATIO)} to ${formatOverheadRatioInput(MAX_OVERHEAD_RATIO)}.`,
  `Default r = ${formatOverheadRatioInput(DEFAULT_OVERHEAD_RATIO)} (${formatNumber(DEFAULT_OVERHEAD_RATIO * 100, DECIMAL_PLACES)}% per-node slack, matching mcSCRAM CLI --overhead-ratio).`,
].join(" ");
