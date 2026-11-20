import { z } from "zod";
import { formatNumber } from "../../../lib/format";

/**
 * Relative margin δ for Wald stopping: target half-width ε = δ·p̂
 * (see convergence_controller.h).
 */
export const DEFAULT_DELTA = 0.001;
export const MIN_DELTA = 0.0001;
export const MAX_DELTA = 0.5;

/** Preset ladder for increment/decrement (ascending — required by {@link stepDelta}). */
export const DELTA_PRESETS = [0.0001, 0.0005,0.001, 0.005, 0.01, 0.05, 0.1] as const;

const PRESET_EPSILON = 1e-12;

const DECIMAL_PLACES = 4;

export function clampDelta(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return DEFAULT_DELTA;
  const clamped = Math.min(MAX_DELTA, Math.max(MIN_DELTA, value));
  return Number(clamped.toFixed(DECIMAL_PLACES));
}

export const deltaSchema = z.coerce
  .number({ error: "Enter a number" })
  .refine(Number.isFinite, { message: "Enter a number" })
  .refine((value) => value > 0, { message: "δ must be greater than 0" })
  .min(MIN_DELTA, { message: `Minimum δ is ${formatDeltaInput(MIN_DELTA)}` })
  .max(MAX_DELTA, { message: `Maximum δ is ${formatDeltaInput(MAX_DELTA)}` })
  .transform(clampDelta);

export type Delta = z.infer<typeof deltaSchema>;

export function stepDelta(value: number, direction: 1 | -1): number {
  const current = clampDelta(value);
  if (direction > 0) {
    for (const preset of DELTA_PRESETS) {
      if (preset > current + PRESET_EPSILON) return preset;
    }
    return DELTA_PRESETS[DELTA_PRESETS.length - 1];
  }
  for (let index = DELTA_PRESETS.length - 1; index >= 0; index -= 1) {
    const preset = DELTA_PRESETS[index];
    if (preset < current - PRESET_EPSILON) return preset;
  }
  return DELTA_PRESETS[0];
}

/** Fixed decimal string for the control (never scientific notation). */
export function formatDeltaInput(delta: number): string {
  const fixed = clampDelta(delta).toFixed(DECIMAL_PLACES);
  return fixed.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

export function formatDeltaDescription(delta: number): string {
  const d = clampDelta(delta);
  const percent = d * 100;
  return `δ = ${formatDeltaInput(d)} · ε = δ·p̂ · ${formatNumber(percent, 4)}% at p̂ = 1`;
}

export const DELTA_FIELD_HINT = [
  "Required precision, relative to the estimate p̂.",
  "Sampling has converged once the confidence interval is no wider than ±δ·p̂ (ε = δ·p̂).",
  `The default, ${formatDeltaInput(DEFAULT_DELTA)}, means within ±${formatNumber(DEFAULT_DELTA * 100, DECIMAL_PLACES)}% of p̂.`,
  `−/+ steps through ${DELTA_PRESETS.map((preset) => formatDeltaInput(preset)).join(", ")}; you can also type any value from ${formatDeltaInput(MIN_DELTA)} to ${formatDeltaInput(MAX_DELTA)}.`,
].join(" ");
