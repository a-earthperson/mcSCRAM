import { z } from "zod";
import { formatNumber } from "../../../../lib/format";

/** User-facing two-sided confidence level (%). Source of truth for the control. */
export const DEFAULT_CONFIDENCE_CI_PERCENT = 95;
export const MIN_CONFIDENCE_CI_PERCENT = 90;
export const MAX_CONFIDENCE_CI_PERCENT = 99.999;

/** Preset ladder used by increment/decrement controls (not manual entry). */
export const CONFIDENCE_CI_PRESETS = [90, 95, 99, 99.9, 99.99, 99.999] as const;

const PRESET_EPSILON = 1e-6;

/** mcSCRAM `ci_confidence` default (fraction in (0, 1)). */
export const DEFAULT_CONFIDENCE = ciPercentToConfidence(DEFAULT_CONFIDENCE_CI_PERCENT);

export const CONFIDENCE_CI_FIELD_HINT =
  "Two-sided confidence level (percent) for Wald stopping bands. Use −/+ to step through 90, 95, 99, 99.9, 99.99, and 99.999% CI, or type any value from 90 to 99.999. The engine receives confidence as a fraction in (0, 1), e.g. 95% CI → confidence = 0.95.";

/** Clamp manual input to the allowed range (no preset snapping). */
export function clampConfidenceCiPercent(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_CONFIDENCE_CI_PERCENT;
  const clamped = Math.min(MAX_CONFIDENCE_CI_PERCENT, Math.max(MIN_CONFIDENCE_CI_PERCENT, value));
  return Number(clamped.toFixed(4));
}

export const confidenceCiPercentSchema = z.coerce
  .number({ error: "Enter a number" })
  .refine(Number.isFinite, { message: "Enter a number" })
  .min(MIN_CONFIDENCE_CI_PERCENT, { message: `Minimum is ${formatCiInput(MIN_CONFIDENCE_CI_PERCENT)}% CI` })
  .max(MAX_CONFIDENCE_CI_PERCENT, { message: `Maximum is ${formatCiInput(MAX_CONFIDENCE_CI_PERCENT)}% CI` })
  .transform(clampConfidenceCiPercent);

export type ConfidenceCiPercent = z.infer<typeof confidenceCiPercentSchema>;

/** Next/previous preset for increment/decrement (always lands on the ladder). */
export function stepConfidenceCiPercent(value: number, direction: 1 | -1): number {
  const current = clampConfidenceCiPercent(value);
  if (direction > 0) {
    for (const preset of CONFIDENCE_CI_PRESETS) {
      if (preset > current + PRESET_EPSILON) return preset;
    }
    return CONFIDENCE_CI_PRESETS[CONFIDENCE_CI_PRESETS.length - 1];
  }
  for (let index = CONFIDENCE_CI_PRESETS.length - 1; index >= 0; index -= 1) {
    const preset = CONFIDENCE_CI_PRESETS[index];
    if (preset < current - PRESET_EPSILON) return preset;
  }
  return CONFIDENCE_CI_PRESETS[0];
}

/** mcSCRAM settings `ci_confidence` (fraction in (0, 1)). */
export function ciPercentToConfidence(ciPercent: number): number {
  return clampConfidenceCiPercent(ciPercent) / 100;
}

export function confidenceToCiPercent(confidence: number): number {
  if (!Number.isFinite(confidence)) return DEFAULT_CONFIDENCE_CI_PERCENT;
  return clampConfidenceCiPercent(confidence * 100);
}

export function formatCiInput(ciPercent: number): string {
  return formatNumber(clampConfidenceCiPercent(ciPercent), 4);
}

export function formatCiDescription(ciPercent: number): string {
  const ci = clampConfidenceCiPercent(ciPercent);
  return `${formatCiInput(ci)}% CI · engine confidence = ${formatNumber(ciPercentToConfidence(ci), 6)}`;
}
