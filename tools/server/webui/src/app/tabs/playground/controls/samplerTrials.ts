import { z } from "zod";

/** One unit on the Trials (N) control equals one billion Bernoulli trials. */

export const TRIALS_PER_BILLION = 1_000_000_000;

export const DEFAULT_TRIALS_SCALAR = 1.0;
export const MIN_TRIALS_SCALAR = 0.1;
export const MAX_TRIALS_SCALAR = 10.0;
export const TRIALS_SCALAR_STEP = 0.1;

export const MIN_TRIALS_COUNT = trialsScalarToCount(MIN_TRIALS_SCALAR);
export const MAX_TRIALS_COUNT = trialsScalarToCount(MAX_TRIALS_SCALAR);
export const DEFAULT_TRIALS_COUNT = trialsScalarToCount(DEFAULT_TRIALS_SCALAR);

export const TRIALS_N_FIELD_HINT =
  "Scalar multiplier for total Bernoulli trials. Actual N = value × 10⁹ (0.1 → 100 million, 1.0 → 1 billion, 10.0 → 10 billion). Adjust in steps of 0.1.";

export function trialsScalarToCount(scalar: number): number {
  return Math.round(scalar * TRIALS_PER_BILLION);
}

export function trialsCountToScalar(count: number): number {
  return count / TRIALS_PER_BILLION;
}

/** Snap to 0.1 steps and clamp to the allowed scalar range. */
export function normalizeTrialsScalar(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_TRIALS_SCALAR;
  const stepped = Math.round(value / TRIALS_SCALAR_STEP) * TRIALS_SCALAR_STEP;
  const clamped = Math.min(MAX_TRIALS_SCALAR, Math.max(MIN_TRIALS_SCALAR, stepped));
  return Number(clamped.toFixed(1));
}

/** Step by {@link TRIALS_SCALAR_STEP} for increment/decrement controls. */
export function stepTrialsScalar(value: number, direction: 1 | -1): number {
  return normalizeTrialsScalar(value + direction * TRIALS_SCALAR_STEP);
}

export const trialsScalarSchema = z.coerce
  .number({ error: "Enter a number" })
  .refine(Number.isFinite, { message: "Enter a number" })
  .min(MIN_TRIALS_SCALAR, { message: `Minimum is ${MIN_TRIALS_SCALAR} (100 million trials)` })
  .max(MAX_TRIALS_SCALAR, { message: `Maximum is ${MAX_TRIALS_SCALAR} (10 billion trials)` })
  .transform(normalizeTrialsScalar);

export type TrialsScalar = z.infer<typeof trialsScalarSchema>;

export function formatTrialsScalar(scalar: number): string {
  return normalizeTrialsScalar(scalar).toFixed(1);
}

function formatTrialsAmount(scalar: number): string {
  const normalized = normalizeTrialsScalar(scalar);
  const count = trialsScalarToCount(normalized);
  if (count >= TRIALS_PER_BILLION) {
    return `${(count / TRIALS_PER_BILLION).toFixed(1)} billion trials`;
  }
  return `${(count / 1_000_000).toFixed(1)} million trials`;
}

export function formatTrialsDescription(scalar: number, stopOnConvergence = false): string {
  const amount = formatTrialsAmount(scalar);
  return stopOnConvergence ? `upto ${amount}` : amount;
}
