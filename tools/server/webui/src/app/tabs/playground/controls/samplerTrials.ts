import { z } from "zod";

export const TRIALS_PER_MILLION = 1_000_000;
export const TRIALS_PER_BILLION = 1_000_000_000;
export const TRIALS_PER_TRILLION = 1_000_000_000_000;

export const TRIALS_SCALE_EXPONENTS = [6, 9, 12] as const;
export type TrialsScaleExponent = (typeof TRIALS_SCALE_EXPONENTS)[number];

export const DEFAULT_TRIALS_SCALE_EXPONENT: TrialsScaleExponent = 9;
export const DEFAULT_TRIALS_MANTISSA = 0; // 0 = auto (unbounded / backend default)
export const TRIALS_MANTISSA_STEP = 1;
export const MIN_TRIALS_MANTISSA = 0;
export const MIN_TRIALS_COUNT = TRIALS_PER_MILLION;
export const MAX_TRIALS_COUNT = 10 * TRIALS_PER_TRILLION;
export const DEFAULT_TRIALS_COUNT = deriveTrialsCount(DEFAULT_TRIALS_MANTISSA, DEFAULT_TRIALS_SCALE_EXPONENT);

export const TRIALS_SCALE_LABELS: Record<TrialsScaleExponent, string> = {
  6: "M",
  9: "B",
  12: "T",
};

export const TRIALS_N_FIELD_HINT =
  "Enter a mantissa and choose the unit (M = ×10⁶, B = ×10⁹, T = ×10¹²). Actual N is derived as mantissa × unit (1 million to 10 trillion). Examples at B: 1 → 1 billion, 10 → 10 billion, 1000 → 1 trillion. −/+ adjusts the mantissa in steps of 1. Leave empty or type auto for unbounded sampling (num_trials = 0).";

/** @deprecated Use {@link DEFAULT_TRIALS_MANTISSA} at {@link DEFAULT_TRIALS_SCALE_EXPONENT}. */
export const DEFAULT_TRIALS_SCALAR = DEFAULT_TRIALS_MANTISSA;

export function isTrialsScaleExponent(value: number): value is TrialsScaleExponent {
  return (TRIALS_SCALE_EXPONENTS as readonly number[]).includes(value);
}

export function trialsScaleFactor(exponent: TrialsScaleExponent): number {
  return 10 ** exponent;
}

/** Derived trial count from mantissa and exponent (not a stored input). */
export function deriveTrialsCount(mantissa: number, scaleExponent: TrialsScaleExponent): number {
  if (!Number.isFinite(mantissa) || mantissa <= 0) return 0;
  return Math.round(mantissa * trialsScaleFactor(scaleExponent));
}

export function clampTrialsCount(count: number): number {
  if (!Number.isFinite(count)) return DEFAULT_TRIALS_COUNT;
  if (count === 0) return 0;
  return Math.min(MAX_TRIALS_COUNT, Math.max(MIN_TRIALS_COUNT, Math.round(count)));
}

/** Effective N sent to the sampler after clamping the derived count (0 = auto). */
export function effectiveTrialsCount(mantissa: number, scaleExponent: TrialsScaleExponent): number {
  const derived = deriveTrialsCount(mantissa, scaleExponent);
  if (derived === 0) return 0;
  return clampTrialsCount(derived);
}

export function normalizeTrialsMantissa(mantissa: number): number {
  if (!Number.isFinite(mantissa)) return DEFAULT_TRIALS_MANTISSA;
  if (mantissa === 0) return 0;
  const stepped = Math.round(mantissa / TRIALS_MANTISSA_STEP) * TRIALS_MANTISSA_STEP;
  return Math.max(TRIALS_MANTISSA_STEP, stepped);
}

export function stepTrialsMantissa(mantissa: number, direction: 1 | -1): number {
  if (mantissa === 0) {
    return direction > 0 ? TRIALS_MANTISSA_STEP : 0;
  }
  const next = mantissa + direction * TRIALS_MANTISSA_STEP;
  if (next < TRIALS_MANTISSA_STEP) return 0;
  return normalizeTrialsMantissa(next);
}

export function createTrialsMantissaSchema(scaleExponent: TrialsScaleExponent) {
  return z.coerce
    .number({ error: "Enter a number" })
    .refine(Number.isFinite, { message: "Enter a number" })
    .min(MIN_TRIALS_MANTISSA, { message: `Minimum is ${MIN_TRIALS_MANTISSA}` })
    .transform(normalizeTrialsMantissa)
    .refine(
      (value) => {
        if (value === 0) return true;
        const count = deriveTrialsCount(value, scaleExponent);
        return count >= MIN_TRIALS_COUNT && count <= MAX_TRIALS_COUNT;
      },
      {
        message: `Derived N must be between ${formatTrialsCount(MIN_TRIALS_COUNT)} and ${formatTrialsCount(MAX_TRIALS_COUNT)} trials`,
      },
    );
}

export type TrialsMantissa = z.infer<ReturnType<typeof createTrialsMantissaSchema>>;

export function formatTrialsMantissa(mantissa: number): string {
  const normalized = normalizeTrialsMantissa(mantissa);
  if (normalized === 0) return "";
  return normalized.toFixed(mantissaDisplayDecimals(normalized));
}

export function formatTrialsCount(count: number): string {
  const abs = Math.abs(count);
  if (abs >= TRIALS_PER_TRILLION) {
    return `${(count / TRIALS_PER_TRILLION).toFixed(1)} trillion`;
  }
  if (abs >= TRIALS_PER_BILLION) {
    return `${(count / TRIALS_PER_BILLION).toFixed(1)} billion`;
  }
  return `${(count / TRIALS_PER_MILLION).toFixed(1)} million`;
}

export function formatTrialsDescription(
  mantissa: number,
  scaleExponent: TrialsScaleExponent,
  stopOnConvergence = false,
): string {
  const derived = deriveTrialsCount(mantissa, scaleExponent);
  if (derived === 0) return "auto";
  const prefix = stopOnConvergence ? "upto " : "";
  if (derived > MAX_TRIALS_COUNT) {
    return `${prefix}${formatTrialsCount(MAX_TRIALS_COUNT)}`;
  }
  if (derived < MIN_TRIALS_COUNT) {
    return `${prefix}${formatTrialsCount(MIN_TRIALS_COUNT)}`;
  }
  return `${prefix}${formatTrialsCount(derived)}`;
}

/** @deprecated Use {@link deriveTrialsCount} with an explicit scale exponent. */
export function trialsMantissaToCount(mantissa: number, scaleExponent: TrialsScaleExponent): number {
  return deriveTrialsCount(mantissa, scaleExponent);
}

/** @deprecated Use {@link deriveTrialsCount} with an explicit scale exponent. */
export function trialsScalarToCount(scalar: number): number {
  return deriveTrialsCount(scalar, DEFAULT_TRIALS_SCALE_EXPONENT);
}

function mantissaDisplayDecimals(mantissa: number): number {
  return Number.isInteger(mantissa) ? 0 : 1;
}
