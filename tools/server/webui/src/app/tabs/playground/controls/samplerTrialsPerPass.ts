import { z } from "zod";
import { formatInteger } from "../../../lib/format";

/**
 * Optimistic per-node trials-per-pass target for the native sample shaper
 * (`settings.requested_trials_per_pass`).
 *
 * Semantics: the *number of Bernoulli trials* each node samples per device pass.
 * It is a target — the realized value may be smaller after the shaper applies
 * device-memory and cache-line-alignment constraints. `0` selects the automatic
 * policy (size each pass to the per-node device-memory budget).
 */
export const DEFAULT_REQUESTED_TRIALS_PER_PASS = 0; // 0 = auto (fill memory budget)
export const MIN_REQUESTED_TRIALS_PER_PASS = 0;

/** Power-of-two ladder for −/+ stepping: trials = 2 ** exponent. */
export const REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN = 6; // 2 ** 6 = 64
export const REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX = 32;

export const MAX_REQUESTED_TRIALS_PER_PASS = 2 ** REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX;

export function trialsPerPassFromExponent(exponent: number): number {
  const exp = Math.min(
    REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX,
    Math.max(REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN, Math.round(exponent)),
  );
  return 2 ** exp;
}

function snapTrialsPerPassExponentDown(trials: number): number {
  for (let exp = REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX; exp >= REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN; exp -= 1) {
    if (2 ** exp <= trials) return exp;
  }
  return REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN;
}

function snapTrialsPerPassExponentUp(trials: number): number {
  for (let exp = REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN; exp <= REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX; exp += 1) {
    if (2 ** exp >= trials) return exp;
  }
  return REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX;
}

export function clampRequestedTrialsPerPass(value: number): number {
  if (!Number.isFinite(value) || value < 0) return DEFAULT_REQUESTED_TRIALS_PER_PASS;
  const clamped = Math.min(MAX_REQUESTED_TRIALS_PER_PASS, Math.max(MIN_REQUESTED_TRIALS_PER_PASS, value));
  return Math.round(clamped);
}

export const requestedTrialsPerPassSchema = z.coerce
  .number({ error: "Enter a whole number of trials" })
  .refine(Number.isFinite, { message: "Enter a whole number of trials" })
  .min(MIN_REQUESTED_TRIALS_PER_PASS, { message: `Minimum is ${MIN_REQUESTED_TRIALS_PER_PASS}` })
  .max(MAX_REQUESTED_TRIALS_PER_PASS, {
    message: `Maximum is ${formatInteger(MAX_REQUESTED_TRIALS_PER_PASS)}`,
  })
  .transform(clampRequestedTrialsPerPass);

export type RequestedTrialsPerPass = z.infer<typeof requestedTrialsPerPassSchema>;

export function stepRequestedTrialsPerPass(value: number, direction: 1 | -1): number {
  const current = clampRequestedTrialsPerPass(value);

  if (direction > 0) {
    if (current === 0) return trialsPerPassFromExponent(REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN);
    const exponent = snapTrialsPerPassExponentUp(current);
    const rung = 2 ** exponent;
    if (rung === current) {
      const nextExponent = exponent + 1;
      if (nextExponent > REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX) {
        return trialsPerPassFromExponent(REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX);
      }
      return trialsPerPassFromExponent(nextExponent);
    }
    return rung;
  }

  if (current === 0) return 0;
  const exponent = snapTrialsPerPassExponentDown(current);
  const rung = 2 ** exponent;
  if (rung === current) {
    const nextExponent = exponent - 1;
    if (nextExponent < REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN) return 0;
    return trialsPerPassFromExponent(nextExponent);
  }
  return rung;
}

/** Compact approximate magnitude for display (e.g. 16,777,216 → "~16.8 million"). */
export function formatApproxTrialsPerPass(trials: number): string {
  const count = clampRequestedTrialsPerPass(trials);
  if (count === 0) return "auto";
  const abs = Math.abs(count);
  if (abs >= 1_000_000_000) {
    return `~${(count / 1_000_000_000).toFixed(1)} billion`;
  }
  if (abs >= 1_000_000) {
    return `~${(count / 1_000_000).toFixed(1)} million`;
  }
  if (abs >= 1_000) {
    return `~${(count / 1_000).toFixed(1)} thousand`;
  }
  return `~${formatInteger(count)}`;
}

export function formatRequestedTrialsPerPassInput(value: number): string {
  const trials = clampRequestedTrialsPerPass(value);
  if (trials === 0) return "";
  return String(trials);
}

export function formatRequestedTrialsPerPassDescription(value: number): string {
  return formatApproxTrialsPerPass(value);
}

export const REQUESTED_TRIALS_PER_PASS_FIELD_HINT = [
  "Optimistic number of Bernoulli trials each node samples per device pass, passed to the sample shaper as requested_trials_per_pass.",
  "The shaper treats it as a target and clamps it to the per-node device-memory budget, then rounds down to a whole cache line, so the realized trials/pass may be smaller.",
  `Use −/+ to step powers of two from ${formatApproxTrialsPerPass(trialsPerPassFromExponent(REQUESTED_TRIALS_PER_PASS_EXPONENT_MIN))} through ${formatApproxTrialsPerPass(trialsPerPassFromExponent(REQUESTED_TRIALS_PER_PASS_EXPONENT_MAX))}, or type any whole number from ${MIN_REQUESTED_TRIALS_PER_PASS} to ${formatInteger(MAX_REQUESTED_TRIALS_PER_PASS)} (auto = fill memory budget).`,
].join(" ");
