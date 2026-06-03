import { z } from "zod";
import { formatNumber } from "../../../../lib/format";

/** mcSCRAM sentinel: negative means oracle diagnostics off (`settings.oracle_p`). */
export const ORACLE_P_OFF = -1;

export const MIN_ORACLE_P = 0;
export const MAX_ORACLE_P = 1;

/** Decade ladder for −/+ stepping across rare-event magnitudes. */
export const ORACLE_P_PRESETS = Array.from({ length: 13 }, (_, index) => 10 ** (index - 12)) as readonly number[];

const PRESET_EPSILON = 1e-18;

export function isOraclePEnabled(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

export function clampOracleP(value: number): number {
  if (!Number.isFinite(value) || value < 0) return ORACLE_P_OFF;
  return Math.min(MAX_ORACLE_P, Math.max(MIN_ORACLE_P, value));
}

export const oraclePValueSchema = z.coerce
  .number({ error: "Enter a number" })
  .refine(Number.isFinite, { message: "Enter a number" })
  .min(MIN_ORACLE_P, { message: "Oracle p must be at least 0" })
  .max(MAX_ORACLE_P, { message: `Maximum oracle p is ${MAX_ORACLE_P}` })
  .transform(clampOracleP);

export type OracleP = z.infer<typeof oraclePValueSchema>;

export function stepOracleP(value: number, direction: 1 | -1): number {
  if (!isOraclePEnabled(value)) {
    return direction > 0 ? ORACLE_P_PRESETS[0] : ORACLE_P_OFF;
  }
  const current = clampOracleP(value);
  if (direction > 0) {
    for (const preset of ORACLE_P_PRESETS) {
      if (preset > current + PRESET_EPSILON) return preset;
    }
    return ORACLE_P_PRESETS[ORACLE_P_PRESETS.length - 1];
  }
  for (let index = ORACLE_P_PRESETS.length - 1; index >= 0; index -= 1) {
    const preset = ORACLE_P_PRESETS[index];
    if (preset < current - PRESET_EPSILON) return preset;
  }
  return ORACLE_P_OFF;
}

/** Default display: six decimal places in scientific notation (manual entry may use fixed decimal). */
export const ORACLE_P_DISPLAY_DECIMALS = 6;

export function formatOraclePScientific(value: number, decimals = ORACLE_P_DISPLAY_DECIMALS): string {
  if (value === 0) return `0.${"0".repeat(decimals)}`;
  return value.toExponential(decimals);
}

export function formatOraclePInput(value: number): string {
  if (!isOraclePEnabled(value)) return "";
  return formatOraclePScientific(clampOracleP(value));
}

export function formatOraclePDescription(value: number): string {
  if (!isOraclePEnabled(value)) {
    return "";
  }
  const p = clampOracleP(value);
  if (p === 0) return "µ = 0.000000 · exact zero-probability oracle";
  const log10 = Math.log10(p);
  return `µ = ${formatOraclePScientific(p)} · log₁₀ p = ${formatNumber(log10, 4)}`;
}

export const ORACLE_P_FIELD_HINT = [
  "Known (true) top-event probability µ used as the Monte-Carlo oracle when diagnostics are enabled.",
  "Leave empty to disable (mcSCRAM oracle = −1). Type any p in [0, 1] (scientific or fixed decimal); displayed values use 6-decimal scientific notation. Use −/+ to step decades from 10⁻¹² through 1.",
  "When set, convergence diagnostics compare the running estimate p̂ against this reference.",
].join(" ");
