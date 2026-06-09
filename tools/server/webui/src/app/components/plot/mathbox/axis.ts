/**
 * Number formatters used as MathBox `format` exprs to label axis ticks. The
 * `scale` primitive generates "nice" tick *positions*; these turn each value
 * into display text. GL labels render arbitrary Unicode (the atlas is built
 * from the page font), so we use true superscript glyphs for exponents.
 */

const SUPERSCRIPTS: Record<string, string> = {
  "-": "\u207B",
  "0": "\u2070",
  "1": "\u00B9",
  "2": "\u00B2",
  "3": "\u00B3",
  "4": "\u2074",
  "5": "\u2075",
  "6": "\u2076",
  "7": "\u2077",
  "8": "\u2078",
  "9": "\u2079",
};

/** Render an integer as Unicode superscript (e.g. -4 -> "\u207B\u2074"). */
export function toSuperscript(value: number | string): string {
  return String(value).replace(/[-0-9]/g, (char) => SUPERSCRIPTS[char] ?? char);
}

/** SI-suffixed integer-ish format (1.2k, 3.4M, 1.0G) for large counts. */
export function formatSI(value: number): string {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs < 1000) return Number.isInteger(value) ? String(value) : trimZeros(value.toPrecision(3));
  const units = ["k", "M", "G", "T", "P"];
  const tier = Math.min(units.length, Math.floor(Math.log10(abs) / 3));
  const scaled = value / 10 ** (tier * 3);
  return `${trimZeros(scaled.toPrecision(3))}${units[tier - 1]}`;
}

/**
 * Power-of-ten label for log-axis ticks: clean powers render as `10\u207F`
 * (with `1` for 10\u2070), and any off-power tick degrades gracefully to a
 * compact `m\u00D710\u207F`. Pairs with `nice:false` log scales, whose ticks
 * land exactly on integer powers.
 */
export function formatPower(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "";
  const exp = Math.round(Math.log10(value));
  if (Math.abs(value - 10 ** exp) <= 1e-9 * 10 ** exp) {
    if (exp === 0) return "1";
    if (exp === 1) return "10";
    return `10${toSuperscript(exp)}`;
  }
  const floorExp = Math.floor(Math.log10(value));
  const mantissa = value / 10 ** floorExp;
  return `${trimZeros(mantissa.toPrecision(2))}\u00D710${toSuperscript(floorExp)}`;
}

/** Default linear-axis tick label: `a.bbbb\u00D710\u1d9c` (4 mantissa decimals). */
export function formatDefault(decimals = 4): (value: number) => string {
  return (value: number) => {
    if (!Number.isFinite(value)) return "";
    if (value === 0) return "0";
    const exp = Math.floor(Math.log10(Math.abs(value)));
    const mantissa = value / 10 ** exp;
    const head = trimZeros(mantissa.toFixed(decimals));
    if (exp === 0) return head;
    return `${head}\u00D710${toSuperscript(exp)}`;
  };
}

/** Shared instance for stable reference equality across renders. */
export const formatDefaultLinear = formatDefault(4);

/** Compact scientific format with Unicode superscript (`1.2\u00D710\u207B\u2078`). */
export function formatSci(sig = 2): (value: number) => string {
  return (value: number) => {
    if (!Number.isFinite(value)) return "";
    if (value === 0) return "0";
    const exp = Math.floor(Math.log10(Math.abs(value)));
    if (exp >= -3 && exp <= 3) return trimZeros(value.toPrecision(sig + 1));
    const mantissa = value / 10 ** exp;
    const head = trimZeros(mantissa.toPrecision(sig));
    return `${head === "1" ? "" : `${head}\u00D7`}10${toSuperscript(exp)}`;
  };
}

/** Mantissa-e-exponent format with `digits` mantissa decimals (1.2e-8). */
export function formatExp(digits = 1): (value: number) => string {
  return (value: number) => {
    if (!Number.isFinite(value)) return "";
    if (value === 0) return "0";
    return value.toExponential(digits);
  };
}

/** Fixed-decimal format with trailing-zero trimming. */
export function formatFixed(digits = 3): (value: number) => string {
  return (value: number) => {
    if (!Number.isFinite(value)) return "";
    return trimZeros(value.toFixed(digits));
  };
}

function trimZeros(text: string): string {
  if (!text.includes(".")) return text;
  return text.replace(/\.?0+$/, "");
}
