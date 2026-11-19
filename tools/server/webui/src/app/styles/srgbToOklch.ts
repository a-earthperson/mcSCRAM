import { converter, parse } from "culori";

const toOklch = converter("oklch");

/** Convert an sRGB color string (hex, rgb/rgba, etc.) to a CSS `oklch()` value. */
export function srgbToOklch(input: string, alpha?: number): string {
  const parsed = parse(input);
  if (!parsed) {
    throw new Error(`Unrecognized color: ${input}`);
  }

  const oklch = toOklch(parsed);
  if (!oklch) {
    throw new Error(`Failed to convert color to OKLCH: ${input}`);
  }

  const a = alpha ?? oklch.alpha ?? 1;
  const l = round(oklch.l * 100, 2);
  const c = round(oklch.c, 4);
  const h = round(oklch.h ?? 0, 2);

  if (a < 1) {
    return `oklch(${l}% ${c} ${h} / ${round(a, 3)})`;
  }
  return `oklch(${l}% ${c} ${h})`;
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
