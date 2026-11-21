/**
 * Color/alpha splitting for the THREE.Color boundary.
 *
 * MathBox primitives forward `color` to `THREE.Color`, which silently drops
 * the alpha channel (with a console warning) — translucency must travel on
 * the node's `opacity` prop instead. Palette tokens resolved from
 * design-language.css (and `withAlpha(...)` results) are legacy `rgba()` /
 * `#rrggbbaa` strings, so every color handed to a primitive is split here:
 * the opaque color goes to THREE, the alpha multiplies into the opacity.
 */

export type ColorAlpha = { readonly color: string; readonly alpha: number };

const cache = new Map<string, ColorAlpha>();

const RGB_FN = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:[,/]\s*([\d.]+)(%?)\s*)?\)$/i;

/** Split a CSS color into an opaque color string and its alpha component. */
export function splitColorAlpha(input: string): ColorAlpha {
  const trimmed = input.trim();
  const cached = cache.get(trimmed);
  if (cached) return cached;

  let result: ColorAlpha = { color: trimmed, alpha: 1 };

  if (trimmed.startsWith("#")) {
    if (trimmed.length === 9) {
      // #rrggbbaa
      result = { color: trimmed.slice(0, 7), alpha: Number.parseInt(trimmed.slice(7, 9), 16) / 255 };
    } else if (trimmed.length === 5) {
      // #rgba
      const a = Number.parseInt(trimmed[4] + trimmed[4], 16) / 255;
      result = { color: trimmed.slice(0, 4), alpha: a };
    }
  } else {
    const match = trimmed.match(RGB_FN);
    if (match) {
      const alphaRaw = match[4];
      const alpha = alphaRaw === undefined ? 1 : Number.parseFloat(alphaRaw) / (match[5] === "%" ? 100 : 1);
      result = {
        color: `rgb(${match[1]}, ${match[2]}, ${match[3]})`,
        alpha: Number.isFinite(alpha) ? Math.min(1, Math.max(0, alpha)) : 1,
      };
    }
  }

  cache.set(trimmed, result);
  return result;
}
