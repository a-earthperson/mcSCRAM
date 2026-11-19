/**
 * Theme-aware styling for the bitpack field and its SYCL overlays.
 *
 * Centralises every colour / width / opacity decision so the rendering
 * primitives stay declarative and the whole scene re-themes from one place when
 * the light/dark palette changes. The bit-state colours keep their original
 * high-contrast defaults (the field reads as raw GPU memory), while the overlay
 * register is drawn from the archive pigment palette and the kernel's colour
 * bucket so it harmonises with the rest of the execution-plan UI.
 */
import type { InkPalette } from "../../../../../components/plot/plotTokens";
import { withAlpha } from "../../../../../components/plot/plotTokens";
import { type ExecKindKey, kindColor } from "../scene/executionScene";

export type BitColors = {
  base: string;
  set: string;
  unset: string;
};

/** Stroke + fill register for one overlay tier. */
export type OverlayStyle = {
  color: string;
  /** Screen-space line width in CSS pixels. */
  width: number;
  /** World-unit margin added to all sides of overlay rects and grid bounds. */
  padding: number;
  fillOpacity: number;
  lineOpacity: number;
};

export type FieldStyle = {
  bits: BitColors;
  pointSize: number;
  vectorWidth: number;
  setBitZDisplacement: number;
  unsetBitZDisplacement: number;
  workItem: OverlayStyle;
  subGroup: OverlayStyle;
  workGroup: OverlayStyle;
};

/** Original raw-memory bit palette (kept stable across themes). */
const DEFAULT_BITS: BitColors = {
  base: "#000",
  set: "#ff4633",
  unset: "#fff14f",
};

export type FieldStyleOptions = {
  bits?: Partial<BitColors>;
  pointSize?: number;
  vectorWidth?: number;
  setBitZDisplacement?: number;
  unsetBitZDisplacement?: number;
};

/**
 * Build a full {@link FieldStyle} from the active palette and the kernel's
 * colour bucket. The kernel hue tints the work-group outline so different
 * kernel families are distinguishable while sharing one overlay grammar.
 */
export function fieldStyle(
  palette: InkPalette,
  kind: ExecKindKey,
  options: FieldStyleOptions = {},
): FieldStyle {
  const accent = kindColor(kind, palette);
  return {
    bits: { ...DEFAULT_BITS, ...options.bits },
    pointSize: options.pointSize ?? 10,
    vectorWidth: 5,
    setBitZDisplacement: options.setBitZDisplacement ?? 0.0001,
    unsetBitZDisplacement: options.unsetBitZDisplacement ?? -0.0001,
    workItem: {
      color: "#0f0", // green
      width: 10,
      padding: 0,
      fillOpacity: 0,
      lineOpacity: 1,
    },
    subGroup: {
      color: "#f00", // red
      width: 30,
      padding: 0,
      fillOpacity: 0,
      lineOpacity: 1.0,
    },
    workGroup: {
      color: "#00f", // blue
      width: 40,
      padding: 0.06,
      fillOpacity: 0.0,
      lineOpacity: 1.0,
    },
  };
}

/** Apply alpha to an overlay colour (delegates to the shared palette helper). */
export function overlayFill(style: OverlayStyle): string {
  return withAlpha(style.color, style.fillOpacity);
}

export function overlayLine(style: OverlayStyle): string {
  return withAlpha(style.color, style.lineOpacity);
}
