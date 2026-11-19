import { useMemo, useSyncExternalStore } from "react";
import { converter, parse } from "culori";

/** Resolved ink + pigment register for Plotly traces (from :root CSS custom properties). */
export type InkPalette = {
  ink: string;
  inkSoft: string;
  inkMute: string;
  inkFaint: string;
  indigo: string;
  indigoDeep: string;
  sage: string;
  ochre: string;
  madder: string;
  paynes: string;
  rose: string;
  /** Full-strength bright pigments — crisp lines & markers (not washes/type). */
  indigoBright: string;
  indigoDeepBright: string;
  sageBright: string;
  ochreBright: string;
  madderBright: string;
  paynesBright: string;
  roseBright: string;
  /** Shadow-side dark pigments — deep strokes, fills, under traces. */
  indigoDark: string;
  indigoDeepDark: string;
  sageDark: string;
  ochreDark: string;
  madderDark: string;
  paynesDark: string;
  roseDark: string;
  /** Halo on filled markers (rim against dense traces). */
  paper: string;
  /** Hover label / plate interior. */
  plate: string;
  rule: string;
  grid: string;
  annotationBg: string;
  oracleBand99: string;
  oracleBand95: string;
  bandSage: string;
  bandMadder: string;
};

export type MethodStyle = {
  sampled: { color: string; symbol: string };
  oracle: { color: string; dash: string };
  oracleBand99: { color: string; dash: string };
  oracleBand95: { color: string; dash: string };
  rare: { color: string; dash: string };
  mcub: { color: string; dash: string };
  fit: { color: string; dash: string };
  parity: { color: string; dash: string };
  selected: { color: string; symbol: string };
};

export type PlotTheme = {
  scheme: "light" | "dark";
  palette: InkPalette;
  colorway: string[];
  colorscale: Array<[number, string]>;
  methodStyle: MethodStyle;
  sans: string;
  mono: string;
};

/** SSR / first-paint fallbacks when document is unavailable (light plate). */
const LIGHT_FALLBACK: InkPalette = {
  ink: "#22201e",
  inkSoft: "#35322f",
  inkMute: "#56504b",
  inkFaint: "#93897f",
  indigo: "#637d9a",
  indigoDeep: "#35577e",
  sage: "#5c7f5d",
  ochre: "#ae8c54",
  madder: "#9a6460",
  paynes: "#545966",
  rose: "#d6b8bc",
  indigoBright: "#316ba9",
  indigoDeepBright: "#20548a",
  sageBright: "#4a8c4e",
  ochreBright: "#b4821b",
  madderBright: "#ad4b47",
  paynesBright: "#556895",
  roseBright: "#c46a79",
  indigoDark: "#254466",
  indigoDeepDark: "#193451",
  sageDark: "#2f5c31",
  ochreDark: "#72551f",
  madderDark: "#6f2d2a",
  paynesDark: "#2d3750",
  roseDark: "#80424d",
  paper: "#fdfaf7",
  plate: "#faf7f3",
  rule: "rgba(34, 32, 30, 0.92)",
  grid: "rgba(34, 32, 30, 0.08)",
  annotationBg: "rgba(250, 247, 243, 0.72)",
  oracleBand99: "rgba(34, 32, 30, 0.48)",
  oracleBand95: "rgba(34, 32, 30, 0.58)",
  bandSage: "rgba(92, 127, 93, 0.12)",
  bandMadder: "rgba(154, 100, 96, 0.10)",
};

export const PLOT_TRANSPARENT = "transparent";

function readCssVar(name: `--${string}`, fallback = ""): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/* ---- CSS color → legacy sRGB resolution ----------------------------
   design-language.css is authored in oklch() and color-mix(), but the
   downstream consumers (THREE.Color via mathbox, Plotly's tinycolor,
   and the withAlpha/darken helpers below) only understand legacy
   `#rrggbb` / `rgb()` / `rgba()` strings. Custom-property reads return
   the raw token text — getComputedStyle does NOT resolve them to a
   color — so every color token is normalized here, keeping the CSS
   file as the single source of truth. */

const toRgb = converter("rgb");

let colorProbe: HTMLSpanElement | null = null;

/** Let the CSS engine evaluate expressions culori can't (e.g. color-mix()). */
function computeColorViaProbe(value: string): string {
  if (typeof document === "undefined" || !document.documentElement) return "";
  if (!colorProbe) {
    colorProbe = document.createElement("span");
    colorProbe.style.display = "none";
    document.documentElement.appendChild(colorProbe);
  }
  colorProbe.style.color = "";
  colorProbe.style.color = value;
  if (!colorProbe.style.color) return "";
  return getComputedStyle(colorProbe).color;
}

/** Token strings are stable per color scheme, so resolution is memoizable. */
const resolvedColorCache = new Map<string, string>();

/** Resolve any CSS color expression to `#rrggbb` or `rgba(r, g, b, a)`. */
export function resolveCssColor(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed.startsWith("#") || /^rgba?\(/i.test(trimmed)) return trimmed;
  const cached = resolvedColorCache.get(trimmed);
  if (cached !== undefined) return cached;

  let parsed = parse(trimmed);
  if (!parsed) {
    const computed = computeColorViaProbe(trimmed);
    parsed = computed ? parse(computed) : undefined;
  }
  const rgb = parsed ? toRgb(parsed) : undefined;
  if (!rgb) return trimmed;

  const channel = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  const r = channel(rgb.r);
  const g = channel(rgb.g);
  const b = channel(rgb.b);
  const alpha = rgb.alpha ?? 1;
  const resolved =
    alpha < 1
      ? `rgba(${r}, ${g}, ${b}, ${Math.round(alpha * 1000) / 1000})`
      : `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  resolvedColorCache.set(trimmed, resolved);
  return resolved;
}

function readCssColor(name: `--${string}`, fallback = ""): string {
  return resolveCssColor(readCssVar(name, fallback));
}

function subscribeToColorScheme(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", onStoreChange);
  return () => mq.removeEventListener("change", onStoreChange);
}

function getColorSchemeSnapshot(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Read the current plate palette from design-language.css custom properties,
    resolved to legacy sRGB strings (THREE / Plotly / tinycolor safe). */
export function getInkPalette(): InkPalette {
  const fb = LIGHT_FALLBACK;
  return {
    ink: readCssColor("--ink", fb.ink),
    inkSoft: readCssColor("--ink-2", fb.inkSoft),
    inkMute: readCssColor("--ink-mute", fb.inkMute),
    inkFaint: readCssColor("--ink-faint", fb.inkFaint),
    indigo: readCssColor("--pigment-indigo", fb.indigo),
    indigoDeep: readCssColor("--pigment-indigo-deep", fb.indigoDeep),
    sage: readCssColor("--pigment-sage", fb.sage),
    ochre: readCssColor("--pigment-ochre", fb.ochre),
    madder: readCssColor("--pigment-madder", fb.madder),
    paynes: readCssColor("--pigment-paynes", fb.paynes),
    rose: readCssColor("--pigment-rose", fb.rose),
    indigoBright: readCssColor("--pigment-indigo-bright", fb.indigoBright),
    indigoDeepBright: readCssColor("--pigment-indigo-deep-bright", fb.indigoDeepBright),
    sageBright: readCssColor("--pigment-sage-bright", fb.sageBright),
    ochreBright: readCssColor("--pigment-ochre-bright", fb.ochreBright),
    madderBright: readCssColor("--pigment-madder-bright", fb.madderBright),
    paynesBright: readCssColor("--pigment-paynes-bright", fb.paynesBright),
    roseBright: readCssColor("--pigment-rose-bright", fb.roseBright),
    indigoDark: readCssColor("--pigment-indigo-dark", fb.indigoDark),
    indigoDeepDark: readCssColor("--pigment-indigo-deep-dark", fb.indigoDeepDark),
    sageDark: readCssColor("--pigment-sage-dark", fb.sageDark),
    ochreDark: readCssColor("--pigment-ochre-dark", fb.ochreDark),
    madderDark: readCssColor("--pigment-madder-dark", fb.madderDark),
    paynesDark: readCssColor("--pigment-paynes-dark", fb.paynesDark),
    roseDark: readCssColor("--pigment-rose-dark", fb.roseDark),
    paper: readCssColor("--plot-marker-rim", fb.paper),
    plate: readCssColor("--paper-cool", fb.plate),
    rule: readCssColor("--rule", fb.rule),
    grid: readCssColor("--plot-grid", fb.grid),
    annotationBg: readCssColor("--plot-annotation-bg", fb.annotationBg),
    oracleBand99: readCssColor("--plot-oracle-band-99", fb.oracleBand99),
    oracleBand95: readCssColor("--plot-oracle-band-95", fb.oracleBand95),
    bandSage: readCssColor("--plot-band-sage", fb.bandSage),
    bandMadder: readCssColor("--plot-band-madder", fb.bandMadder),
  };
}

/** Categorical trace order: ink, then botanical pigments. */
export function getInkColorway(palette: InkPalette = getInkPalette()): string[] {
  return [palette.ink, palette.indigo, palette.madder, palette.sage, palette.ochre, palette.paynes];
}

/** Continuous scalar wash for heatmaps / density fields. */
export function getInkColorscale(): Array<[number, string]> {
  const fb = LIGHT_FALLBACK;
  return [
    [0.0, readCssColor("--plot-colorscale-0", fb.plate)],
    [0.25, readCssColor("--plot-colorscale-25", "#b7c6d6")],
    [0.5, readCssColor("--plot-colorscale-50", fb.indigo)],
    [0.75, readCssColor("--plot-colorscale-75", "#2f4257")],
    [1.0, readCssColor("--plot-colorscale-100", fb.ink)],
  ];
}

/** Method trace styling (sampled, oracle, fit, etc.). */
export function getMethodStyle(palette: InkPalette = getInkPalette()): MethodStyle {
  return {
    sampled: { color: palette.indigoDeep, symbol: "circle" },
    oracle: { color: palette.ink, dash: "dash" },
    oracleBand99: { color: palette.oracleBand99, dash: "solid" },
    oracleBand95: { color: palette.oracleBand95, dash: "dash" },
    rare: { color: palette.madder, dash: "dot" },
    mcub: { color: palette.sage, dash: "dashdot" },
    fit: { color: palette.ochre, dash: "dot" },
    parity: { color: palette.ink, dash: "dash" },
    selected: { color: palette.madder, symbol: "diamond-open" },
  };
}

export function getPlotFontStacks(): { sans: string; mono: string } {
  return {
    sans: readCssVar("--sans", "'Inter Tight', sans-serif"),
    mono: readCssVar("--mono", "'IBM Plex Mono', monospace"),
  };
}

/** Resolved font-family stack for MathBox GL tick labels (from `.plot-mathbox`). */
export function readPlotTickFont(element?: Element | null): string {
  if (typeof document !== "undefined" && element) {
    const value = getComputedStyle(element).getPropertyValue("--plot-tick-font").trim();
    if (value) return value;
  }
  return getPlotFontStacks().mono;
}

/** Re-read tokens when the OS color scheme changes (dark mode). */
export function usePlotTheme(): PlotTheme {
  const scheme = useSyncExternalStore(subscribeToColorScheme, getColorSchemeSnapshot, () => "light");
  return useMemo(() => {
    const palette = getInkPalette();
    const { sans, mono } = getPlotFontStacks();
    const resolvedScheme: PlotTheme["scheme"] = scheme === "dark" ? "dark" : "light";
    return {
      scheme: resolvedScheme,
      palette,
      colorway: getInkColorway(palette),
      colorscale: getInkColorscale(),
      methodStyle: getMethodStyle(palette),
      sans,
      mono,
    };
  }, [scheme]);
}

/** Apply alpha to hex or rgb()/rgba() colors returned from getComputedStyle. */
export function withAlpha(color: string, alpha: number): string {
  const trimmed = color.trim();
  if (trimmed.startsWith("#") && trimmed.length === 7) {
    const r = Number.parseInt(trimmed.slice(1, 3), 16);
    const g = Number.parseInt(trimmed.slice(3, 5), 16);
    const b = Number.parseInt(trimmed.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  const rgb = trimmed.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
  if (rgb) return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${alpha})`;
  return trimmed;
}

/** Multiply RGB toward black (`factor` in [0,1]; 0.65 -> 35% darker). */
export function darken(color: string, factor: number): string {
  const trimmed = color.trim();
  const scale = (value: number) => Math.max(0, Math.min(255, Math.round(value * factor)));
  if (trimmed.startsWith("#") && trimmed.length === 7) {
    const r = Number.parseInt(trimmed.slice(1, 3), 16);
    const g = Number.parseInt(trimmed.slice(3, 5), 16);
    const b = Number.parseInt(trimmed.slice(5, 7), 16);
    return `rgb(${scale(r)}, ${scale(g)}, ${scale(b)})`;
  }
  const rgb = trimmed.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
  if (rgb) return `rgb(${scale(Number(rgb[1]))}, ${scale(Number(rgb[2]))}, ${scale(Number(rgb[3]))})`;
  return trimmed;
}
