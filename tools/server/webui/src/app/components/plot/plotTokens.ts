import { useMemo, useSyncExternalStore } from "react";

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
  inkMute: "#58524d",
  inkFaint: "#94897e",
  indigo: "#6b7f96",
  indigoDeep: "#4a5e76",
  sage: "#8ba189",
  ochre: "#b39866",
  madder: "#a8807a",
  paynes: "#555a64",
  rose: "#d5b9b1",
  indigoBright: "#2f6ba8",
  indigoDeepBright: "#21528a",
  sageBright: "#4a9d5f",
  ochreBright: "#c8942c",
  madderBright: "#c6505d",
  paynesBright: "#4e6a95",
  roseBright: "#df7d8b",
  indigoDark: "#244466",
  indigoDeepDark: "#1a3451",
  sageDark: "#285d32",
  ochreDark: "#745425",
  madderDark: "#722c30",
  paynesDark: "#2b3850",
  roseDark: "#7e4b44",
  paper: "#ece8e4",
  plate: "#faf7f3",
  rule: "rgba(34, 32, 30, 0.92)",
  grid: "rgba(34, 32, 30, 0.08)",
  annotationBg: "rgba(250, 247, 243, 0.72)",
  oracleBand99: "rgba(34, 32, 30, 0.48)",
  oracleBand95: "rgba(34, 32, 30, 0.58)",
  bandSage: "rgba(139, 161, 137, 0.12)",
  bandMadder: "rgba(168, 128, 122, 0.10)",
};

export const PLOT_TRANSPARENT = "transparent";

function readCssVar(name: `--${string}`, fallback = ""): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
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

/** Read the current plate palette from design-language.css custom properties. */
export function getInkPalette(): InkPalette {
  const fb = LIGHT_FALLBACK;
  return {
    ink: readCssVar("--ink", fb.ink),
    inkSoft: readCssVar("--ink-2", fb.inkSoft),
    inkMute: readCssVar("--ink-mute", fb.inkMute),
    inkFaint: readCssVar("--ink-faint", fb.inkFaint),
    indigo: readCssVar("--pigment-indigo", fb.indigo),
    indigoDeep: readCssVar("--pigment-indigo-deep", fb.indigoDeep),
    sage: readCssVar("--pigment-sage", fb.sage),
    ochre: readCssVar("--pigment-ochre", fb.ochre),
    madder: readCssVar("--pigment-madder", fb.madder),
    paynes: readCssVar("--pigment-paynes", fb.paynes),
    rose: readCssVar("--pigment-rose", fb.rose),
    indigoBright: readCssVar("--pigment-indigo-bright", fb.indigoBright),
    indigoDeepBright: readCssVar("--pigment-indigo-deep-bright", fb.indigoDeepBright),
    sageBright: readCssVar("--pigment-sage-bright", fb.sageBright),
    ochreBright: readCssVar("--pigment-ochre-bright", fb.ochreBright),
    madderBright: readCssVar("--pigment-madder-bright", fb.madderBright),
    paynesBright: readCssVar("--pigment-paynes-bright", fb.paynesBright),
    roseBright: readCssVar("--pigment-rose-bright", fb.roseBright),
    indigoDark: readCssVar("--pigment-indigo-dark", fb.indigoDark),
    indigoDeepDark: readCssVar("--pigment-indigo-deep-dark", fb.indigoDeepDark),
    sageDark: readCssVar("--pigment-sage-dark", fb.sageDark),
    ochreDark: readCssVar("--pigment-ochre-dark", fb.ochreDark),
    madderDark: readCssVar("--pigment-madder-dark", fb.madderDark),
    paynesDark: readCssVar("--pigment-paynes-dark", fb.paynesDark),
    roseDark: readCssVar("--pigment-rose-dark", fb.roseDark),
    paper: readCssVar("--plot-marker-rim", fb.paper),
    plate: readCssVar("--paper-cool", fb.plate),
    rule: readCssVar("--rule", fb.rule),
    grid: readCssVar("--plot-grid", fb.grid),
    annotationBg: readCssVar("--plot-annotation-bg", fb.annotationBg),
    oracleBand99: readCssVar("--plot-oracle-band-99", fb.oracleBand99),
    oracleBand95: readCssVar("--plot-oracle-band-95", fb.oracleBand95),
    bandSage: readCssVar("--plot-band-sage", fb.bandSage),
    bandMadder: readCssVar("--plot-band-madder", fb.bandMadder),
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
    [0.0, readCssVar("--plot-colorscale-0", fb.plate)],
    [0.25, readCssVar("--plot-colorscale-25", "#b8c1cb")],
    [0.5, readCssVar("--plot-colorscale-50", fb.indigo)],
    [0.75, readCssVar("--plot-colorscale-75", "#3a4658")],
    [1.0, readCssVar("--plot-colorscale-100", fb.ink)],
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
