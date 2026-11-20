import { useEffect, useRef } from "react";
import { PLOT_TRANSPARENT, usePlotTheme } from "./plotTokens";

export {
  getInkPalette,
  getInkColorway,
  getInkColorscale,
  getMethodStyle,
  usePlotTheme,
  withAlpha,
  PLOT_TRANSPARENT,
  type InkPalette,
  type MethodStyle,
  type PlotTheme,
} from "./plotTokens";

type PlotProps = {
  data: unknown[];
  layout?: Record<string, any>;
  config?: Record<string, unknown>;
  className?: string;
};

const defaultConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

type PlotlyApi = typeof import("plotly.js-dist-min").default;

let plotlyPromise: Promise<PlotlyApi> | null = null;

function getPlotly(): Promise<PlotlyApi> {
  plotlyPromise ??= import("plotly.js-dist-min").then((module) => module.default);
  return plotlyPromise;
}

export function Plot({ data, layout, config, className }: PlotProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const { palette, colorway, sans, mono } = usePlotTheme();

  useEffect(() => {
    const element = ref.current;
    let cancelled = false;
    if (!element) return undefined;

    const axisDefaults = {
      showline: true,
      linecolor: palette.ink,
      linewidth: 1,
      mirror: false,
      ticks: "outside" as const,
      tickcolor: palette.ink,
      ticklen: 4,
      tickwidth: 1,
      tickfont: { family: mono, color: palette.ink, size: 11 },
      gridcolor: palette.grid,
      gridwidth: 1,
      zeroline: false,
      automargin: true,
      titlefont: { family: sans, color: palette.ink, size: 12 },
    };

    const legendDefaults = {
      orientation: "h" as const,
      y: -0.22,
      font: { family: sans, color: palette.ink, size: 12 },
      bgcolor: PLOT_TRANSPARENT,
      bordercolor: PLOT_TRANSPARENT,
    };

    getPlotly().then((Plotly) => {
      if (cancelled) return;
      const incoming = layout ?? {};
      const merged: Record<string, any> = {
        autosize: true,
        margin: { l: 64, r: 24, t: 22, b: 56 },
        paper_bgcolor: PLOT_TRANSPARENT,
        plot_bgcolor: PLOT_TRANSPARENT,
        font: { family: sans, color: palette.ink, size: 12 },
        colorway,
        hoverlabel: {
          bgcolor: palette.plate,
          bordercolor: palette.ink,
          font: { family: mono, color: palette.ink, size: 12 },
          align: "left" as const,
        },
        ...incoming,
        xaxis: { ...axisDefaults, ...(incoming.xaxis ?? {}) },
        yaxis: { ...axisDefaults, ...(incoming.yaxis ?? {}) },
        legend: { ...legendDefaults, ...(incoming.legend ?? {}) },
      };
      Plotly.react(element, data, merged, { ...defaultConfig, ...config });
    });
    return () => {
      cancelled = true;
      getPlotly().then((Plotly) => Plotly.purge(element));
    };
  }, [data, layout, config, palette, colorway, sans, mono]);

  return <div ref={ref} className={className ?? "plot"} />;
}
