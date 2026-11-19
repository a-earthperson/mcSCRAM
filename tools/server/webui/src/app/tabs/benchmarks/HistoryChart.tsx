import { useMemo } from "react";
import { formatNumber } from "../../lib/format";
import { flagSummary } from "./compare";
import type { BenchFlags } from "./types";

export type ChartPoint = {
  version: string;
  created_utc: string | null;
  value: number | null;
  flags: BenchFlags;
  channel: string;
};

type HistoryChartProps = {
  title: string;
  unit: string;
  points: ChartPoint[];
};

export function HistoryChart({ title, unit, points }: HistoryChartProps) {
  const series = useMemo(
    () => points.map((p, i) => ({ ...p, i, y: p.value })).filter((p) => p.y != null) as Array<ChartPoint & { i: number; y: number }>,
    [points],
  );
  const width = 640;
  const height = 220;
  const pad = { l: 52, r: 16, t: 16, b: 44 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  if (series.length === 0) {
    return (
      <div className="bench-chart bench-chart--empty">
        <p className="eyebrow">{title}</p>
        <p>No samples for this metric.</p>
      </div>
    );
  }
  const ys = series.map((p) => p.y);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) {
    yMin -= Math.abs(yMin) * 0.05 + 0.01;
    yMax += Math.abs(yMax) * 0.05 + 0.01;
  }
  const xAt = (i: number) => pad.l + (points.length <= 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const yAt = (y: number) => pad.t + ((yMax - y) / (yMax - yMin)) * innerH;
  const d = series
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xAt(p.i).toFixed(1)} ${yAt(p.y).toFixed(1)}`)
    .join(" ");
  return (
    <figure className="bench-chart">
      <figcaption>
        <span className="eyebrow">{title}</span>
        <span className="bench-chart__unit">{unit}</span>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} over versions`}>
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + innerH} className="bench-chart__axis" />
        <line x1={pad.l} y1={pad.t + innerH} x2={pad.l + innerW} y2={pad.t + innerH} className="bench-chart__axis" />
        <text x={4} y={pad.t + 4} className="bench-chart__tick">{formatNumber(yMax, 3)}</text>
        <text x={4} y={pad.t + innerH} className="bench-chart__tick">{formatNumber(yMin, 3)}</text>
        <path d={d} className="bench-chart__line" fill="none" />
        {series.map((p) => {
          const flags = flagSummary(p.flags);
          return (
            <g key={`${p.version}-${p.created_utc}-${p.i}`}>
              <circle
                cx={xAt(p.i)}
                cy={yAt(p.y)}
                r={4.5}
                className={p.channel === "develop" ? "bench-chart__dot bench-chart__dot--develop" : "bench-chart__dot"}
              >
                <title>
                  {p.version}: {formatNumber(p.y, 4)} {unit}
                  {flags.length ? ` · changed ${flags.join(", ")}` : ""}
                </title>
              </circle>
              {flags.length ? (
                <text x={xAt(p.i)} y={yAt(p.y) - 10} className="bench-chart__flag" textAnchor="middle">
                  *
                </text>
              ) : null}
            </g>
          );
        })}
        {points.map((p, i) => (
          <text key={`${p.version}-x-${i}`} x={xAt(i)} y={height - 12} className="bench-chart__xlabel" textAnchor="middle">
            {p.version}
          </text>
        ))}
      </svg>
    </figure>
  );
}
