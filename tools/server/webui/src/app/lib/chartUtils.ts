import type { DeviceSummary, LogFit, PerformanceRun } from "../types";
import { DEVICE_ORDER } from "../theme/constants";

export function buildErrorVsTrialsByDevice(runs: PerformanceRun[]): Record<string, { x: number[]; y: number[]; text: string[] }> {
  const out: Record<string, { x: number[]; y: number[]; text: string[] }> = {};
  for (const run of runs) {
    if (!run.totalBits || !run.relError || run.relError <= 0) continue;
    const bucket = out[run.deviceShort] ?? (out[run.deviceShort] = { x: [], y: [], text: [] });
    bucket.x.push(run.totalBits);
    bucket.y.push(run.relError);
    bucket.text.push(`${run.modelName} · ${run.deviceShort} · ${run.compileFlag}`);
  }
  return out;
}

export function fitLine(
  fit: LogFit | undefined,
  color: string,
  name: string,
  xRange?: [number, number],
): Record<string, unknown> | null {
  if (!fit || fit.slope === null || fit.intercept === null) return null;
  const [xMin, xMax] = xRange ?? [10, 100_000];
  const logMin = Math.log10(Math.max(xMin, 1));
  const logMax = Math.log10(Math.max(xMax, xMin * 10));
  const xs: number[] = [];
  const steps = 16;
  for (let i = 0; i <= steps; i += 1) {
    xs.push(Math.pow(10, logMin + ((logMax - logMin) * i) / steps));
  }
  const ys = xs.map((x) => Math.pow(10, fit.slope! * Math.log10(x) + fit.intercept!));
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: xs,
    y: ys,
    line: { color, dash: "dot", width: 1.5 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

export function formatFitSlope(fit: LogFit | undefined): string {
  if (!fit || fit.slope === null) return "n/a";
  return fit.slope.toFixed(2);
}

export function orderDevices(devices: DeviceSummary[]): DeviceSummary[] {
  return [...devices].sort((a, b) => {
    const ai = DEVICE_ORDER.indexOf(a.deviceShort);
    const bi = DEVICE_ORDER.indexOf(b.deviceShort);
    if (ai === -1 && bi === -1) return a.deviceShort.localeCompare(b.deviceShort);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}

export function groupBy<T, K extends string>(rows: T[], pick: (row: T) => K | null | undefined): Record<K, T[]> {
  const out = {} as Record<K, T[]>;
  for (const row of rows) {
    const key = pick(row);
    if (!key) continue;
    if (!out[key]) out[key] = [];
    out[key].push(row);
  }
  return out;
}

export function medianOf(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number => value !== null && value !== undefined && !Number.isNaN(value)).sort((a, b) => a - b);
  if (!present.length) return null;
  const middle = Math.floor(present.length / 2);
  return present.length % 2 ? present[middle] : (present[middle - 1] + present[middle]) / 2;
}
