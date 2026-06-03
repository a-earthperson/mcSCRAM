import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Plot, usePlotTheme } from "../../../components/Plot";
import { PLOT_TRANSPARENT, type PlotTrace, withAlpha } from "../../../components/plotDesign";
import {
  DEFAULT_WALD_ALPHA,
  confidenceFromAlpha,
  normalQuantileTwoSided,
  type ConvergenceStatsPayload,
} from "../../../lib/liveConvergenceStats";
import { formatInteger, formatNumber } from "../../../lib/format";
import { EmptyState, SectionTitle } from "../../ui";
import { mcscramWebSocketUrl } from "../../mcscramClient";
import type { SamplingStatus } from "./controls/samplingStatus";
import type { SamplingControls } from "./controls/useSamplingControls";
import type { SamplingRunMetricsStore } from "./samplingRunMetrics";

type LiveProgressUpdate = ConvergenceStatsPayload;

type LiveResult = {
  analysis_time?: number;
  warnings?: string;
  probabilities?: Array<{
    p_total?: number;
    analysis_time?: number;
  }>;
};

type LiveStatus = Omit<SamplingStatus, "last_update" | "last_result"> & {
  last_update: LiveProgressUpdate | null;
  last_result: LiveResult | null;
};

type LiveEvent =
  | { event: "snapshot"; status: LiveStatus }
  | { event: "server_started_calculation"; pid?: number; input_files?: string[]; options?: Record<string, unknown> }
  | { event: "model_loaded"; model: unknown }
  | { event: "calculation_reset" }
  | { event: "calculation_stopped"; status: LiveStatus }
  | { event: "progress"; sequence: number; timestamp: number; update: LiveProgressUpdate }
  | { event: "calculation_completed"; result: LiveResult }
  | { event: "calculation_failed"; error: string };

type ConnectionState = "connecting" | "open" | "closed" | "error";

type LiveProbabilityPoint = {
  iterations?: number;
  trials: number;
  mean: number;
  p01?: number;
  p05?: number;
  p95?: number;
  p99?: number;
  epsilon?: number;
  oracleP?: number;
};

const MAX_REPLAY_PATH_POINTS = 10_000;
const SAMPLE_RATE_WINDOW = 12;

export function SamplingTrajectoryPanel({
  samplingControls,
  runMetricsStore,
}: {
  samplingControls: SamplingControls | null;
  runMetricsStore: SamplingRunMetricsStore;
}) {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [replayPathPoints, setReplayPathPoints] = useState<LiveProbabilityPoint[]>([]);
  const [streamError, setStreamError] = useState<string | null>(null);
  const samplingControlsRef = useRef(samplingControls);
  const runMetricsStoreRef = useRef(runMetricsStore);
  const pointsRef = useRef<LiveProbabilityPoint[]>([]);
  const pendingUpdatesRef = useRef<LiveProgressUpdate[]>([]);
  const sampleWindowRef = useRef<Array<{ trials: number; timeMs: number }>>([]);
  const latestSampleRateRef = useRef<number | null>(null);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    samplingControlsRef.current = samplingControls;
  }, [samplingControls]);
  useEffect(() => {
    runMetricsStoreRef.current = runMetricsStore;
  }, [runMetricsStore]);

  const flushPendingUpdates = () => {
    frameRef.current = null;
    const pending = pendingUpdatesRef.current;
    if (!pending.length) return;
    pendingUpdatesRef.current = [];
    const next = appendProbabilityUpdates(pointsRef.current, pending, MAX_REPLAY_PATH_POINTS);
    const latestUpdate = pending[pending.length - 1];
    runMetricsStoreRef.current.setMetrics({
      trialsPerSecond: latestSampleRateRef.current,
      completedPasses: isFiniteNumber(latestUpdate.iterations) ? latestUpdate.iterations : null,
    });
    if (next !== pointsRef.current) {
      pointsRef.current = next;
      setReplayPathPoints(next);
    }
  };

  const scheduleUpdateFlush = (update: LiveProgressUpdate) => {
    pendingUpdatesRef.current.push(update);
    updateSampleRate(update);
    if (frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(flushPendingUpdates);
    }
  };

  const clearReplayPath = () => {
    pendingUpdatesRef.current = [];
    sampleWindowRef.current = [];
    latestSampleRateRef.current = null;
    pointsRef.current = [];
    setReplayPathPoints([]);
    runMetricsStoreRef.current.setMetrics({ trialsPerSecond: null, completedPasses: null });
    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  };

  const updateSampleRate = (update: LiveProgressUpdate) => {
    if (!isPositiveNumber(update.trials)) return;
    sampleWindowRef.current = [
      ...sampleWindowRef.current,
      { trials: update.trials, timeMs: performance.now() },
    ].slice(-SAMPLE_RATE_WINDOW);
    const first = sampleWindowRef.current[0];
    const last = sampleWindowRef.current[sampleWindowRef.current.length - 1];
    if (!first || !last || last.trials < first.trials || last.timeMs <= first.timeMs) {
      latestSampleRateRef.current = null;
      return;
    }
    const deltaSeconds = (last.timeMs - first.timeMs) / 1000;
    latestSampleRateRef.current = deltaSeconds > 0 ? (last.trials - first.trials) / deltaSeconds : null;
  };

  useEffect(() => {
    const wsUrl = mcscramWebSocketUrl({ replay: true });
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let stopped = false;
    let retryDelayMs = 500;

    const setControlError = (error: string | null) => {
      setStreamError(error);
      samplingControlsRef.current?.setError(error);
    };

    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket(wsUrl);

      socket.addEventListener("open", () => {
        retryDelayMs = 500;
        setConnection("open");
        setControlError(null);
      });

      socket.addEventListener("message", (message) => {
        const event = JSON.parse(message.data) as LiveEvent;
        const controls = samplingControlsRef.current;

        if (event.event === "snapshot") {
          controls?.setStatus(event.status);
          setControlError(event.status.last_error);
          clearReplayPath();
          return;
        }
        if (event.event === "calculation_reset" || event.event === "server_started_calculation") {
          clearReplayPath();
          setControlError(null);
          return;
        }
        if (event.event === "calculation_stopped") {
          controls?.setStatus(event.status);
          return;
        }
        if (event.event === "calculation_completed") {
          controls?.setStatus((previous) => completedStatus(previous, event.result));
          return;
        }
        if (event.event === "calculation_failed") {
          controls?.setStatus((previous) => failedStatus(previous, event.error));
          setControlError(event.error);
          return;
        }
        if (event.event === "progress") {
          if (controls?.clientPausedRef.current) return;
          scheduleUpdateFlush(event.update);
          setControlError(null);
        }
      });

      socket.addEventListener("error", () => {
        setConnection("error");
        setControlError(`Could not connect to ${wsUrl}; retrying...`);
      });

      socket.addEventListener("close", () => {
        if (stopped) return;
        setConnection("closed");
        reconnectTimer = window.setTimeout(connect, retryDelayMs);
        retryDelayMs = Math.min(retryDelayMs * 2, 5_000);
      });
    };

    connect();

    return () => {
      stopped = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
      socket?.close();
    };
  }, []);

  return (
    <div className="panel span-3">
      <SectionTitle
        eyebrow="Output: Cumulative Trials since Start"
        title="Full sampled probability trajectory with cumulative trials"
        text="This plot reconstructs the sampled probabilities since the start of the run."
      />
      {streamError ? <p className="live-demo-error playground-sampling-error">{streamError}</p> : null}
      {replayPathPoints.length ? (
        <LiveProbabilityPathPlot points={replayPathPoints} xMode="absolute" clientAlpha={DEFAULT_WALD_ALPHA} />
      ) : (
        <EmptyState text={connection === "open" ? "Waiting for replayed probability points." : "Connecting to the sampling stream..."} />
      )}
    </div>
  );
}

const LiveProbabilityPathPlot = memo(function LiveProbabilityPathPlot({
  points,
  xMode,
  clientAlpha,
}: {
  points: LiveProbabilityPoint[];
  xMode: "window" | "absolute";
  clientAlpha: number;
}) {
  const { palette: inkPalette } = usePlotTheme();
  const { data, layout } = useMemo(() => {
    const palette = xMode === "absolute"
      ? {
        line: inkPalette.madder,
        band: inkPalette.ochre,
        marker: inkPalette.madder,
        legendgroup: "replay",
      }
      : {
        line: inkPalette.indigoDeep,
        band: inkPalette.indigoDeep,
        marker: inkPalette.indigoDeep,
        legendgroup: "live",
      };
    const firstTrials = points[0]?.trials ?? 0;
    const x = xMode === "window"
      ? points.map((point) => Math.max(0, point.trials - firstTrials))
      : points.map((point) => point.trials);
    const oracleP = points.find((point) => isPositiveNumber(point.oracleP))?.oracleP;
    const band99 = liveConfidenceBandTrace({
      name: "sampled p01-p99",
      x,
      lower: points.map((point) => point.p01),
      upper: points.map((point) => point.p99),
      color: palette.band,
      opacity: 0.08,
      legendgroup: palette.legendgroup,
    });
    const band90 = liveConfidenceBandTrace({
      name: "sampled p05-p95",
      x,
      lower: points.map((point) => point.p05),
      upper: points.map((point) => point.p95),
      color: palette.band,
      opacity: 0.16,
      legendgroup: palette.legendgroup,
    });
    const clientConfidence = confidenceFromAlpha(clientAlpha);
    const clientConfidenceLabel = formatNumber(clientConfidence * 100, 4);
    const clientAlphaZ = normalQuantileTwoSided(clientConfidence);
    const clientAlphaLower = points.map((point) => {
      const stdErr = Math.sqrt((point.mean * (1 - point.mean)) / point.trials);
      return Math.max(0, point.mean - clientAlphaZ * stdErr);
    });
    const clientAlphaUpper = points.map((point) => {
      const stdErr = Math.sqrt((point.mean * (1 - point.mean)) / point.trials);
      return Math.min(1, point.mean + clientAlphaZ * stdErr);
    });
    const clientAlphaBand = liveConfidenceBandTrace({
      name: `client ${clientConfidenceLabel}% confidence interval`,
      x,
      lower: clientAlphaLower,
      upper: clientAlphaUpper,
      color: inkPalette.sage,
      opacity: 0.14,
      legendgroup: palette.legendgroup,
    });
    const oracleLines = oracleReferenceTraces(points, x, oracleP, xMode, clientAlpha, inkPalette.ink);
    const yValues = [
      ...points.flatMap((point) => [point.mean, point.p01, point.p05, point.p95, point.p99]),
      ...(Array.isArray(clientAlphaBand?.y) ? clientAlphaBand.y as Array<number | null | undefined> : []),
      ...oracleLines.flatMap((trace) => Array.isArray(trace.y) ? trace.y as Array<number | null | undefined> : []),
    ];
    const meanLine: PlotTrace = {
      type: "scatter",
      mode: "lines+markers",
      name: "live mean",
      x,
      y: points.map((point) => point.mean),
      customdata: points.map((point) => [
        point.iterations ?? null,
        formatInteger(point.trials),
        formatNumber(point.p05, 4),
        formatNumber(point.p95, 4),
        formatNumber(point.epsilon, 4),
      ]),
      line: { color: palette.line, width: 1.8 },
      marker: { color: palette.marker, size: 4 },
      legendgroup: palette.legendgroup,
      hovertemplate: `${xMode === "window" ? "additional" : "total"} trials=%{x:.3s}<br>total trials=%{customdata[1]}<br>mean=%{y:.4e}<br>iterations=%{customdata[0]}<br>p05=%{customdata[2]}<br>p95=%{customdata[3]}<br>epsilon=%{customdata[4]}<extra>live run</extra>`,
    };

    return {
      data: [band99, band90, clientAlphaBand, ...oracleLines, meanLine].filter(Boolean) as PlotTrace[],
      layout: {
        xaxis: {
          title: {
            text: xMode === "window"
              ? "Additional Bernoulli trials since first streamed sample"
              : "Cumulative Bernoulli trials",
            standoff: 10,
          },
          rangemode: "tozero",
          tickformat: "~s",
        },
        yaxis: {
          title: { text: "Probability estimate", standoff: 12 },
          tickformat: ".3e",
          exponentformat: "e",
          showexponent: "all",
          range: linearAxisRange(yValues, { floor: 0, ceiling: 1, pad: 0.1 }),
        },
        legend: { y: -0.28, orientation: "h", font: { size: 10 } },
        margin: { l: 86, r: 18, t: 20, b: 112 },
      },
    };
  }, [clientAlpha, inkPalette, points, xMode]);

  return (
    <Plot
      className="plot convergence-large-plot live-probability-plot"
      data={data}
      layout={layout}
    />
  );
});

function completedStatus(previous: SamplingStatus | null, result: LiveResult): SamplingStatus {
  const now = Date.now() / 1000;
  return {
    status: "completed",
    pid: previous?.pid ?? null,
    alive: false,
    started_at: previous?.started_at ?? null,
    finished_at: now,
    update_count: previous?.update_count ?? 0,
    last_update: previous?.last_update ?? null,
    last_result: result,
    last_error: null,
  };
}

function failedStatus(previous: SamplingStatus | null, error: string): SamplingStatus {
  const now = Date.now() / 1000;
  return {
    status: "failed",
    pid: previous?.pid ?? null,
    alive: false,
    started_at: previous?.started_at ?? null,
    finished_at: now,
    update_count: previous?.update_count ?? 0,
    last_update: previous?.last_update ?? null,
    last_result: previous?.last_result ?? null,
    last_error: error,
  };
}

function probabilityPoint(update: LiveProgressUpdate | null | undefined): LiveProbabilityPoint | null {
  if (!isPositiveNumber(update?.trials) || !isPositiveNumber(update?.mean)) return null;
  return {
    iterations: update.iterations,
    trials: update.trials,
    mean: update.mean,
    p01: update.p01,
    p05: update.p05,
    p95: update.p95,
    p99: update.p99,
    epsilon: update.epsilon,
    oracleP: update.oracle_p,
  };
}

function appendProbabilityUpdates(
  points: LiveProbabilityPoint[],
  updates: LiveProgressUpdate[],
  maxPoints: number,
): LiveProbabilityPoint[] {
  let next = points;
  let lastKey = points.length ? probabilityPointKey(points[points.length - 1]) : "";

  for (const update of updates) {
    const point = probabilityPoint(update);
    if (!point) continue;
    const key = probabilityPointKey(point);
    if (key === lastKey) continue;
    if (next === points) next = points.slice();
    next.push(point);
    lastKey = key;
  }

  if (next === points) return points;
  return next.length > maxPoints ? next.slice(-maxPoints) : next;
}

function probabilityPointKey(point: LiveProbabilityPoint): string {
  return `${point.iterations ?? ""}:${point.trials}`;
}

function liveConfidenceBandTrace({
  name,
  x,
  lower,
  upper,
  color,
  opacity,
  legendgroup,
}: {
  name: string;
  x: number[];
  lower: Array<number | null | undefined>;
  upper: Array<number | null | undefined>;
  color: string;
  opacity: number;
  legendgroup?: string;
}): PlotTrace | null {
  const rows = x
    .map((value, index) => ({ x: value, lo: lower[index], hi: upper[index] }))
    .filter((point): point is { x: number; lo: number; hi: number } => Number.isFinite(point.x) && point.x >= 0 && isPositiveNumber(point.lo) && isPositiveNumber(point.hi));
  if (rows.length < 2) return null;
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: [...rows.map((point) => point.x), ...rows.map((point) => point.x).reverse()],
    y: [...rows.map((point) => point.hi), ...rows.map((point) => point.lo).reverse()],
    fill: "toself",
    fillcolor: withAlpha(color, opacity),
    line: { color: PLOT_TRANSPARENT, width: 0 },
    hoverinfo: "skip",
    legendgroup,
  };
}

function oracleReferenceTraces(
  points: LiveProbabilityPoint[],
  x: number[],
  oracleP: number | null | undefined,
  xMode: "window" | "absolute",
  clientAlpha: number,
  inkColor: string,
): PlotTrace[] {
  if (!isPositiveNumber(oracleP)) return [];
  const rows = points
    .map((point, index) => ({ x: x[index], totalTrials: point.trials }))
    .filter((point) => Number.isFinite(point.x) && point.x >= 0 && isPositiveNumber(point.totalTrials));
  if (rows.length < 2) return [];

  const xs = rows.map((row) => row.x);
  const ciRows = xMode === "window" ? rows.filter((row) => row.x > 0) : rows;
  const effectiveTrials = (row: { x: number; totalTrials: number }) => xMode === "window" ? row.x : row.totalTrials;
  const halfWidth = (trials: number, z: number) => z * Math.sqrt((oracleP * (1 - oracleP)) / trials + 1e-15);
  const upper = (z: number) => ciRows.map((row) => Math.min(1, oracleP + halfWidth(effectiveTrials(row), z)));
  const lower = (z: number) => ciRows.map((row) => Math.max(0, oracleP - halfWidth(effectiveTrials(row), z)));
  const oracleConfidence = confidenceFromAlpha(clientAlpha);
  const oracleZ = normalQuantileTwoSided(oracleConfidence);
  const oracleConfidenceLabel = formatNumber(oracleConfidence * 100, 4);
  const line = (name: string, ys: number[], dash: string, width = 1.2): PlotTrace => ({
    type: "scatter",
    mode: "lines",
    name,
    x: ys.length === rows.length ? xs : ciRows.map((row) => row.x),
    y: ys,
    line: { color: inkColor, dash, width },
    hovertemplate: `${name}<br>${xMode === "window" ? "additional" : "total"} trials=%{x:.3s}<br>p=%{y:.4e}<extra>oracle</extra>`,
  });

  return [
    line("Oracle p", rows.map(() => oracleP), "dash", 1.8),
    line("Oracle 99% CI upper", upper(2.576), "solid", 1),
    line("Oracle 99% CI lower", lower(2.576), "solid", 1),
    line("Oracle 95% CI upper", upper(1.96), "dash", 1),
    line("Oracle 95% CI lower", lower(1.96), "dash", 1),
    line(`Oracle ${oracleConfidenceLabel}% confidence upper`, upper(oracleZ), "dashdot", 1.1),
    line(`Oracle ${oracleConfidenceLabel}% confidence lower`, lower(oracleZ), "dashdot", 1.1),
  ];
}

function linearAxisRange(values: Array<number | null | undefined>, options: { floor?: number; ceiling?: number; pad?: number } = {}): [number, number] | undefined {
  const present = values.filter(isFiniteNumber).map((value) => {
    let next = value;
    if (options.floor !== undefined) next = Math.max(next, options.floor);
    if (options.ceiling !== undefined) next = Math.min(next, options.ceiling);
    return next;
  });
  if (!present.length) return undefined;
  let min = Math.min(...present);
  let max = Math.max(...present);
  if (min === max) {
    const delta = Math.max(Math.abs(min) * 0.1, 1e-12);
    return [Math.max(options.floor ?? -Infinity, min - delta), Math.min(options.ceiling ?? Infinity, max + delta)];
  }
  const span = max - min;
  const pad = options.pad ?? 0.08;
  return [
    Math.max(options.floor ?? -Infinity, min - span * pad),
    Math.min(options.ceiling ?? Infinity, max + span * pad),
  ];
}

function isPositiveNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value) && value > 0;
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value);
}
