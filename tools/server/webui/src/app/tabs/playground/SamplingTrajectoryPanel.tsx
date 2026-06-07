import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Plot, usePlotTheme } from "../../components/plot/Plot";
import {
  logAxisRange,
  PLOT_TRANSPARENT,
  type PlotTrace,
  withAlpha,
} from "../../components/plot/plotDesign";
import {
  confidenceFromAlpha,
  normalQuantileTwoSided,
  type ConvergenceStatsPayload,
} from "../../lib/liveConvergenceStats";
import { formatInteger, formatNumber } from "../../lib/format";
import { EmptyState, SectionTitle } from "../../components";
import { mcscramWebSocketUrl } from "../../api/mcscramClient";
import type { SamplingStatus } from "./controls/samplingStatus";
import type { SamplingControls } from "./controls/useSamplingControls";
import type { SamplingRunMetricsStore } from "./samplingRunMetrics";
import { createSampleRateTracker } from "./sampleRateTracker";

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
  targetEpsilon?: number;
  epsilonConverged?: boolean;
  oracleP?: number;
};

const MAX_REPLAY_PATH_POINTS = 10_000;
const MAX_RENDER_PATH_POINTS = 3_000;
const MARKER_RENDER_THRESHOLD = 1_400;
const EPSILON_TRACE_OPACITY = 0.48;
const EPSILON_LINE_ALPHA = 0.62;
const EPSILON_MARKER_ALPHA = 0.42;
const EPSILON_MARKER_SIZE = 2;

export function SamplingTrajectoryPanel({
  samplingControls,
  runMetricsStore,
  clientAlpha,
}: {
  samplingControls: SamplingControls | null;
  runMetricsStore: SamplingRunMetricsStore;
  clientAlpha: number;
}) {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [replayPathPoints, setReplayPathPoints] = useState<LiveProbabilityPoint[]>([]);
  const [streamError, setStreamError] = useState<string | null>(null);
  const samplingControlsRef = useRef(samplingControls);
  const runMetricsStoreRef = useRef(runMetricsStore);
  const pointsRef = useRef<LiveProbabilityPoint[]>([]);
  const pendingUpdatesRef = useRef<LiveProgressUpdate[]>([]);
  const sampleRateTrackerRef = useRef(createSampleRateTracker());
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

  const scheduleUpdateFlush = (update: LiveProgressUpdate, timestampSeconds: number) => {
    pendingUpdatesRef.current.push(update);
    const observation = sampleRateTrackerRef.current.observe(update.trials ?? Number.NaN, timestampSeconds);
    latestSampleRateRef.current = observation?.trialsPerSecond ?? null;
    if (frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(flushPendingUpdates);
    }
  };

  const resetRunMetrics = () => {
    latestSampleRateRef.current = null;
    runMetricsStoreRef.current.setMetrics({ trialsPerSecond: null, completedPasses: null });
  };

  const resetReplayPath = () => {
    pendingUpdatesRef.current = [];
    pointsRef.current = [];
    setReplayPathPoints([]);
    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  };

  const clearReplayPath = () => {
    sampleRateTrackerRef.current.clear();
    resetReplayPath();
    resetRunMetrics();
  };

  useEffect(() => {
    const wsUrl = mcscramWebSocketUrl({ replay: true });
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let connectTimer: number | undefined;
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
        if (event.event === "calculation_reset") {
          clearReplayPath();
          setControlError(null);
          return;
        }
        if (event.event === "server_started_calculation") {
          sampleRateTrackerRef.current.start();
          resetReplayPath();
          resetRunMetrics();
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
          scheduleUpdateFlush(event.update, event.timestamp);
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

    connectTimer = window.setTimeout(connect, 0);

    return () => {
      stopped = true;
      if (connectTimer !== undefined) window.clearTimeout(connectTimer);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
      socket?.close();
    };
  }, []);

  const waitingText = connection === "open"
    ? "Waiting for replayed stream points."
    : "Connecting to the sampling stream...";

  return (
    <div className="panel span-3 playground-sampling-outputs">
      <div className="playground-sampling-outputs__section">
        <SectionTitle
          eyebrow="Output: Cumulative Trials since Start"
          title="Sampled probability and epsilon trajectory with cumulative trials"
          text="The primary axis tracks sampled probability intervals; the right axis overlays server epsilon in log-scale with the configured target half-width."
        />
        {streamError ? <p className="live-demo-error playground-sampling-error">{streamError}</p> : null}
        {replayPathPoints.length ? (
          <LiveProbabilityPathPlot points={replayPathPoints} xMode="absolute" clientAlpha={clientAlpha} />
        ) : (
          <EmptyState text={waitingText} />
        )}
      </div>
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
    const epsilonPalette = {
      line: inkPalette.paynes,
      marker: inkPalette.paynes,
      legendgroup: xMode === "absolute" ? "replay-epsilon" : "live-epsilon",
    };
    const renderPoints = thinProbabilityPoints(points, MAX_RENDER_PATH_POINTS);
    const x = trialsXAxis(renderPoints, xMode);
    const renderWithMarkers = renderPoints.length <= MARKER_RENDER_THRESHOLD;
    const traceType = renderWithMarkers ? "scatter" : "scattergl";
    const traceMode = renderWithMarkers ? "lines+markers" : "lines";
    const oracleP = renderPoints.find((point) => isPositiveNumber(point.oracleP))?.oracleP;
    const band99 = liveConfidenceBandTrace({
      name: "sampled p01-p99",
      x,
      lower: renderPoints.map((point) => point.p01),
      upper: renderPoints.map((point) => point.p99),
      color: palette.band,
      opacity: 0.08,
      legendgroup: palette.legendgroup,
    });
    const band90 = liveConfidenceBandTrace({
      name: "sampled p05-p95",
      x,
      lower: renderPoints.map((point) => point.p05),
      upper: renderPoints.map((point) => point.p95),
      color: palette.band,
      opacity: 0.16,
      legendgroup: palette.legendgroup,
    });
    const clientConfidence = confidenceFromAlpha(clientAlpha);
    const clientConfidenceLabel = formatNumber(clientConfidence * 100, 4);
    const clientAlphaZ = normalQuantileTwoSided(clientConfidence);
    const clientAlphaLower = renderPoints.map((point) => {
      const stdErr = Math.sqrt((point.mean * (1 - point.mean)) / point.trials);
      return Math.max(0, point.mean - clientAlphaZ * stdErr);
    });
    const clientAlphaUpper = renderPoints.map((point) => {
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
    const oracleLines = oracleReferenceTraces(renderPoints, x, oracleP, xMode, clientAlpha, inkPalette.ink);
    const yValues = [
      ...renderPoints.flatMap((point) => [point.mean, point.p01, point.p05, point.p95, point.p99]),
      ...(Array.isArray(clientAlphaBand?.y) ? clientAlphaBand.y as Array<number | null | undefined> : []),
      ...oracleLines.flatMap((trace) => Array.isArray(trace.y) ? trace.y as Array<number | null | undefined> : []),
    ];
    const epsilonValues = renderPoints.map((point) => point.epsilon);
    const targetEpsilon = renderPoints.find((point) => isPositiveNumber(point.targetEpsilon))?.targetEpsilon;
    const hasEpsilonAxis = epsilonValues.some(isPositiveNumber) || isPositiveNumber(targetEpsilon);
    const epsilonRange = hasEpsilonAxis ? logAxisRange(epsilonValues, { include: [targetEpsilon], pad: 0.12 }) : undefined;
    const epsilonAxisTicks = epsilonLogAxisTicks(epsilonRange, targetEpsilon);
    const meanLine: PlotTrace = {
      type: traceType,
      mode: traceMode,
      name: "live mean",
      x,
      y: renderPoints.map((point) => point.mean),
      customdata: renderPoints.map((point) => [
        point.iterations ?? null,
        formatInteger(point.trials),
        formatNumber(point.p05, 4),
        formatNumber(point.p95, 4),
        formatNumber(point.epsilon, 4),
      ]),
      line: { color: palette.line, width: 1.8 },
      marker: { color: palette.marker, size: renderWithMarkers ? 4 : 0 },
      legendgroup: palette.legendgroup,
      hovertemplate: `${xMode === "window" ? "additional" : "total"} trials=%{x:.3s}<br>total trials=%{customdata[1]}<br>mean=%{y:.4e}<br>iterations=%{customdata[0]}<br>p05=%{customdata[2]}<br>p95=%{customdata[3]}<br>epsilon=%{customdata[4]}<extra>live run</extra>`,
    };
    const epsilonLine: PlotTrace | null = hasEpsilonAxis
      ? {
        type: traceType,
        mode: traceMode,
        name: "server epsilon",
        x,
        y: epsilonValues,
        yaxis: "y2",
        customdata: renderPoints.map((point) => [
          point.iterations ?? null,
          formatInteger(point.trials),
          formatNumber(point.targetEpsilon, 4),
          point.epsilonConverged ? "yes" : "no",
        ]),
        opacity: EPSILON_TRACE_OPACITY,
        line: { color: withAlpha(epsilonPalette.line, EPSILON_LINE_ALPHA), width: 1.5 },
        marker: { color: withAlpha(epsilonPalette.marker, EPSILON_MARKER_ALPHA), size: renderWithMarkers ? EPSILON_MARKER_SIZE : 0 },
        legendgroup: epsilonPalette.legendgroup,
        hovertemplate: `${xMode === "window" ? "additional" : "total"} trials=%{x:.3s}<br>total trials=%{customdata[1]}<br>epsilon=%{y:.4e}<br>target=%{customdata[2]}<br>converged=%{customdata[3]}<br>iterations=%{customdata[0]}<extra>live run</extra>`,
      }
      : null;
    const targetEpsilonLine: PlotTrace | null = isPositiveNumber(targetEpsilon) && x.length > 1
      ? {
        type: "scatter",
        mode: "lines",
        name: `target ε=${formatNumber(targetEpsilon, 3)}`,
        x: [x[0], x[x.length - 1]],
        y: [targetEpsilon, targetEpsilon],
        yaxis: "y2",
        line: { color: withAlpha(epsilonPalette.line, 0.82), dash: "dash", width: 1.3 },
        legendgroup: epsilonPalette.legendgroup,
        hovertemplate: "target epsilon=%{y:.4e}<extra>target</extra>",
      }
      : null;

    return {
      data: [band99, band90, clientAlphaBand, ...oracleLines, meanLine, targetEpsilonLine, epsilonLine]
        .filter(Boolean) as PlotTrace[],
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
        yaxis2: {
          title: {
            text: "Epsilon (half-width, log scale)",
            standoff: 12,
            font: { color: withAlpha(epsilonPalette.line, 0.88) },
          },
          type: "log",
          tickmode: "array",
          tickvals: epsilonAxisTicks?.tickvals,
          ticktext: epsilonAxisTicks?.ticktext,
          range: epsilonRange,
          overlaying: "y",
          side: "right",
          showgrid: false,
          showline: true,
          linecolor: withAlpha(epsilonPalette.line, 0.75),
          tickcolor: withAlpha(epsilonPalette.line, 0.75),
          ticks: "outside",
          ticklen: 4,
          tickwidth: 1,
          tickfont: { color: withAlpha(epsilonPalette.line, 0.82), size: 11 },
          minor: { ticks: "off", showgrid: false },
          automargin: true,
          visible: hasEpsilonAxis,
        },
        legend: { y: -0.28, orientation: "h", font: { size: 10 } },
        margin: { l: 86, r: hasEpsilonAxis ? 98 : 22, t: 20, b: 112 },
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
    targetEpsilon: update.target_epsilon,
    epsilonConverged: update.epsilon_converged,
    oracleP: update.oracle_p,
  };
}

function trialsXAxis(points: LiveProbabilityPoint[], xMode: "window" | "absolute"): number[] {
  const firstTrials = points[0]?.trials ?? 0;
  return xMode === "window"
    ? points.map((point) => Math.max(0, point.trials - firstTrials))
    : points.map((point) => point.trials);
}

function epsilonLogAxisTicks(
  log10Range: [number, number] | undefined,
  targetEpsilon: number | undefined,
): { tickvals: number[]; ticktext: string[] } | undefined {
  if (!log10Range) return undefined;
  const [lo, hi] = log10Range;
  const tickvals: number[] = [];
  for (let exp = Math.floor(lo); exp <= Math.ceil(hi); exp += 1) {
    tickvals.push(10 ** exp);
  }
  if (isPositiveNumber(targetEpsilon)) {
    const logTarget = Math.log10(targetEpsilon);
    const onDecade = tickvals.some((tick) => Math.abs(Math.log10(tick) - logTarget) < 1e-9);
    if (!onDecade && logTarget >= lo && logTarget <= hi) {
      tickvals.push(targetEpsilon);
      tickvals.sort((a, b) => a - b);
    }
  }
  if (!tickvals.length) return undefined;
  return {
    tickvals,
    ticktext: tickvals.map((value) => value.toExponential(3)),
  };
}

function thinProbabilityPoints(points: LiveProbabilityPoint[], maxPoints: number): LiveProbabilityPoint[] {
  if (!points.length || points.length <= maxPoints) return points;
  if (maxPoints <= 1) return [points[points.length - 1]];
  const step = (points.length - 1) / (maxPoints - 1);
  const sampled: LiveProbabilityPoint[] = [];
  let lastIndex = -1;
  for (let bucket = 0; bucket < maxPoints; bucket += 1) {
    const index = Math.min(points.length - 1, Math.round(bucket * step));
    if (index === lastIndex) continue;
    sampled.push(points[index]);
    lastIndex = index;
  }
  if (lastIndex !== points.length - 1) sampled.push(points[points.length - 1]);
  return sampled;
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
