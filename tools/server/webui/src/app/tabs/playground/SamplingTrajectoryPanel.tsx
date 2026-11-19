import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { DiagramFullscreenButton } from "../../components/ui/DiagramFullscreenButton";
import { darken, usePlotTheme, withAlpha, type InkPalette } from "../../components/plot/plotTokens";
import {
  PlotMathbox,
  type PlotMathboxAxis,
  type PlotMathboxLegendItem,
  type PlotMathboxReference,
  type PlotMathboxSeries,
} from "../../components/plot/PlotMathbox";
import { formatDefaultLinear, formatExp, formatPower, formatSI } from "../../components/plot/mathbox/axis";
import type { AxisScale, TrajectoryStroke } from "../../components/plot/mathbox/trajectoryPlot";
import {
  confidenceFromAlpha,
  gbasProjectedPasses,
  normalQuantileTwoSided,
  type ConvergenceStatsPayload,
} from "../../lib/liveConvergenceStats";
import { formatNumber } from "../../lib/format";
import { SectionTitle } from "../../components";
import { mcscramWebSocketUrl } from "../../api/mcscramClient";
import type { SamplingStatus } from "./controls/samplingStatus";
import type { SamplingControls } from "./controls/useSamplingControls";
import type { SamplingRunMetricsStore } from "./samplingRunMetrics";
import { createSampleRateTracker } from "./sampleRateTracker";
import { createSampleTimingTracker, type SampleTimingObservation } from "./sampleTimingTracker";

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
  // Per-policy convergence criteria (current half-width, target, projected N).
  waldEpsilon?: number;
  waldTargetEpsilon?: number;
  waldProjectedTrials?: number;
  bayesEpsilon?: number;
  bayesTargetEpsilon?: number;
  bayesProjectedTrials?: number;
  gbasEpsilon?: number;
  gbasTargetEpsilon?: number;
  gbasProjectedTrials?: number;
  // GBAS (Huber 2017) exact relative-error stream.
  gbasRunCiLow?: number;
  gbasRunCiHigh?: number;
  gbasRunCi95Low?: number;
  gbasRunCi95High?: number;
  gbasRunCi99Low?: number;
  gbasRunCi99High?: number;
  gbasMean?: number;
  gbasCiLow?: number;
  gbasCiHigh?: number;
  gbasConverged?: boolean;
  gbasTargetDelta?: number;
};

const MAX_REPLAY_PATH_POINTS = 10_000;
const MAX_RENDER_PATH_POINTS = 3_000;

/** Stable formatter refs so axis label exprs are not re-pushed every render. */
const PROBABILITY_Y_FORMAT = formatDefaultLinear;
const EPSILON_Y_FORMAT = formatExp(2);

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
  const sampleTimingTrackerRef = useRef(createSampleTimingTracker());
  const latestSampleRateRef = useRef<number | null>(null);
  const latestTimingRef = useRef<SampleTimingObservation | null>(null);
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
    const projectedPasses = gbasProjectedPasses(latestUpdate);
    runMetricsStoreRef.current.setMetrics({
      trialsPerSecond: latestSampleRateRef.current,
      completedPasses: isFiniteNumber(latestUpdate.iterations) ? latestUpdate.iterations : null,
      gbasProjectedPasses: projectedPasses ?? null,
      elapsedSeconds: latestTimingRef.current?.elapsedSeconds ?? null,
      timeToConvergenceSeconds: latestTimingRef.current?.timeToConvergenceSeconds ?? null,
      epsilonConverged: latestTimingRef.current?.epsilonConverged ?? false,
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
    latestTimingRef.current = sampleTimingTrackerRef.current.observe(
      timestampSeconds,
      update.converged === true,
    );
    if (frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(flushPendingUpdates);
    }
  };

  const resetRunMetrics = () => {
    latestSampleRateRef.current = null;
    latestTimingRef.current = null;
    runMetricsStoreRef.current.resetMetrics();
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
    sampleTimingTrackerRef.current.clear();
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
          sampleTimingTrackerRef.current.start();
          resetReplayPath();
          resetRunMetrics();
          setControlError(null);
          return;
        }
        if (event.event === "calculation_stopped") {
          controls?.notifyStreamStopped();
          controls?.setStatus(event.status);
          return;
        }
        if (event.event === "calculation_completed") {
          controls?.notifyStreamStopped();
          controls?.setStatus((previous) => completedStatus(previous, event.result));
          return;
        }
        if (event.event === "calculation_failed") {
          controls?.notifyStreamStopped();
          controls?.setStatus((previous) => failedStatus(previous, event.error));
          setControlError(event.error);
          return;
        }
        if (event.event === "progress") {
          if (controls?.clientPausedRef.current) return;
          controls?.notifyStreamProgress();
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

  return (
    <div className="panel span-3 playground-sampling-outputs">
      <div className="playground-sampling-outputs__section">
        <SectionTitle
          eyebrow="Output: p̂ vs trials"
          title="Top-event estimate"
          text="Running mean p̂ of the compiled top event vs cumulative trials, with interval bands. Oracle bands when µ is set."
        />
        {streamError ? <p className="live-demo-error playground-sampling-error">{streamError}</p> : null}
        <LiveProbabilityPathPlot points={replayPathPoints} xMode="absolute" clientAlpha={clientAlpha} />
      </div>
      <div className="playground-sampling-outputs__section">
        <SectionTitle
          eyebrow="Output: GBAS interval"
          title="GBAS estimate"
          text="GBAS interval on the top-event estimate (Gamma(k, k−1); independent of p̂). Marked when success-count k is reached."
        />
        <LiveGbasPathPlot points={replayPathPoints} clientAlpha={clientAlpha} />
      </div>
      <div className="playground-sampling-outputs__section">
        <SectionTitle
          eyebrow="Output: ε vs target"
          title="Stopping half-width"
          text="Per policy (GBAS solid, Wald dashed, Bayes dotted): current ε, target ε, projected N."
        />
        <ConvergenceCriteriaPlot points={replayPathPoints} yScale={{ type: "log", base: 10 }} xScale={{ type: "log", base: 10 }} />
      </div>
      {/* <div className="playground-sampling-outputs__section">
        <SectionTitle
          eyebrow="Output: Convergence Criterion"
          title="Epsilon vs. target epsilon"
          text="Streamed half-width ε from the active stopping policy alongside its δ·p̂ target. Convergence is ε ≤ target ε."
        />
        <EpsilonPlot points={replayPathPoints} yScale={{ type: "log", base: 2 }} xScale={{ type: "log", base: 2 }} />
      </div> */}
    </div>
  );
}

/** Per-plot log/linear axis configuration (defaults to linear when omitted). */
type AxisScaleProps = { xScale?: AxisScale; yScale?: AxisScale };

/** Overlay per-plot scale config onto a scene's axes (linear when unset). */
function applyScales<T extends { x: PlotMathboxAxis; y: PlotMathboxAxis }>(
  scene: T,
  { xScale, yScale }: AxisScaleProps,
): T {
  if (!xScale && !yScale) return scene;
  return {
    ...scene,
    x: xScale ? { ...scene.x, scale: xScale } : scene.x,
    y: yScale ? { ...scene.y, scale: yScale } : scene.y,
  };
}

const LiveProbabilityPathPlot = memo(function LiveProbabilityPathPlot({
  points,
  xMode,
  clientAlpha,
  xScale,
  yScale,
}: {
  points: LiveProbabilityPoint[];
  xMode: "window" | "absolute";
  clientAlpha: number;
} & AxisScaleProps) {
  const { palette } = usePlotTheme();
  const scene = useMemo(
    () => buildProbabilityScene(points, xMode, clientAlpha, palette),
    [points, xMode, clientAlpha, palette],
  );
  const view = applyScales(scene, { xScale, yScale });

  return (
    <TrajectoryPlotFrame>
      <PlotMathbox
        className="convergence-large-plot live-probability-plot"
        x={view.x}
        y={view.y}
        series={view.series}
        references={view.references}
        legend={view.legend}
      />
    </TrajectoryPlotFrame>
  );
});

const LiveGbasPathPlot = memo(function LiveGbasPathPlot({
  points,
  clientAlpha,
  xScale,
  yScale,
}: {
  points: LiveProbabilityPoint[];
  clientAlpha: number;
} & AxisScaleProps) {
  const { palette } = usePlotTheme();
  const scene = useMemo(() => buildGbasScene(points, clientAlpha, palette), [points, clientAlpha, palette]);
  const view = applyScales(scene, { xScale, yScale });
  return (
    <TrajectoryPlotFrame>
      <PlotMathbox
        className="convergence-large-plot live-gbas-plot"
        x={view.x}
        y={view.y}
        series={view.series}
        references={view.references}
        legend={view.legend}
      />
    </TrajectoryPlotFrame>
  );
});

const ConvergenceCriteriaPlot = memo(function ConvergenceCriteriaPlot({
  points,
  xScale,
  yScale,
}: { points: LiveProbabilityPoint[] } & AxisScaleProps) {
  const { palette } = usePlotTheme();
  const scene = useMemo(() => buildConvergenceCriteriaScene(points, palette), [points, palette]);
  const view = applyScales(scene, { xScale, yScale });
  return (
    <TrajectoryPlotFrame>
      <PlotMathbox
        className="convergence-large-plot live-criteria-plot"
        x={view.x}
        y={view.y}
        series={view.series}
        references={view.references}
        legend={view.legend}
      />
    </TrajectoryPlotFrame>
  );
});

const EpsilonPlot = memo(function EpsilonPlot({
  points,
  xScale,
  yScale,
}: { points: LiveProbabilityPoint[] } & AxisScaleProps) {
  const { palette } = usePlotTheme();
  const scene = useMemo(
    () =>
      buildEpsilonScene(points, {
        currentColor: palette.ochre,
        targetColor: palette.ink,
        ink: palette.ink,
      }),
    [points, palette],
  );
  const view = applyScales(scene, { xScale, yScale });
  return (
    <TrajectoryPlotFrame>
      <PlotMathbox
        className="convergence-large-plot live-epsilon-plot"
        x={view.x}
        y={view.y}
        series={view.series}
        references={view.references}
        legend={view.legend}
      />
    </TrajectoryPlotFrame>
  );
});

function TrajectoryPlotFrame({ children }: { children: ReactNode }) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  return (
    <div ref={rootRef} className="trajectory-plot-frame">
      {children}
      <div className="trajectory-plot-frame__hud">
        <div className="trajectory-plot-frame__controls" aria-label="Plot view controls">
          <DiagramFullscreenButton targetRef={rootRef} />
        </div>
      </div>
    </div>
  );
}

type ProbabilityScene = {
  x: PlotMathboxAxis;
  y: PlotMathboxAxis;
  series: PlotMathboxSeries[];
  references: PlotMathboxReference[];
  legend: PlotMathboxLegendItem[];
};

function buildProbabilityScene(
  points: LiveProbabilityPoint[],
  xMode: "window" | "absolute",
  clientAlpha: number,
  palette: InkPalette,
): ProbabilityScene {
  const colors = xMode === "absolute"
    ? { line: palette.madder, band: palette.ochre }
    : { line: palette.indigoDeep, band: palette.indigoDeep };

  const renderPoints = thinProbabilityPoints(points, MAX_RENDER_PATH_POINTS);
  const xs = trialsXAxis(renderPoints, xMode);
  const oracleP = renderPoints.find((point) => isPositiveNumber(point.oracleP))?.oracleP;

  const clientConfidence = confidenceFromAlpha(clientAlpha);
  const clientConfidenceLabel = formatNumber(clientConfidence * 100, 4);
  const clientZ = normalQuantileTwoSided(clientConfidence);
  const clientLower = renderPoints.map((point) => {
    const stdErr = Math.sqrt((point.mean * (1 - point.mean)) / point.trials);
    return Math.max(0, point.mean - clientZ * stdErr);
  });
  const clientUpper = renderPoints.map((point) => {
    const stdErr = Math.sqrt((point.mean * (1 - point.mean)) / point.trials);
    return Math.min(1, point.mean + clientZ * stdErr);
  });

  const meanValues = renderPoints.map((point) => point.mean);
  const series: PlotMathboxSeries[] = [];
  series.push({
    id: "band99",
    kind: "band",
    x: xs,
    lower: renderPoints.map((point) => point.p01 ?? Number.NaN),
    upper: renderPoints.map((point) => point.p99 ?? Number.NaN),
    color: colors.band,
    opacity: 0.1,
  });
  series.push({
    id: "band90",
    kind: "band",
    x: xs,
    lower: renderPoints.map((point) => point.p05 ?? Number.NaN),
    upper: renderPoints.map((point) => point.p95 ?? Number.NaN),
    color: colors.band,
    opacity: 0.18,
  });
  series.push({
    id: "clientBand",
    kind: "band",
    x: xs,
    lower: clientLower,
    upper: clientUpper,
    color: palette.sage,
    opacity: 0.16,
  });
  const oracleSeries = oracleLineSeries(renderPoints, xs, oracleP, xMode, clientAlpha, palette.ink);
  series.push(...oracleSeries);
  series.push({
    id: "mean",
    kind: "line",
    x: xs,
    y: meanValues,
    color: colors.line,
    width: 0,
    markerSize: 3,
    markerColor: colors.line,
    markerOpacity: 0.7,
  });

  const yValues: Array<number | null | undefined> = [
    ...renderPoints.flatMap((point) => [point.mean, point.p01, point.p05, point.p95, point.p99]),
    ...clientLower,
    ...clientUpper,
    ...oracleSeries.flatMap((entry) => (entry.kind === "line" ? Array.from(entry.y) : [])),
  ];
  const yRange = linearAxisRange(yValues, { floor: 0, ceiling: 1, pad: 0.1 }) ?? [0, 1];
  const xMax = Math.max(1, ...xs.filter(isFiniteNumber));

  const legend: PlotMathboxLegendItem[] = [
    { label: "sampled mean", color: colors.line, marker: true },
    { label: "p05–p95", color: colors.band, fill: true },
    { label: "p01–p99", color: colors.band, fill: true },
    { label: `client ${clientConfidenceLabel}% CI`, color: palette.sage, fill: true },
  ];
  if (isPositiveNumber(oracleP)) {
    legend.push({ label: "oracle p ± CI", color: palette.ink, stroke: "dashed" });
  }

  return {
    x: {
      range: [0, xMax],
      title: xMode === "window"
        ? "Additional Bernoulli trials since first streamed sample"
        : "Cumulative Bernoulli trials",
      format: formatSI,
      color: palette.ink,
    },
    y: {
      range: yRange,
      title: "Probability estimate",
      format: PROBABILITY_Y_FORMAT,
      tickCount: 5,
      color: palette.ink,
    },
    series,
    references: [],
    legend,
  };
}

/** GBAS (Huber 2017) path plot: running mean + exact relative-error band + oracle. */
function buildGbasScene(
  points: LiveProbabilityPoint[],
  clientAlpha: number,
  palette: InkPalette,
): ProbabilityScene {
  const renderPoints = thinProbabilityPoints(points, MAX_RENDER_PATH_POINTS);
  const xs = renderPoints.map((point) => point.trials);
  const lower = renderPoints.map((point) => point.gbasRunCiLow ?? Number.NaN);
  const upper = renderPoints.map((point) => point.gbasRunCiHigh ?? Number.NaN);
  const ci95Lower = renderPoints.map((point) => point.gbasRunCi95Low ?? Number.NaN);
  const ci95Upper = renderPoints.map((point) => point.gbasRunCi95High ?? Number.NaN);
  const ci99Lower = renderPoints.map((point) => point.gbasRunCi99Low ?? Number.NaN);
  const ci99Upper = renderPoints.map((point) => point.gbasRunCi99High ?? Number.NaN);
  const hasCi95 = ci95Lower.some(isFiniteNumber) && ci95Upper.some(isFiniteNumber);
  const hasCi99 = ci99Lower.some(isFiniteNumber) && ci99Upper.some(isFiniteNumber);

  const oracleP = renderPoints.find((point) => isPositiveNumber(point.oracleP))?.oracleP;
  const meanValues = renderPoints.map((point) => point.mean);
  // Cool, authoritative register — distinct from the warm madder/ochre primary path plot.
  const lineColor = palette.indigoDeep;

  // Series ids are fixed across stream resets so the plot controller can refill
  // GPU buffers via source.set instead of tearing down draw subtrees.
  const oracleSeries = oracleLineSeries(renderPoints, xs, oracleP, "absolute", clientAlpha, palette.ink);
  const series: PlotMathboxSeries[] = [
    { id: "gbasBand99", kind: "band", x: xs, lower: ci99Lower, upper: ci99Upper, color: lineColor, opacity: 0.1 },
    { id: "gbasBand95", kind: "band", x: xs, lower: ci95Lower, upper: ci95Upper, color: lineColor, opacity: 0.16 },
    { id: "gbasBand", kind: "band", x: xs, lower, upper, color: palette.sage, opacity: 0.18 },
    ...oracleSeries,
    {
      id: "gbasMean",
      kind: "line",
      x: xs,
      y: meanValues,
      color: lineColor,
      width: 0,
      markerSize: 3,
      markerColor: lineColor,
      markerOpacity: 0.7,
    },
  ];

  const last = renderPoints[renderPoints.length - 1];
  const converged = last?.gbasConverged === true;
  const showCiLow = converged && isFiniteNumber(last?.gbasCiLow);
  const showCiHigh = converged && isFiniteNumber(last?.gbasCiHigh);
  const showMean = converged && isFiniteNumber(last?.gbasMean);
  const references: PlotMathboxReference[] = [
    {
      id: "gbas-ci-low",
      axis: "y",
      value: showCiLow ? last.gbasCiLow! : 0,
      color: withAlpha(lineColor, 0.55),
      stroke: "dashed",
      opacity: showCiLow ? 0.55 : 0,
    },
    {
      id: "gbas-ci-high",
      axis: "y",
      value: showCiHigh ? last.gbasCiHigh! : 0,
      color: withAlpha(lineColor, 0.55),
      stroke: "dashed",
      opacity: showCiHigh ? 0.55 : 0,
    },
    {
      id: "gbas-mean",
      axis: "y",
      value: showMean ? last.gbasMean! : 0,
      color: withAlpha(lineColor, 0.85),
      stroke: "solid",
      opacity: showMean ? 0.85 : 0,
      label: "p̂ GBAS",
    },
  ];

  const yValues: Array<number | null | undefined> = [
    ...meanValues,
    ...lower,
    ...upper,
    ...(hasCi99 ? [...ci99Lower, ...ci99Upper] : []),
    converged ? last.gbasCiLow : undefined,
    converged ? last.gbasCiHigh : undefined,
    ...oracleSeries.flatMap((entry) => (entry.kind === "line" ? Array.from(entry.y) : [])),
  ];
  const yRange = linearAxisRange(yValues, { floor: 0, ceiling: 1, pad: 0.12 }) ?? [0, 1];
  const xMax = Math.max(1, ...xs.filter(isFiniteNumber));

  const targetDelta = renderPoints.map((point) => point.gbasTargetDelta).find(isFiniteNumber);
  const clientCiLabel = isFiniteNumber(targetDelta)
    ? `GBAS ${formatNumber((1 - targetDelta) * 100, 4)}% CI`
    : "GBAS CI";

  const legend: PlotMathboxLegendItem[] = [
    { label: "sampled mean", color: lineColor, marker: true },
    { label: clientCiLabel, color: palette.sage, fill: true },
  ];
  if (hasCi95) legend.push({ label: "GBAS 95% CI", color: lineColor, fill: true });
  if (hasCi99) legend.push({ label: "GBAS 99% CI", color: lineColor, fill: true });
  if (converged) {
    legend.push({ label: "p̂ ± CI (at k)", color: lineColor, stroke: "dashed" });
  }
  if (isPositiveNumber(oracleP)) {
    legend.push({ label: "oracle p ± CI", color: palette.ink, stroke: "dashed" });
  }

  return {
    x: {
      range: [0, xMax],
      title: "Cumulative Bernoulli trials",
      format: formatSI,
      color: palette.ink,
    },
    y: {
      range: yRange,
      title: "Probability estimate (GBAS)",
      format: PROBABILITY_Y_FORMAT,
      tickCount: 5,
      color: palette.ink,
    },
    series,
    references,
    legend,
  };
}

/** Per-policy convergence-criteria plot: ε (solid) + target ε (dashed), one hue per policy. */
function buildConvergenceCriteriaScene(
  points: LiveProbabilityPoint[],
  palette: InkPalette,
): ProbabilityScene {
  const renderPoints = thinProbabilityPoints(points, MAX_RENDER_PATH_POINTS);
  const xs = renderPoints.map((point) => point.trials);

  // Encoding: policy = complementary hue + fixed stroke (GBAS solid, Wald dashed,
  // Bayes dotted). Quantity is encoded on top of that: current ε at base alpha,
  // target ε at higher alpha, projected-N as a thicker, darker vertical guide.
  const policies: Array<{
    key: string;
    label: string;
    color: string[];
    markerSize?: number[];
    markerColor?: string[];
    width: number[];
    opacity?: number[];
    stroke: TrajectoryStroke[];
    epsilon: (point: LiveProbabilityPoint) => number | undefined;
    target: (point: LiveProbabilityPoint) => number | undefined;
    projected: (point: LiveProbabilityPoint) => number | undefined;
  }> = [
    {
      key: "gbas",
      label: "GBAS",
      opacity: [0.6, 0.8, 1.0],
      markerSize: [1.5, 0, 0],
      markerColor: [palette.inkSoft, palette.ochre, palette.ochreDark],
      color: [palette.inkFaint, palette.inkMute, palette.inkSoft],
      width: [1.0, 2.0, 1.0],
      stroke: ["solid", "dashed", "solid"],
      epsilon: (point) => point.gbasEpsilon,
      target: (point) => point.gbasTargetEpsilon,
      projected: (point) => point.gbasProjectedTrials,
    },
    {
      key: "wald",
      label: "Wald",
      opacity: [1.0, 0.8, 1.0],
      markerSize: [4.5, 0, 0],
      markerColor: [palette.ochre, palette.ochre, palette.ochreDark],
      color: [palette.ochreBright, palette.ochreBright, palette.ochreBright],
      width: [1, 3.0, 4.0],
      stroke: ["solid", "solid", "dashed"],
      epsilon: (point) => point.waldEpsilon,
      target: (point) => point.waldTargetEpsilon,
      projected: (point) => point.waldProjectedTrials,
    },
    {
      key: "bayes",
      label: "Bayes",
      opacity: [0.5, 0.8, 1.0],
      markerSize: [1.5, 0, 0],
      markerColor: [palette.indigoBright, palette.ochre, palette.ochreDark],
      color: [palette.indigoBright, palette.indigoBright, palette.indigoBright],
      width: [1, 2.0, 4.0],
      stroke: ["solid", "dashed", "dotted"],
      epsilon: (point) => point.bayesEpsilon,
      target: (point) => point.bayesTargetEpsilon,
      projected: (point) => point.bayesProjectedTrials,
    },
  ];

  const series: PlotMathboxSeries[] = [];
  const references: PlotMathboxReference[] = [];
  const yValues: Array<number | null | undefined> = [];
  const dataXMax = Math.max(1, ...xs.filter(isFiniteNumber));
  let projectedXMax = 0;

  // Per-quantity styling on top of the per-policy hue + stroke.
  const CURRENT_ALPHA = 1.0;
  const TARGET_ALPHA = 1.0;

  const legend: PlotMathboxLegendItem[] = [];

  for (const policy of policies) {
    const epsilonValues = renderPoints.map(policy.epsilon);
    const targetValues = renderPoints.map(policy.target);
    const hasEpsilon = epsilonValues.some(isFiniteNumber);
    const hasTarget = targetValues.some(isFiniteNumber);
    const projected = lastFinitePositive(renderPoints.map(policy.projected));

    series.push({
      id: `${policy.key}-epsilon`,
      kind: "line",
      x: xs,
      y: epsilonValues.map((value) => value ?? Number.NaN),
      color: policy.color[0],
      markerSize: policy.markerSize?.[0] ?? 0,
      markerColor: policy.markerColor?.[0] ?? policy.color[0],
      width: policy.width[0],
      stroke: policy.stroke[0],
      opacity: policy.opacity?.[0] ?? CURRENT_ALPHA,
    });
    series.push({
      id: `${policy.key}-target`,
      kind: "line",
      x: xs,
      y: targetValues.map((value) => value ?? Number.NaN),
      color: policy.color[1],
      width: policy.width[1],
      stroke: policy.stroke[1],
      opacity: policy.opacity?.[1] ?? TARGET_ALPHA,
    });
    if (hasEpsilon) yValues.push(...epsilonValues);
    if (hasTarget) yValues.push(...targetValues);

    // Projected total trials → a thick, darker vertical "deadline" guide in the
    // policy's hue + stroke. Hidden (opacity 0) until a projection is known.
    references.push({
      id: `${policy.key}-projected`,
      axis: "x",
      value: projected ?? 1,
      color: policy.color[2],
      width: policy.width[2],
      stroke: policy.stroke[2],
      opacity: projected !== undefined ? (policy.opacity?.[2] ?? CURRENT_ALPHA) : 0,
    });
    if (projected !== undefined) projectedXMax = Math.max(projectedXMax, projected);

    legend.push({ label: policy.label, color: policy.color[0], stroke: policy.stroke[0] });
  }

  const xMax = Math.max(dataXMax, projectedXMax, 1) * 1.05;

  const yRange = linearAxisRange(yValues, { floor: 0, pad: 0.12 }) ?? [0, 1];

  return {
    x: {
      range: [0, xMax],
      title: "Cumulative Bernoulli trials",
      format: formatSI,
      color: palette.ink,
    },
    y: {
      range: yRange,
      title: "ε  /  target ε",
      format: formatPower,
      color: palette.ink,
    },
    series,
    references,
    legend,
  };
}

/** Last finite, strictly-positive value in a series (for projected-N guides). */
function lastFinitePositive(values: Array<number | null | undefined>): number | undefined {
  for (let i = values.length - 1; i >= 0; i -= 1) {
    if (isPositiveNumber(values[i])) return values[i] as number;
  }
  return undefined;
}

/** Oracle reference p line plus its analytic Wald CI envelopes (host curves). */
function oracleLineSeries(
  points: LiveProbabilityPoint[],
  xs: number[],
  oracleP: number | null | undefined,
  xMode: "window" | "absolute",
  clientAlpha: number,
  inkColor: string,
): Array<Extract<PlotMathboxSeries, { kind: "line" }>> {
  const line = (
    id: string,
    y: number[],
    stroke: TrajectoryStroke,
    width: number,
  ): Extract<PlotMathboxSeries, { kind: "line" }> => ({
    id,
    kind: "line",
    x: xs,
    y,
    color: inkColor,
    width,
    opacity: 0.7,
    stroke,
  });

  if (!isPositiveNumber(oracleP)) {
    const emptyY = xs.map(() => Number.NaN);
    return [
      line("oracle-99-hi", emptyY, "solid", 1),
      line("oracle-99-lo", emptyY, "solid", 1),
      line("oracle-95-hi", emptyY, "dashed", 1),
      line("oracle-95-lo", emptyY, "dashed", 1),
      line("oracle-cl-hi", emptyY, "dotted", 1.1),
      line("oracle-cl-lo", emptyY, "dotted", 1.1),
      line("oracle-p", emptyY, "dashed", 1.8),
    ];
  }

  const effectiveTrials = (index: number) => (xMode === "window" ? xs[index] : points[index].trials);
  const halfWidth = (trials: number, z: number) => z * Math.sqrt((oracleP * (1 - oracleP)) / trials + 1e-15);
  const upper = (z: number) => points.map((_, index) => Math.min(1, oracleP + halfWidth(effectiveTrials(index), z)));
  const lower = (z: number) => points.map((_, index) => Math.max(0, oracleP - halfWidth(effectiveTrials(index), z)));
  const oracleZ = normalQuantileTwoSided(confidenceFromAlpha(clientAlpha));

  return [
    line("oracle-99-hi", upper(2.576), "solid", 1),
    line("oracle-99-lo", lower(2.576), "solid", 1),
    line("oracle-95-hi", upper(1.96), "dashed", 1),
    line("oracle-95-lo", lower(1.96), "dashed", 1),
    line("oracle-cl-hi", upper(oracleZ), "dotted", 1.1),
    line("oracle-cl-lo", lower(oracleZ), "dotted", 1.1),
    line("oracle-p", points.map(() => oracleP), "dashed", 1.8),
  ];
}

function buildEpsilonScene(
  points: LiveProbabilityPoint[],
  options: { currentColor: string; targetColor: string; ink: string },
): ProbabilityScene {
  const renderPoints = thinProbabilityPoints(points, MAX_RENDER_PATH_POINTS);
  const xs = renderPoints.map((point) => point.trials);
  const currentValues = renderPoints.map((point) => point.epsilon);
  const targetValues = renderPoints.map((point) => point.targetEpsilon);

  const xMax = Math.max(1, ...xs.filter(isFiniteNumber));
  const yRange = linearAxisRange([...currentValues, ...targetValues], { floor: 0, pad: 0.12 }) ?? [0, 1];

  const series: PlotMathboxSeries[] = [
    {
      id: "epsilon-current",
      kind: "line",
      x: xs,
      y: currentValues.map((value) => value ?? Number.NaN),
      color: options.currentColor,
      width: 1.8,
      stroke: "solid",
      markerSize: 4,
      markerColor: options.currentColor,
    },
    {
      id: "epsilon-target",
      kind: "line",
      x: xs,
      y: targetValues.map((value) => value ?? Number.NaN),
      color: options.targetColor,
      width: 1.6,
      stroke: "dashed",
      markerSize: 4,
      markerColor: options.targetColor,
    },
  ];

  const legend: PlotMathboxLegendItem[] = [
    { label: "ε current", color: options.currentColor, stroke: "solid", marker: true },
    { label: "ε target", color: options.targetColor, stroke: "dashed", marker: true },
  ];

  return {
    x: {
      range: [0, xMax],
      title: "Cumulative Bernoulli trials",
      format: formatSI,
      color: options.ink,
    },
    y: {
      range: yRange,
      title: "ε / target ε",
      format: EPSILON_Y_FORMAT,
      color: options.currentColor,
    },
    series,
    references: [],
    legend,
  };
}

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
    waldEpsilon: update.wald_epsilon,
    waldTargetEpsilon: update.wald_target_epsilon,
    waldProjectedTrials: update.wald_projected_trials,
    bayesEpsilon: update.bayes_epsilon,
    bayesTargetEpsilon: update.bayes_target_epsilon,
    bayesProjectedTrials: update.bayes_projected_trials,
    gbasEpsilon: update.gbas_epsilon,
    gbasTargetEpsilon: update.gbas_target_epsilon,
    gbasProjectedTrials: update.gbas_projected_trials,
    gbasRunCiLow: update.gbas_run_ci_low,
    gbasRunCiHigh: update.gbas_run_ci_high,
    gbasRunCi95Low: update.gbas_run_ci95_low,
    gbasRunCi95High: update.gbas_run_ci95_high,
    gbasRunCi99Low: update.gbas_run_ci99_low,
    gbasRunCi99High: update.gbas_run_ci99_high,
    gbasMean: update.gbas_mean,
    gbasCiLow: update.gbas_ci_low,
    gbasCiHigh: update.gbas_ci_high,
    gbasConverged: update.gbas_converged,
    gbasTargetDelta: update.gbas_target_delta,
  };
}

function trialsXAxis(points: LiveProbabilityPoint[], xMode: "window" | "absolute"): number[] {
  const firstTrials = points[0]?.trials ?? 0;
  return xMode === "window"
    ? points.map((point) => Math.max(0, point.trials - firstTrials))
    : points.map((point) => point.trials);
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
