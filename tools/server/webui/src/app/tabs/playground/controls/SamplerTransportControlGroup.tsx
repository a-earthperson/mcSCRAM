import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/ui/StickyPanel";
import type { SampleShape } from "../../../api/faultTreeApi";
import { formatInteger, formatNumber, formatStreamDurationSeconds } from "../../../lib/format";
import { Metric } from "../../../components";
import type { SamplingRunMetrics } from "../samplingRunMetrics";
import { SamplerTransportControls } from "./SamplerTransportControls";
import { StopOnConvergenceField } from "./StopOnConvergenceField";
import type { SamplingControls } from "./useSamplingControls";

export function SamplerTransportControlGroup({
  controls,
  stopOnConvergence,
  onStopOnConvergenceChange,
  controlsDisabled = false,
  sampleShape,
  sampleShapeLoading = false,
  sampleShapeError = null,
  samplingRunMetrics,
  hSpan,
  vSpan,
}: {
  controls: SamplingControls;
  stopOnConvergence: boolean;
  onStopOnConvergenceChange: (enabled: boolean) => void;
  controlsDisabled?: boolean;
  sampleShape: SampleShape | null;
  sampleShapeLoading?: boolean;
  sampleShapeError?: string | null;
  samplingRunMetrics: SamplingRunMetrics;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Run" className="sticky-panel-group--run" hSpan={hSpan} vSpan={vSpan}>
      <div className="sampler-transport-controls">
        <div className="sampler-transport-controls__primary-row">
          <StopOnConvergenceField
            checked={stopOnConvergence}
            onChange={onStopOnConvergenceChange}
            disabled={controlsDisabled}
          />
          <SamplerTransportControls
            resetPending={controls.resetPending}
            samplingControlPending={controls.samplingControlPending}
            samplingActive={controls.samplingActive}
            streamPhase={controls.streamPhase}
            controlsLocked={controls.controlsLocked}
            clientPaused={controls.clientPaused}
            resetCalculation={controls.resetCalculation}
            toggleSampling={controls.toggleSampling}
            setPaused={controls.setPaused}
          />
        </div>
      </div>
      <SampleShapeMetrics
        sampleShape={sampleShape}
        loading={sampleShapeLoading}
        error={sampleShapeError}
        stopOnConvergence={stopOnConvergence}
        samplingRunMetrics={samplingRunMetrics}
      />
      {controls.error ? <p className="sampler-transport-controls__error">{controls.error}</p> : null}
    </StickyPanelGroup>
  );
}

function SampleShapeMetrics({
  sampleShape,
  loading,
  error,
  stopOnConvergence,
  samplingRunMetrics,
}: {
  sampleShape: SampleShape | null;
  loading: boolean;
  error: string | null;
  stopOnConvergence: boolean;
  samplingRunMetrics: SamplingRunMetrics;
}) {
  const detail = error ?? (loading ? "Resolving native sample shape…" : undefined);
  const trialsPerPass = metricValue(sampleShape?.trials_per_pass, loading);
  const totalPassBudget = stopOnConvergence
    ? (samplingRunMetrics.gbasProjectedPasses ?? sampleShape?.total_iterations)
    : sampleShape?.total_iterations;
  const totalIterations = metricValue(totalPassBudget, loading && !stopOnConvergence);
  const trialsPerSecond = formatSampleRate(samplingRunMetrics.trialsPerSecond);
  const trialsPerSecondDetail = passesPerSecondDetail(
    samplingRunMetrics.trialsPerSecond,
    sampleShape?.trials_per_pass,
  );
  const completedPasses = metricValue(samplingRunMetrics.completedPasses, false);
  const completedPassDetail = completedPassPercentage(samplingRunMetrics.completedPasses, totalPassBudget);
  const elapsedTime = formatStreamDurationSeconds(samplingRunMetrics.elapsedSeconds);
  const convergenceTime = formatConvergenceTime(
    samplingRunMetrics,
    stopOnConvergence,
    totalPassBudget,
    sampleShape?.trials_per_pass,
  );

  return (
    <>
      <div className="sampler-transport-controls__timing-metrics" aria-busy={loading}>
        <Metric label="Elapsed Time" value={elapsedTime} />
        <Metric label="Convergence Time" value={convergenceTime.value} detail={convergenceTime.detail} />
      </div>
      <div className="sampler-transport-controls__metrics" aria-busy={loading}>
        <Metric label="TRIALS PER PASS" value={trialsPerPass} detail={detail} />
        <Metric label="TOTAL PASSES" value={totalIterations} />
        <Metric label="TRIALS PER SECOND" value={trialsPerSecond} detail={trialsPerSecondDetail} />
        <Metric label="COMPLETED" value={completedPasses} detail={completedPassDetail} />
      </div>
    </>
  );
}

function metricValue(value: number | null | undefined, loading: boolean): string {
  if (loading) return "…";
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return formatInteger(value);
}

function formatSampleRate(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${new Intl.NumberFormat("en-US", {
    notation: "compact",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)}/s`;
}

function passesPerSecondDetail(
  trialsPerSecond: number | null | undefined,
  trialsPerPass: number | null | undefined,
): string | undefined {
  if (
    trialsPerSecond === null ||
    trialsPerSecond === undefined ||
    !Number.isFinite(trialsPerSecond) ||
    trialsPerPass === null ||
    trialsPerPass === undefined ||
    !Number.isFinite(trialsPerPass) ||
    trialsPerPass <= 0
  ) {
    return undefined;
  }
  const passesPerSecond = trialsPerSecond / trialsPerPass;
  if (!Number.isFinite(passesPerSecond) || passesPerSecond < 0) {
    return undefined;
  }
  return `${new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(passesPerSecond)} pass/sec`;
}

function formatConvergenceTime(
  metrics: SamplingRunMetrics,
  stopOnConvergence: boolean,
  totalPassBudget: number | undefined,
  trialsPerPass: number | undefined,
): { value: string; detail?: string } {
  const actual = metrics.timeToConvergenceSeconds;
  if (actual !== null && actual !== undefined && Number.isFinite(actual)) {
    return { value: formatStreamDurationSeconds(actual) };
  }
  if (!stopOnConvergence) {
    return { value: "—" };
  }
  const remaining = estimateRemainingConvergenceSeconds(
    metrics.completedPasses,
    metrics.trialsPerSecond,
    totalPassBudget,
    trialsPerPass,
  );
  if (remaining === null) {
    return { value: "—" };
  }
  return {
    value: `~${formatStreamDurationSeconds(remaining)}`,
    detail: "est. remaining",
  };
}

/** Remaining wall time from pass budget, completed passes, and the live trial rate. */
function estimateRemainingConvergenceSeconds(
  completedPasses: number | null,
  trialsPerSecond: number | null,
  totalPassBudget: number | undefined,
  trialsPerPass: number | undefined,
): number | null {
  if (
    completedPasses === null ||
    !Number.isFinite(completedPasses) ||
    trialsPerSecond === null ||
    !Number.isFinite(trialsPerSecond) ||
    trialsPerSecond <= 0 ||
    totalPassBudget === undefined ||
    !Number.isFinite(totalPassBudget) ||
    totalPassBudget <= 0 ||
    trialsPerPass === undefined ||
    !Number.isFinite(trialsPerPass) ||
    trialsPerPass <= 0
  ) {
    return null;
  }
  const remainingPasses = totalPassBudget - completedPasses;
  if (remainingPasses <= 0) return 0;
  return (remainingPasses * trialsPerPass) / trialsPerSecond;
}

function completedPassPercentage(completed: number | null | undefined, total: number | null | undefined): string | undefined {
  if (
    completed === null ||
    completed === undefined ||
    !Number.isFinite(completed) ||
    total === null ||
    total === undefined ||
    !Number.isFinite(total) ||
    total <= 0
  ) {
    return undefined;
  }
  return `${formatNumber((completed / total) * 100, 2)}% complete`;
}
