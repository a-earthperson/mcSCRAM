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
  samplingRunMetrics,
}: {
  sampleShape: SampleShape | null;
  loading: boolean;
  error: string | null;
  samplingRunMetrics: SamplingRunMetrics;
}) {
  const detail = error ?? (loading ? "Resolving native sample shape…" : undefined);
  const trialsPerPass = metricValue(sampleShape?.trials_per_pass, loading);
  const totalIterations = metricValue(sampleShape?.total_iterations, loading);
  const trialsPerSecond = formatSampleRate(samplingRunMetrics.trialsPerSecond);
  const trialsPerSecondDetail = passesPerSecondDetail(
    samplingRunMetrics.trialsPerSecond,
    sampleShape?.trials_per_pass,
  );
  const completedPasses = metricValue(samplingRunMetrics.completedPasses, false);
  const completedPassDetail = completedPassPercentage(samplingRunMetrics.completedPasses, sampleShape?.total_iterations);
  const elapsedTime = formatStreamDurationSeconds(samplingRunMetrics.elapsedSeconds);
  const timeToConvergence = formatStreamDurationSeconds(samplingRunMetrics.timeToConvergenceSeconds);

  return (
    <>
      <div className="sampler-transport-controls__timing-metrics" aria-busy={loading}>
        <Metric label="Elapsed Time" value={elapsedTime} />
        <Metric label="Convergence Time" value={timeToConvergence} />
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
