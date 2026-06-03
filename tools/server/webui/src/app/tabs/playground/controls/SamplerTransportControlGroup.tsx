import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/StickyPanel";
import type { SampleShape } from "../../../faultTreeApi";
import { formatInteger } from "../../../../lib/format";
import { Metric } from "../../../ui";
import { SamplerTransportControls } from "./SamplerTransportControls";
import type { SamplingControls } from "./useSamplingControls";

export function SamplerTransportControlGroup({
  controls,
  sampleShape,
  sampleShapeLoading = false,
  sampleShapeError = null,
  hSpan,
  vSpan,
}: {
  controls: SamplingControls;
  sampleShape: SampleShape | null;
  sampleShapeLoading?: boolean;
  sampleShapeError?: string | null;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Run" className="sticky-panel-group--run" hSpan={hSpan} vSpan={vSpan}>
      <SamplerTransportControls
        resetPending={controls.resetPending}
        samplingControlPending={controls.samplingControlPending}
        samplingActive={controls.samplingActive}
        controlsLocked={controls.controlsLocked}
        clientPaused={controls.clientPaused}
        resetCalculation={controls.resetCalculation}
        toggleSampling={controls.toggleSampling}
        setPaused={controls.setPaused}
      />
      <SampleShapeMetrics
        sampleShape={sampleShape}
        loading={sampleShapeLoading}
        error={sampleShapeError}
      />
      {controls.error ? <p className="sampler-transport-controls__error">{controls.error}</p> : null}
    </StickyPanelGroup>
  );
}

function SampleShapeMetrics({
  sampleShape,
  loading,
  error,
}: {
  sampleShape: SampleShape | null;
  loading: boolean;
  error: string | null;
}) {
  const detail = error ?? (loading ? "Resolving native sample shape…" : undefined);
  const trialsPerPass = metricValue(sampleShape?.trials_per_pass, loading);
  const totalIterations = metricValue(sampleShape?.total_iterations, loading);

  return (
    <div className="sampler-transport-controls__metrics" aria-busy={loading}>
      <Metric label="TRIALS_PER_PASS" value={trialsPerPass} detail={detail} />
      <Metric label="TOTAL_ITERATIONS" value={totalIterations} />
    </div>
  );
}

function metricValue(value: number | undefined, loading: boolean): string {
  if (loading) return "…";
  if (value === undefined) return "—";
  return formatInteger(value);
}
