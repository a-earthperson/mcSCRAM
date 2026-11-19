import { Fragment, useEffect, useState, type Dispatch, type SetStateAction } from "react";
import type { DagPayload, FaultTreePayload } from "../../components/canvas";
import { StickyPanel, StickyPanelSlot, StickyPanelSpan } from "../../components/ui/StickyPanel";
import {
  fetchSampleShapeForModel,
  type CompileDagOptions,
  type SampleShape,
} from "../../api/faultTreeApi";
import { classNames } from "../../lib/format";
import { SectionTitle } from "../../components";
import { CompilerControlGroup } from "./controls/CompilerControlGroup";
import { InputControlFields } from "./controls/InputControlGroup";
import { SamplerControlGroup } from "./controls/SamplerControlGroup";
import { SamplerTransportControlGroup } from "./controls/SamplerTransportControlGroup";
import { useInputSettings } from "./controls/useInputSettings";
import { useSamplerSettings } from "./controls/useSamplerSettings";
import { useSamplingControls, type SamplingControls } from "./controls/useSamplingControls";
import { useSamplingRunMetrics, type SamplingRunMetricsStore } from "./samplingRunMetrics";

export type ControlPanelProps = {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  modelIds: string[];
  oracleP: number | null;
  blocked: boolean;
  compileOptions: CompileDagOptions;
  onCompileOptionsChange: Dispatch<SetStateAction<CompileDagOptions>>;
  catalogLoading: boolean;
  catalogError: string | null;
  modelCount: number;
  faultTree: FaultTreePayload | null;
  sourceDag: DagPayload | null;
  compiledDag: DagPayload | null;
  kernelDag: DagPayload | null;
  samplingRunMetricsStore: SamplingRunMetricsStore;
  onSamplingControlsChange?: (controls: SamplingControls | null) => void;
  onSamplerConfidenceChange?: (confidence: number) => void;
};

/**
 * Grid spans for the pipeline-stage control groups (Input · Compiler · Sampler · Run).
 *
 * Horizontal dock (<1900px, 12 cols): title+input 2 · compiler 2 · sampler 5 · run 3 = 12, meta below.
 * Vertical gutter (≥1900px, 5 cols): every group spans the full width and
 *   stacks into a single ledger column.
 */
const CONTROL_GROUP_SPANS = {
  container: { hSpan: { row: 1, col: 12 }, vSpan: { row: 6, col: 5 } },
  input: { hSpan: { row: 1, col: 2 }, vSpan: { row: 1, col: 5 } },
  compiler: { hSpan: { row: 1, col: 3 }, vSpan: { row: 1, col: 5 } },
  sampler: { hSpan: { row: 1, col: 4 }, vSpan: { row: 1, col: 5 } },
  transport: { hSpan: { row: 1, col: 3 }, vSpan: { row: 1, col: 5 } },
} as const;

/**
 * Playground sticky controls — a flat, ruled "control plate" that mirrors the
 * Models archive surface. Layout uses StickyPanel slot spans:
 * - Vertical (≥1900px): 5-column grid, `vSpan` on each slot (sections stack).
 * - Horizontal (<1900px): 12-column grid, `hSpan` on each slot (stage columns).
 */
export function ControlPanel({
  selectedId,
  onSelectedIdChange,
  modelIds,
  oracleP,
  blocked,
  compileOptions,
  onCompileOptionsChange,
  catalogLoading,
  catalogError,
  modelCount,
  faultTree,
  sourceDag,
  compiledDag,
  kernelDag,
  samplingRunMetricsStore,
  onSamplingControlsChange,
  onSamplerConfidenceChange,
}: ControlPanelProps) {
  const inputSettings = useInputSettings({ selectedId, oraclePDefault: oracleP });
  const samplerSettings = useSamplerSettings();
  const [sampleShape, setSampleShape] = useState<SampleShape | null>(null);
  const [sampleShapeLoading, setSampleShapeLoading] = useState(false);
  const [sampleShapeError, setSampleShapeError] = useState<string | null>(null);
  const samplingControls = useSamplingControls({
    getRunOptions: () => ({
      ...(selectedId ? { model_id: selectedId } : {}),
      compilation_level: compileOptions.compilationLevel,
      expand_atleast_gates: compileOptions.expandAtleastGates,
      expand_xor_gates: compileOptions.expandXorGates,
      recognize_atleast_groups: compileOptions.recognizeAtleastGroups,
      recognize_xor_groups: compileOptions.recognizeXorGroups,
      ccf: compileOptions.ccf,
      keep_null_gates: compileOptions.keepNullGates,
      num_trials: samplerSettings.trialsCount,
      early_stop: samplerSettings.stopOnConvergence,
      requested_trials_per_pass: samplerSettings.requestedTrialsPerPass,
      confidence: samplerSettings.confidence,
      delta: samplerSettings.delta,
      policy: "gbas",
      ...(inputSettings.oracleEnabled ? { oracle: inputSettings.oracleP } : {}),
    }),
  });
  const inputsLocked = samplingControls.samplingSessionActive || samplingControls.controlsLocked;
  const samplingRunMetrics = useSamplingRunMetrics(samplingRunMetricsStore);

  useEffect(() => {
    if (!selectedId) {
      setSampleShape(null);
      setSampleShapeError(null);
      setSampleShapeLoading(false);
      return undefined;
    }

    let cancelled = false;
    setSampleShape(null);
    setSampleShapeError(null);
    setSampleShapeLoading(true);

    fetchSampleShapeForModel(selectedId, compileOptions, {
      numTrials: samplerSettings.trialsCount,
      requestedTrialsPerPass: samplerSettings.requestedTrialsPerPass,
    })
      .then((payload) => {
        if (cancelled) return;
        setSampleShape(payload.sample_shape);
        setSampleShapeError(payload.error ?? null);
      })
      .catch((reason) => {
        if (cancelled) return;
        setSampleShape(null);
        setSampleShapeError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!cancelled) setSampleShapeLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [compileOptions, samplerSettings.requestedTrialsPerPass, samplerSettings.trialsCount, selectedId]);

  useEffect(() => {
    onSamplingControlsChange?.(samplingControls);
  }, [
    onSamplingControlsChange,
    samplingControls.status,
    samplingControls.error,
    samplingControls.resetPending,
    samplingControls.samplingControlPending,
    samplingControls.streamPhase,
    samplingControls.clientPaused,
    samplingControls.samplingActive,
    samplingControls.samplingSessionActive,
    samplingControls.controlsLocked,
    samplingControls.notifyStreamProgress,
    samplingControls.notifyStreamStopped,
    samplingControls.resetCalculation,
    samplingControls.toggleSampling,
    samplingControls.setPaused,
  ]);
  useEffect(() => () => onSamplingControlsChange?.(null), [onSamplingControlsChange]);
  useEffect(() => {
    onSamplerConfidenceChange?.(samplerSettings.confidence);
  }, [onSamplerConfidenceChange, samplerSettings.confidence]);

  return (
    <StickyPanel className="playground-controls" aria-label="Compile and sample">
      <StickyPanelSlot area="controls-grid-inner" hSpan={CONTROL_GROUP_SPANS.container.hSpan} vSpan={CONTROL_GROUP_SPANS.container.vSpan}>
        
        <StickyPanelSlot area="title" className="pg-title" vSpan={CONTROL_GROUP_SPANS.input.vSpan} hSpan={CONTROL_GROUP_SPANS.input.hSpan}>
          <div className="md-hidden">
            <SectionTitle eyebrow="Playground" title="Compile and sample" />
          </div>
          <div className="sticky-panel-group sticky-panel-group--input pg-title-input">
            <fieldset className="sticky-panel-group__fieldset">
              <legend className="sticky-panel-group__legend">Input</legend>
              <div className="sticky-panel-group__body">
                <InputControlFields
                  selectedId={selectedId}
                  onSelectedIdChange={onSelectedIdChange}
                  modelIds={modelIds}
                  blocked={blocked}
                  disabled={inputsLocked}
                  inputSettings={inputSettings}
                />
              </div>
            </fieldset>
          </div>
        </StickyPanelSlot>
        <CompilerControlGroup
          compileOptions={compileOptions}
          onCompileOptionsChange={onCompileOptionsChange}
          disabled={inputsLocked}
          {...CONTROL_GROUP_SPANS.compiler}
        />
        <SamplerControlGroup
          settings={samplerSettings}
          disabled={inputsLocked}
          {...CONTROL_GROUP_SPANS.sampler}
        />
        <SamplerTransportControlGroup
          controls={samplingControls}
          stopOnConvergence={samplerSettings.stopOnConvergence}
          onStopOnConvergenceChange={samplerSettings.setStopOnConvergence}
          controlsDisabled={inputsLocked}
          sampleShape={sampleShape}
          sampleShapeLoading={sampleShapeLoading}
          sampleShapeError={sampleShapeError}
          samplingRunMetrics={samplingRunMetrics}
          {...CONTROL_GROUP_SPANS.transport}
        />
      </StickyPanelSlot>
    </StickyPanel>
  );
}

function PlaygroundModelMeta({
  catalogLoading,
  catalogError,
  modelCount,
  faultTree,
  sourceDag,
  compiledDag,
  kernelDag,
}: {
  catalogLoading: boolean;
  catalogError: string | null;
  modelCount: number;
  faultTree: FaultTreePayload | null;
  sourceDag: DagPayload | null;
  compiledDag: DagPayload | null;
  kernelDag: DagPayload | null;
}) {
  if (catalogLoading) {
    return <p className="playground-model-meta">Loading catalog…</p>;
  }
  if (catalogError) {
    return <p className="playground-model-meta">{catalogError}</p>;
  }

  const stages: Array<{ kind: "fault-tree" | "source" | "compiled" | "kernel"; label: string }> = [];
  if (faultTree) {
    stages.push({ kind: "fault-tree", label: `FT ${compactFaultTreeCounts(faultTree)}` });
  }

  const dagGateCounts = [
    sourceDag?.counts ? { kind: "source" as const, value: sourceDag.counts.gate ?? 0 } : null,
    compiledDag?.counts ? { kind: "compiled" as const, value: compiledDag.counts.gate ?? 0 } : null,
    kernelDag?.counts ? { kind: "kernel" as const, value: kernelDag.counts.gate ?? 0 } : null,
  ].filter((entry): entry is { kind: "source" | "compiled" | "kernel"; value: number } => entry !== null);
  if (dagGateCounts.length) {
    stages.push({
      kind: dagGateCounts[dagGateCounts.length - 1].kind,
      label: `DAG gates ${dagGateCounts.map((entry) => formatMetaInteger(entry.value)).join("→")}`,
    });
  }
  const basicEvents = kernelDag?.counts?.basic_event ?? compiledDag?.counts?.basic_event ?? sourceDag?.counts?.basic_event;
  if (basicEvents !== undefined) {
    stages.push({ kind: "source", label: `BE ${formatMetaInteger(basicEvents)}` });
  }

  if (!stages.length) {
    return <p className="playground-model-meta">{modelCount} XML inputs</p>;
  }

  return (
    <p className="playground-model-meta">
      {stages.map((stage, index) => (
        <Fragment key={stage.kind}>
          {index > 0 ? <span className="playground-model-meta__sep" aria-hidden="true"> · </span> : null}
          <span className={classNames("playground-model-meta__stage", `playground-model-meta__stage--${stage.kind}`)}>
            {stage.label}
          </span>
        </Fragment>
      ))}
    </p>
  );
}

function compactFaultTreeCounts(tree: FaultTreePayload): string {
  const gates = tree.nodes.filter((node) => node.kind === "gate").length;
  const basicEvents = tree.nodes.filter((node) => node.kind === "basic_event").length;
  return `${formatMetaInteger(gates)}g/${formatMetaInteger(basicEvents)}be/${formatMetaInteger(tree.edges.length)}e`;
}

function formatMetaInteger(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}
