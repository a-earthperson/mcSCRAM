import { Fragment, useEffect, useState, type Dispatch, type SetStateAction } from "react";
import type { DagPayload, FaultTreePayload } from "../../components/canvas";
import { StickyPanel, StickyPanelSlot } from "../../components/StickyPanel";
import {
  fetchSampleShapeForModel,
  type CompileDagOptions,
  type SampleShape,
} from "../../faultTreeApi";
import { classNames } from "../../../lib/format";
import { SectionTitle } from "../../ui";
import { CompilerControlGroup } from "./controls/CompilerControlGroup";
import { InputControlGroup } from "./controls/InputControlGroup";
import { SamplerControlGroup } from "./controls/SamplerControlGroup";
import { SamplerTransportControlGroup } from "./controls/SamplerTransportControlGroup";
import { useInputSettings } from "./controls/useInputSettings";
import { useSamplerSettings } from "./controls/useSamplerSettings";
import { useSamplingControls } from "./controls/useSamplingControls";
import { formatCounts, formatFaultTreeCounts } from "./format";

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
};

/** Grid spans for scoped control groups (Input · Compiler · Sampler · Run). */
const CONTROL_GROUP_SPANS = {
  input: { hSpan: { row: 1, col: 2 }, vSpan: { row: 1, col: 5 } },
  compiler: { hSpan: { row: 1, col: 3 }, vSpan: { row: 2, col: 5 } },
  sampler: { hSpan: { row: 1, col: 2 }, vSpan: { row: 2, col: 5 } },
  transport: { hSpan: { row: 1, col: 3 }, vSpan: { row: 1, col: 5 } },
} as const;

/**
 * Playground sticky controls. Layout uses StickyPanel slot spans:
 * - Vertical (≥1900px): 5-column grid, `vSpan` on each slot.
 * - Horizontal (<1900px): 12-column grid, `hSpan` on each slot.
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
}: ControlPanelProps) {
  const inputSettings = useInputSettings({ selectedId, oraclePDefault: oracleP });
  const samplerSettings = useSamplerSettings();
  const [sampleShape, setSampleShape] = useState<SampleShape | null>(null);
  const [sampleShapeLoading, setSampleShapeLoading] = useState(false);
  const [sampleShapeError, setSampleShapeError] = useState<string | null>(null);
  const samplingControls = useSamplingControls({
    getRunOptions: () => ({
      num_trials: samplerSettings.trialsCount,
      early_stop: samplerSettings.stopOnConvergence,
      overhead_ratio: samplerSettings.overheadRatio,
      ...(samplerSettings.stopOnConvergence
        ? { confidence: samplerSettings.confidence, delta: samplerSettings.delta }
        : {}),
      ...(inputSettings.oracleEnabled ? { oracle: inputSettings.oracleP } : {}),
    }),
  });

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
      overheadRatio: samplerSettings.overheadRatio,
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
  }, [compileOptions, samplerSettings.overheadRatio, samplerSettings.trialsCount, selectedId]);

  return (
    <StickyPanel className="playground-controls" aria-label="DAG playground controls">
      <StickyPanelSlot area="controls-grid-inner" hSpan={{ row: 2, col: 12 }} vSpan={{ row: 8, col: 5 }}>
        <StickyPanelSlot area="title" vSpan={{ row: 1, col: 5 }} hSpan={{ row: 1, col: 2 }}>
          <SectionTitle eyebrow="Playground" title="End-to-End Pipeline" />
        </StickyPanelSlot>
        <InputControlGroup
          selectedId={selectedId}
          onSelectedIdChange={onSelectedIdChange}
          modelIds={modelIds}
          blocked={blocked}
          inputSettings={inputSettings}
          {...CONTROL_GROUP_SPANS.input}
        />
        <CompilerControlGroup
          compileOptions={compileOptions}
          onCompileOptionsChange={onCompileOptionsChange}
          {...CONTROL_GROUP_SPANS.compiler}
        />
        <SamplerControlGroup
          settings={samplerSettings}
          disabled={samplingControls.samplingActive || samplingControls.controlsLocked}
          {...CONTROL_GROUP_SPANS.sampler}
        />
        <SamplerTransportControlGroup
          controls={samplingControls}
          sampleShape={sampleShape}
          sampleShapeLoading={sampleShapeLoading}
          sampleShapeError={sampleShapeError}
          {...CONTROL_GROUP_SPANS.transport}
        />
        <StickyPanelSlot vSpan={{ row: 1, col: 5 }} hSpan={{ row: 1, col: 12 }}>
          <PlaygroundModelMeta
            catalogLoading={catalogLoading}
            catalogError={catalogError}
            modelCount={modelCount}
            faultTree={faultTree}
            sourceDag={sourceDag}
            compiledDag={compiledDag}
            kernelDag={kernelDag}
          />
        </StickyPanelSlot>
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

  type MetaStage =
    | { kind: "fault-tree"; label: string }
    | { kind: "source" | "compiled" | "kernel"; counts: NonNullable<DagPayload["counts"]> };
  const stages: MetaStage[] = [];
  if (faultTree) stages.push({ kind: "fault-tree", label: formatFaultTreeCounts(faultTree) });
  if (sourceDag?.counts) stages.push({ kind: "source", counts: sourceDag.counts });
  if (compiledDag?.counts) stages.push({ kind: "compiled", counts: compiledDag.counts });
  if (kernelDag?.counts) stages.push({ kind: "kernel", counts: kernelDag.counts });

  if (!stages.length) {
    return <p className="playground-model-meta">{modelCount} XML inputs</p>;
  }

  return (
    <p className="playground-model-meta">
      {stages.map((stage, index) => (
        <Fragment key={stage.kind}>
          {index > 0 ? <span className="playground-model-meta__sep" aria-hidden="true"> · </span> : null}
          <span className={classNames("playground-model-meta__stage", `playground-model-meta__stage--${stage.kind}`)}>
            {stage.kind === "fault-tree" ? `fault tree ${stage.label}` : `${stage.kind} ${formatCounts(stage.counts)}`}
          </span>
        </Fragment>
      ))}
    </p>
  );
}
