import { useEffect, useMemo, useState } from "react";
import { usePlotTheme } from "../../../../components/plotTokens";
import { formatBytes, formatInteger } from "../../../../lib/format";
import { EmptyState, SectionTitle } from "../../../ui";
import type { CompileDagOptions, SampleShapeOptions } from "../../../faultTreeApi";
import { DEFAULT_OVERHEAD_RATIO } from "../controls/samplerOverheadRatio";
import { DEFAULT_TRIALS_COUNT } from "../controls/samplerTrials";
import { DeviceSummary } from "./DeviceSummary";
import { ExecutionGraph } from "./ExecutionGraph";
import { ExecutionStatGrid, type ExecutionStat } from "./ExecutionStatGrid";
import { PartitionDetail } from "./PartitionDetail";
import { PartitionLedger } from "./PartitionLedger";
import { SampleShaperSummary } from "./SampleShaperSummary";
import { buildExecutionScene, compactCount } from "./mathbox/executionScene";
import { useExecutionPlan } from "./mathbox/useExecutionPlan";
import "./execution-plan.css";

const DEFAULT_SAMPLE_OPTIONS: SampleShapeOptions = {
  numTrials: DEFAULT_TRIALS_COUNT,
  overheadRatio: DEFAULT_OVERHEAD_RATIO,
};

export function ExecutionPlanPanel({
  selectedId,
  compileOptions,
  blocked,
  blockedText,
}: {
  selectedId: string;
  compileOptions: CompileDagOptions;
  blocked?: boolean;
  blockedText?: string | null;
}) {
  const { palette } = usePlotTheme();
  const planCompileOptions = useMemo(
    () => ({ ...compileOptions, ccf: true }),
    [compileOptions],
  );
  const { plan, loading, error } = useExecutionPlan({
    selectedId,
    compileOptions: planCompileOptions,
    sampleOptions: DEFAULT_SAMPLE_OPTIONS,
  });

  const scene = useMemo(() => (plan ? buildExecutionScene(plan) : null), [plan]);
  const [selectedPartitionId, setSelectedPartitionId] = useState<number | null>(null);

  // Reset the selection whenever a new plan arrives.
  useEffect(() => {
    setSelectedPartitionId(null);
  }, [plan]);

  const selectedPartition = useMemo(
    () => plan?.partitions.find((partition) => partition.id === selectedPartitionId) ?? null,
    [plan, selectedPartitionId],
  );

  const totals = useMemo(() => {
    if (!plan) return null;
    const totalWorkItems = plan.partitions.reduce((sum, p) => sum + p.work_item_count, 0);
    const peakResident = plan.partitions.reduce((max, p) => Math.max(max, p.resident_bytes), 0);
    return { totalWorkItems, peakResident };
  }, [plan]);

  const title = plan?.device?.name
    ? `${plan.model ?? selectedId} on ${plan.device.name}`
    : ((plan?.model ?? selectedId) || "Execution plan");

  const summaryText = plan
    ? `${formatInteger(plan.partition_count)} kernel partitions · ${compactCount(totals?.totalWorkItems ?? 0)} launched work-items · ${plan.bitpack_type} lanes · ${plan.schema}`
    : "DirectEval partition graph: device capabilities, sample shaping, and per-kernel launch geometry.";

  return (
    <div className="panel span-3 exec-plan">
      <SectionTitle eyebrow="Execution Plan" title={title} text={summaryText} />
      {error ? <p className="live-demo-error playground-dag-error">{error}</p> : null}
      <ExecutionPlanBody
        blocked={blocked}
        blockedText={blockedText}
        loading={loading}
        plan={plan}
        scene={scene}
        palette={palette}
        totals={totals}
        selectedPartitionId={selectedPartitionId}
        selectedPartition={selectedPartition}
        onSelect={setSelectedPartitionId}
        hasError={Boolean(error)}
      />
    </div>
  );
}

function ExecutionPlanBody({
  blocked,
  blockedText,
  loading,
  plan,
  scene,
  palette,
  totals,
  selectedPartitionId,
  selectedPartition,
  onSelect,
  hasError,
}: {
  blocked?: boolean;
  blockedText?: string | null;
  loading: boolean;
  plan: ReturnType<typeof useExecutionPlan>["plan"];
  scene: ReturnType<typeof buildExecutionScene> | null;
  palette: ReturnType<typeof usePlotTheme>["palette"];
  totals: { totalWorkItems: number; peakResident: number } | null;
  selectedPartitionId: number | null;
  selectedPartition: NonNullable<ReturnType<typeof useExecutionPlan>["plan"]>["partitions"][number] | null;
  onSelect: (partitionId: number | null) => void;
  hasError: boolean;
}) {
  if (blocked) {
    return <EmptyState text={blockedText ?? "Select a model to compile its execution plan."} />;
  }
  if (hasError && !plan) {
    return <EmptyState text="The native execution plan is unavailable for this model." />;
  }
  if (!plan || !scene) {
    return <EmptyState text={loading ? "Computing execution plan…" : "Select a model to compile its execution plan."} />;
  }
  if (!scene.towers.length) {
    return <EmptyState text="This compiled PDAG produced no kernel partitions." />;
  }

  const totalStats: ExecutionStat[] = [
    { label: "partitions", value: formatInteger(plan.partition_count) },
    { label: "dependency edges", value: formatInteger(plan.dependency_edges.length) },
    { label: "semantic nodes", value: formatInteger(plan.node_count) },
    { label: "Σ work-items", value: compactCount(totals?.totalWorkItems ?? 0) },
    { label: "peak resident", value: formatBytes(totals?.peakResident ?? 0) },
    { label: "execution layers", value: formatInteger(scene.layerCount) },
  ];

  return (
    <div className={loading ? "exec-plan__inner exec-plan__inner--loading" : "exec-plan__inner"} aria-busy={loading}>
      <div className="exec-plan__summaries">
        <DeviceSummary device={plan.device} bitpackType={plan.bitpack_type} />
        <SampleShaperSummary sample={plan.sample_shaper} />
        <section className="exec-card" aria-label="Plan totals">
          <header className="exec-card__head">
            <p className="exec-card__eyebrow">Plan totals</p>
            <h3 className="exec-card__title">Partition graph</h3>
            <div className="exec-card__tags">
              <span className="exec-pill exec-pill--quiet">{plan.schema}</span>
            </div>
          </header>
          <ExecutionStatGrid stats={totalStats} columns={2} />
        </section>
      </div>

      <ExecutionGraph scene={scene} selectedId={selectedPartitionId} onSelect={onSelect} />

      <div className="exec-plan__inspect">
        <div className="exec-plan__ledger">
          <PartitionLedger
            partitions={plan.partitions}
            selectedId={selectedPartitionId}
            onSelect={onSelect}
            palette={palette}
          />
        </div>
        <div className="exec-plan__detail">
          <PartitionDetail partition={selectedPartition} palette={palette} onSelect={onSelect} />
        </div>
      </div>
    </div>
  );
}
