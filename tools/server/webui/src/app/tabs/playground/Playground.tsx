import { useCallback, useEffect, useMemo, useState } from "react";
import type { DagPayload, FaultTreePayload } from "../../components/canvas";
import { DagWebGlDiagram, FaultTreeDiagram } from "../../components/canvas";
import {
  fetchDagBundleForModel,
  fetchFaultTreeForModel,
  fetchXmlInputCatalog,
  type CompileDagOptions,
  type XmlInputDescriptor,
} from "../../api/faultTreeApi";
import { classNames } from "../../lib/format";
import { EmptyState, SectionTitle } from "../../ui";
import { ControlPanel } from "./ControlPanel";
import { ExecutionPlanPanel } from "./execution_plan";
import { SamplingTrajectoryPanel } from "./SamplingTrajectoryPanel";
import type { SamplingControls } from "./controls/useSamplingControls";
import { formatCounts, formatFaultTreeCounts } from "./format";
import { createSamplingRunMetricsStore } from "./samplingRunMetrics";
import "../../styles/playground.css";

const DEFAULT_COMPILE_OPTIONS: CompileDagOptions = {
  compilationLevel: 2,
  expandAtleastGates: false,
  expandXorGates: false,
  recognizeAtleastGroups: true,
  recognizeXorGroups: true,
  ccf: true,
  keepNullGates: false,
};

export function Playground() {
  const [catalog, setCatalog] = useState<XmlInputDescriptor[] | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [selectedId, setSelectedId] = useState("");
  const [faultTree, setFaultTree] = useState<FaultTreePayload | null>(null);
  const [faultTreeError, setFaultTreeError] = useState<string | null>(null);
  const [faultTreeLoading, setFaultTreeLoading] = useState(false);
  const [sourceDag, setSourceDag] = useState<DagPayload | null>(null);
  const [sourceDagError, setSourceDagError] = useState<string | null>(null);
  const [sourceDagLoading, setSourceDagLoading] = useState(false);
  const [compiledDag, setCompiledDag] = useState<DagPayload | null>(null);
  const [compiledDagError, setCompiledDagError] = useState<string | null>(null);
  const [compiledDagLoading, setCompiledDagLoading] = useState(false);
  const [kernelDag, setKernelDag] = useState<DagPayload | null>(null);
  const [kernelDagError, setKernelDagError] = useState<string | null>(null);
  const [kernelDagLoading, setKernelDagLoading] = useState(false);
  const [compileOptions, setCompileOptions] = useState<CompileDagOptions>(DEFAULT_COMPILE_OPTIONS);
  const [samplingControls, setSamplingControls] = useState<SamplingControls | null>(null);
  const samplingRunMetricsStore = useMemo(() => createSamplingRunMetricsStore(), []);

  useEffect(() => {
    let cancelled = false;
    setCatalogLoading(true);
    setCatalogError(null);
    fetchXmlInputCatalog()
      .then((payload) => {
        if (cancelled) return;
        setCatalog(payload.models);
        setCatalogError(null);
        setSelectedId((current) => {
          if (current && payload.modelIds.includes(current)) return current;
          return payload.modelIds[0] ?? "";
        });
      })
      .catch((reason) => {
        if (cancelled) return;
        setCatalog(null);
        setCatalogError(errorMessage(reason));
        setSelectedId("");
      })
      .finally(() => {
        if (!cancelled) setCatalogLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const modelIds = useMemo(
    () => (catalog ?? []).map((entry) => entry.modelId).sort((a, b) => a.localeCompare(b)),
    [catalog],
  );
  const selectedModel = useMemo(
    () => (catalog ?? []).find((entry) => entry.modelId === selectedId) ?? null,
    [catalog, selectedId],
  );

  useEffect(() => {
    if (!selectedId) {
      setFaultTree(null);
      setFaultTreeError(null);
      setFaultTreeLoading(false);
      return undefined;
    }
    let cancelled = false;
    setFaultTreeLoading(true);
    setFaultTreeError(null);
    fetchFaultTreeForModel(selectedId)
      .then((payload) => {
        if (cancelled) return;
        setFaultTree(payload);
        setFaultTreeError(null);
      })
      .catch((reason) => {
        if (cancelled) return;
        setFaultTree(null);
        setFaultTreeError(errorMessage(reason));
      })
      .finally(() => {
        if (!cancelled) setFaultTreeLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    setSourceDagLoading(true);
    setCompiledDagLoading(true);
    setKernelDagLoading(true);
    setSourceDagError(null);
    setCompiledDagError(null);
    setKernelDagError(null);
    fetchDagBundleForModel(selectedId, { ...compileOptions, ccf: true })
      .then((payload) => {
        if (cancelled) return;
        setSourceDag(payload.source);
        setCompiledDag(payload.compiled);
        setKernelDag(payload.kernel);
        setSourceDagError(null);
        setCompiledDagError(null);
        setKernelDagError(null);
      })
      .catch((reason) => {
        if (cancelled) return;
        const message = errorMessage(reason);
        setSourceDagError(message);
        setCompiledDagError(message);
        setKernelDagError(message);
      })
      .finally(() => {
        if (!cancelled) {
          setSourceDagLoading(false);
          setCompiledDagLoading(false);
          setKernelDagLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [compileOptions, selectedId]);

  const handleSamplingControlsChange = useCallback((controls: SamplingControls | null) => {
    setSamplingControls(controls);
  }, []);

  const titleModel = faultTreeLoading || sourceDagLoading || compiledDagLoading || kernelDagLoading
    ? selectedId
    : (kernelDag?.model ?? compiledDag?.model ?? sourceDag?.model ?? faultTree?.model ?? selectedId);
  const showStaleFaultTree = Boolean(faultTree && faultTreeLoading);
  const showStaleSourceDag = Boolean(sourceDag && sourceDagLoading);
  const showStaleCompiledDag = Boolean(compiledDag && compiledDagLoading);
  const showStaleKernelDag = Boolean(kernelDag && kernelDagLoading);
  const blocked = catalogLoading || Boolean(catalogError && !catalog);
  const blockedEmpty = blocked
    ? catalogLoading
      ? "Loading XML input catalog…"
      : (catalogError ?? "No XML inputs available.")
    : null;
  const faultTreeEmpty = blockedEmpty ?? (faultTreeLoading ? "Loading fault tree…" : "Select a model to render it.");
  const sourceEmpty = blockedEmpty ?? (sourceDagLoading ? "Loading source DAG…" : "Select a model to render it.");
  const compiledEmpty = blockedEmpty ?? (compiledDagLoading ? "Loading compiled PDAG…" : "Select a model to render it.");
  const kernelEmpty = blockedEmpty ?? (kernelDagLoading ? "Loading kernel DAG…" : "Select a model to render it.");

  return (
    <section className="section-grid playground sticky-panel-host">
      <div className="panel span-3">
        <SectionTitle
          eyebrow="Input: Fault Tree"
          title={faultTree?.model ? `${faultTree.model}: logical structure` : "Fault-tree structure"}
          text={faultTree ? formatFaultTreeCounts(faultTree) : "Logical gate and basic-event structure from the selected MEF input."}
        />
        {faultTreeError && faultTree ? <p className="live-demo-error playground-dag-error">{faultTreeError}</p> : null}
        <FaultTreePanel
          tree={faultTree}
          error={faultTreeError}
          loading={faultTreeLoading}
          showStale={showStaleFaultTree}
          emptyText={faultTreeEmpty}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Input MEF DAG"
          title={titleModel}
          text={sourceDag ? formatCounts(sourceDag.counts) : "Parsed XML structure from the selected fault-tree input."}
        />
        {catalogError && !catalog ? <EmptyState text={catalogError} /> : null}
        {sourceDagError && sourceDag ? <p className="live-demo-error playground-dag-error">{sourceDagError}</p> : null}
        <DagPanel
          dag={sourceDag}
          error={sourceDagError}
          loading={sourceDagLoading}
          showStale={showStaleSourceDag}
          emptyText={sourceEmpty}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Compiled PDAG"
          title={titleModel}
          text={compiledDag ? compileDetail(compiledDag, compileOptions) : "Native compiler output for the current normalization switches."}
        />
        {compiledDagError && compiledDag ? <p className="live-demo-error playground-dag-error">{compiledDagError}</p> : null}
        <DagPanel
          dag={compiledDag}
          error={compiledDagError}
          loading={compiledDagLoading}
          showStale={showStaleCompiledDag}
          emptyText={compiledEmpty}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Kernel Execution DAG"
          title={titleModel}
          text={kernelDag ? kernelDetail(kernelDag) : "LayerManager task DAG grouped by execution layer and kernel partition."}
        />
        {kernelDagError && kernelDag ? <p className="live-demo-error playground-dag-error">{kernelDagError}</p> : null}
        <DagPanel
          dag={kernelDag}
          error={kernelDagError}
          loading={kernelDagLoading}
          showStale={showStaleKernelDag}
          emptyText={kernelEmpty}
          className="playground-dag-view--kernel"
        />
      </div>

      <ExecutionPlanPanel
        selectedId={selectedId}
        compileOptions={compileOptions}
        blocked={blocked}
        blockedText={blockedEmpty}
      />

      <SamplingTrajectoryPanel
        samplingControls={samplingControls}
        runMetricsStore={samplingRunMetricsStore}
      />

      <ControlPanel
        selectedId={selectedId}
        onSelectedIdChange={setSelectedId}
        modelIds={modelIds}
        oracleP={selectedModel?.oracle_p ?? null}
        blocked={blocked}
        compileOptions={compileOptions}
        onCompileOptionsChange={setCompileOptions}
        catalogLoading={catalogLoading}
        catalogError={catalogError}
        modelCount={modelIds.length}
        faultTree={faultTree}
        sourceDag={sourceDag}
        compiledDag={compiledDag}
        kernelDag={kernelDag}
        samplingRunMetricsStore={samplingRunMetricsStore}
        onSamplingControlsChange={handleSamplingControlsChange}
      />
    </section>
  );
}

function FaultTreePanel({
  tree,
  error,
  loading,
  showStale,
  emptyText,
}: {
  tree: FaultTreePayload | null;
  error: string | null;
  loading: boolean;
  showStale: boolean;
  emptyText: string;
}) {
  if (error && !tree) return <EmptyState text={error} />;
  if (tree) {
    return (
      <div
        className={classNames("playground-dag-view", showStale && "playground-dag-view--loading")}
        aria-busy={showStale || loading}
      >
        <FaultTreeDiagram tree={tree} />
      </div>
    );
  }
  if (error) return null;
  return <EmptyState text={emptyText} />;
}

function DagPanel({
  dag,
  error,
  loading,
  showStale,
  emptyText,
  className,
}: {
  dag: DagPayload | null;
  error: string | null;
  loading: boolean;
  showStale: boolean;
  emptyText: string;
  className?: string;
}) {
  if (error && !dag) return <EmptyState text={error} />;
  if (dag) {
    return (
      <div className={classNames("playground-dag-view", className, showStale && "playground-dag-view--loading")} aria-busy={showStale || loading}>
        <DagWebGlDiagram dag={dag} />
      </div>
    );
  }
  if (error) return null;
  return <EmptyState text={emptyText} />;
}

function compileDetail(dag: DagPayload, options: CompileDagOptions): string {
  const counts = formatCounts(dag.counts);
  const switches = [
    options.expandAtleastGates ? "ATLEAST expanded" : "ATLEAST preserved",
    options.expandXorGates ? "XOR expanded" : "XOR preserved",
    options.recognizeAtleastGroups ? "ATLEAST recognition on" : "ATLEAST recognition off",
    options.recognizeXorGroups ? "XOR recognition on" : "XOR recognition off",
    "CCF on",
    options.keepNullGates ? "null gates kept" : "null gates removed",
  ].join(" · ");
  return `${counts} · C${options.compilationLevel} · ${switches}`;
}

function kernelDetail(dag: DagPayload): string {
  const counts = formatCounts(dag.counts);
  const partitionCount = dag.kernel?.partition_count ?? dag.kernel?.partitions?.length ?? 0;
  const sampleShape = dag.kernel?.sample_shape;
  const shape = sampleShape
    ? ` · sample ${sampleShape.batch_size}x${sampleShape.bitpacks_per_batch}${sampleShape.total_iterations ? ` · ${sampleShape.total_iterations} iters` : ""}`
    : "";
  return `${counts} · ${partitionCount} kernel partitions${shape}`;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
