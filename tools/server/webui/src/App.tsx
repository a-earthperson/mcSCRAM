import { useEffect, useMemo, useState } from "react";
import { Plot, inkColorscale, inkPalette } from "./components/Plot";
import {
  Artifacts,
  CompilerData,
  CompilerRow,
  ExactResult,
  ModelCard,
  PerformanceRun,
  PlotAsset,
  SiteData,
  Summary,
  Tallies,
  TraceIndexData,
  TraceModelData,
  TraceSeries,
} from "./types";
import { classNames, formatBytes, formatDurationMs, formatInteger, formatNumber } from "./lib/format";

type Tab = "overview" | "performance" | "compiler" | "models" | "artifacts";

const tabs: Array<{ id: Tab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "performance", label: "Performance" },
  { id: "compiler", label: "Compiler" },
  { id: "models", label: "Model Explorer" },
  { id: "artifacts", label: "Artifacts" },
];

async function loadJson<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!response.ok) throw new Error(`Failed to load ${name}: ${response.status}`);
  return (await response.json()) as T;
}

function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [data, setData] = useState<SiteData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewAsset, setPreviewAsset] = useState<PlotAsset | null>(null);

  useEffect(() => {
    Promise.all([
      loadJson<Summary>("summary.json"),
      loadJson<ModelCard[]>("models.json"),
      loadJson<PerformanceRun[]>("performance.json"),
      loadJson<CompilerData>("compiler.json"),
      loadJson<TraceIndexData>("traces-index.json"),
      loadJson<Artifacts>("artifacts.json"),
      loadJson<Tallies>("tallies.json"),
      loadJson<ExactResult[]>("exact-results.json"),
    ])
      .then(([summary, models, performance, compiler, traces, artifacts, tallies, exactResults]) => {
        setData({ summary, models, performance, compiler, traces, artifacts, tallies, exactResults });
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  if (error) {
    return <main className="shell error-state">Could not load generated benchmark data: {error}</main>;
  }

  if (!data) {
    return (
      <main className="shell loading-state">
        <div className="pulse" />
        <p>Loading mcSCRAM benchmark atlas...</p>
      </main>
    );
  }

  return (
    <main className="shell">
      <Hero data={data} />
      <nav className="tabs" aria-label="Benchmark site sections">
        {tabs.map((item) => (
          <button key={item.id} className={classNames("tab", tab === item.id && "active")} onClick={() => setTab(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>
      {tab === "overview" && <Overview data={data} onPreviewAsset={setPreviewAsset} />}
      {tab === "performance" && <PerformanceDashboard data={data} />}
      {tab === "compiler" && <CompilerDashboard data={data} />}
      {tab === "models" && <ModelExplorer data={data} onPreviewAsset={setPreviewAsset} />}
      {tab === "artifacts" && <ArtifactGallery artifacts={data.artifacts} onPreviewAsset={setPreviewAsset} />}
      {previewAsset && <AssetPreviewModal asset={previewAsset} onClose={() => setPreviewAsset(null)} />}
    </main>
  );
}

function Hero({ data }: { data: SiteData }) {
  const { summary } = data;
  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">Static research supplement</p>
        <h1>mcSCRAM Benchmarks</h1>
        <h2>Aralia Fault Trees</h2>
        <p className="lede">
          Interactive benchmark results for mcSCRAM: convergence behavior, GPU throughput, compiler reductions, per-model traces,
          generated plots, and the low-level artifacts that back the headline comparisons.
        </p>
        <div className="hero-actions">
          <a href={summary.mcScramRepo}>mcSCRAM</a>
          <a href={summary.sourceRepo}>Benchmark Repo</a>
          <a href={summary.doi}>Zenodo DOI</a>
        </div>
      </div>
      <div className="hero-card">
        <Metric label="Models" value={formatInteger(summary.counts.models)} />
        <Metric label="Convergence Runs" value={formatInteger(summary.counts.performanceRuns)} />
        <Metric label="Compiler Rows" value={formatInteger(summary.counts.compilerRows)} />
        <Metric label="Trace Files" value={formatInteger(summary.counts.traceFiles)} />
      </div>
    </header>
  );
}

function Overview({ data, onPreviewAsset }: { data: SiteData; onPreviewAsset: (asset: PlotAsset) => void }) {
  const topAssets = useMemo(
    () =>
      data.artifacts.plots
        .filter((asset) => asset.category !== "Convergence")
        .slice(0, 8),
    [data.artifacts.plots],
  );
  const fastest = useMemo(
    () => [...data.performance].sort((a, b) => (b.bitsPerSecond ?? 0) - (a.bitsPerSecond ?? 0)).slice(0, 8),
    [data.performance],
  );

  return (
    <section className="section-grid">
      <div className="panel span-2">
        <SectionTitle
          eyebrow="What is published"
          title="A browser-first view of the benchmark archive"
          text="The site is generated from the LFS-backed CSV archive into compact JSON. High-level charts stay interactive, and every aggregate links back to raw result files, scripts, or figure artifacts."
        />
        <div className="metric-grid">
          <Metric label="Plot Assets" value={formatInteger(data.summary.counts.plotAssets)} detail="PNG/SVG copied for web display" />
          <Metric label="Vector Figures" value={formatInteger(data.summary.counts.vectorFigures)} detail="EPS downloads linked to GitHub" />
          <Metric label="Scripts" value={formatInteger(data.summary.counts.scripts)} detail="MATLAB, Python, and shell provenance" />
          <Metric label="Tally Events" value={formatInteger(data.summary.counts.tallyEvents)} detail="Low-level monitored marginal tallies" />
        </div>
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Canonical sources" title="Data lineage" />
        <dl className="source-list">
          {Object.entries(data.summary.canonicalSources).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Exact corner" title="BDD / CCF checks" />
        <CompactTable
          headers={["Model", "Method", "Probability", "Products", "Total time"]}
          rows={data.exactResults.map((row) => [
            row.modelId,
            row.method,
            formatNumber(row.probability, 5),
            formatInteger(row.sopProducts),
            `${formatNumber(row.totalTime, 3)} s`,
          ])}
        />
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Throughput envelope" title="Bits per second vs PDAG nodes" />
        <ThroughputPlot runs={data.performance} />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Fastest runs" title="Top measured throughput" />
        <CompactTable
          headers={["Model", "Device", "Nodes", "Bits/s"]}
          rows={fastest.map((run) => [
            run.modelName,
            run.deviceName,
            formatInteger(run.numNodes),
            formatNumber(run.bitsPerSecond, 3),
          ])}
        />
      </div>
      <div className="panel span-3">
        <SectionTitle eyebrow="Narrative figures" title="Curated plot gallery" />
        <div className="asset-strip">
          {topAssets.map((asset) => (
            <AssetCard key={asset.id} asset={asset} onPreview={onPreviewAsset} />
          ))}
        </div>
      </div>
    </section>
  );
}

function PerformanceDashboard({ data }: { data: SiteData }) {
  const [device, setDevice] = useState("all");
  const [flag, setFlag] = useState("all");
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () =>
      data.performance.filter((run) => {
        const matchesDevice = device === "all" || run.deviceName === device;
        const matchesFlag = flag === "all" || run.compileFlag === flag;
        const q = query.toLowerCase();
        const matchesQuery = !q || run.modelName.toLowerCase().includes(q) || run.family.toLowerCase().includes(q);
        return matchesDevice && matchesFlag && matchesQuery;
      }),
    [data.performance, device, flag, query],
  );
  const topRuns = useMemo(() => [...filtered].sort((a, b) => (b.bitsPerSecond ?? 0) - (a.bitsPerSecond ?? 0)).slice(0, 40), [filtered]);

  return (
    <section className="section-grid">
      <div className="panel span-3 controls">
        <Select label="Device" value={device} onChange={setDevice} options={["all", ...data.summary.facets.devices]} />
        <Select label="Compile flag" value={flag} onChange={setFlag} options={["all", ...data.summary.facets.compileFlags]} />
        <label>
          Search
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="baobab, das, isp..." />
        </label>
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Scaling" title="Throughput vs graph size" />
        <ThroughputPlot runs={filtered} />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Accuracy" title="Relative error vs convergence time" />
        <Plot
          data={[
            {
              type: "scattergl",
              mode: "markers",
              x: filtered.map((run) => run.convergenceTimeMs),
              y: filtered.map((run) => run.relError),
              text: filtered.map((run) => `${run.modelName}<br>${run.deviceName}<br>${run.compileFlag}`),
              marker: {
                size: 9,
                color: filtered.map((run) => run.numNodes ?? 0),
                colorscale: inkColorscale,
                showscale: true,
                line: { color: inkPalette.ink, width: 0.5 },
                opacity: 0.85,
              },
              hovertemplate: "%{text}<br>time=%{x:.3s} ms<br>rel error=%{y:.3e}<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "convergence time (ms)", type: "log" }, yaxis: { title: "relative error", type: "log" } }}
        />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Internals" title="Kernel and layer layout" />
        <Plot
          data={[
            {
              type: "scatter",
              mode: "markers",
              x: filtered.map((run) => run.numLayers),
              y: filtered.map((run) => run.numKernels),
              text: filtered.map((run) => `${run.modelName}<br>${run.numNodes} nodes`),
              marker: {
                size: filtered.map((run) => Math.max(7, Math.log10((run.numNodes ?? 1) + 1) * 5)),
                color: inkPalette.ink,
                opacity: 0.7,
                line: { color: inkPalette.paper, width: 0.5 },
              },
              hovertemplate: "%{text}<br>layers=%{x}<br>kernels=%{y}<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "layers" }, yaxis: { title: "kernels" } }}
        />
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Distribution" title="Top monitored tallies" text="Largest marginal means from the 300 second RTX 4090 tally export." />
        <Plot
          data={[
            {
              type: "bar",
              x: data.tallies.topEvents.slice(0, 35).map((event) => event.eventId),
              y: data.tallies.topEvents.slice(0, 35).map((event) => event.mean),
              marker: {
                color: inkPalette.ink,
                line: { color: inkPalette.ink, width: 0.5 },
              },
              hovertemplate: "%{x}<br>mean=%{y:.3e}<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "event id", tickangle: -45 }, yaxis: { title: "mean", type: "log" } }}
        />
      </div>
      <div className="panel span-3">
        <SectionTitle eyebrow="Run table" title={`${formatInteger(filtered.length)} filtered convergence rows`} />
        <RunTable runs={topRuns} />
      </div>
    </section>
  );
}

function CompilerDashboard({ data }: { data: SiteData }) {
  const [flag, setFlag] = useState("all");
  const rows = useMemo(() => data.compiler.rows.filter((row) => flag === "all" || row.compileFlag === flag), [data.compiler.rows, flag]);
  const summaries = useMemo(
    () => data.compiler.summaries.filter((row) => flag === "all" || row.compileFlag === flag),
    [data.compiler.summaries, flag],
  );
  const topCompression = useMemo(
    () =>
      [...rows]
        .filter((row) => row.compressionAllNodes !== null)
        .sort((a, b) => (a.compressionAllNodes ?? 1) - (b.compressionAllNodes ?? 1))
        .slice(0, 35),
    [rows],
  );

  return (
    <section className="section-grid">
      <div className="panel span-3 controls">
        <Select label="Compile flag" value={flag} onChange={setFlag} options={["all", ...data.summary.facets.compileFlags]} />
        <Metric label="Rows" value={formatInteger(rows.length)} />
        <Metric label="Median all-node compression" value={formatNumber(medianOf(rows.map((row) => row.compressionAllNodes)), 3)} />
        <Metric label="Median preprocessing" value={`${formatNumber(medianOf(rows.map((row) => row.preprocessingTimeUs)), 3)} us`} />
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Reduction" title="Compiler compression by flag and normalization switches" />
        <Plot
          data={[
            {
              type: "bar",
              x: summaries.map((row) => `${row.compileFlag} kn:${row.noKn ? "off" : "on"} xor:${row.noXor ? "off" : "on"}`),
              y: summaries.map((row) => row.compressionAllNodes.median),
              error_y: {
                type: "data",
                symmetric: false,
                array: summaries.map((row) => Math.max(0, (row.compressionAllNodes.p95 ?? 0) - (row.compressionAllNodes.median ?? 0))),
                arrayminus: summaries.map((row) => Math.max(0, (row.compressionAllNodes.median ?? 0) - (row.compressionAllNodes.p05 ?? 0))),
              },
              marker: {
                color: inkPalette.indigo,
                line: { color: inkPalette.ink, width: 0.5 },
              },
              hovertemplate: "%{x}<br>median=%{y:.3f}<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "compiler configuration", tickangle: -35 }, yaxis: { title: "all-node compression factor" } }}
        />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Structure" title="Fan-in vs compression" />
        <Plot
          data={[
            {
              type: "scattergl",
              mode: "markers",
              x: rows.map((row) => row.pdagAvgFanIn),
              y: rows.map((row) => row.compressionAllNodes),
              text: rows.map((row) => `${row.modelName}<br>${row.compileFlag}`),
              marker: {
                size: 7,
                color: rows.map((row) => row.pdagGates ?? 0),
                colorscale: inkColorscale,
                showscale: true,
                line: { color: inkPalette.ink, width: 0.5 },
                opacity: 0.85,
              },
              hovertemplate: "%{text}<br>avg fan-in=%{x:.3f}<br>compression=%{y:.3f}<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "PDAG avg gate fan-in" }, yaxis: { title: "all-node compression factor" } }}
        />
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Compiler cost" title="Preprocessing time vs model gates" />
        <Plot
          data={[
            {
              type: "scattergl",
              mode: "markers",
              x: rows.map((row) => row.modelGates),
              y: rows.map((row) => row.preprocessingTimeUs),
              text: rows.map((row) => `${row.modelName}<br>${row.compileFlag}`),
              marker: {
                size: 7,
                color: inkPalette.ochre,
                opacity: 0.78,
                line: { color: inkPalette.ink, width: 0.5 },
              },
              hovertemplate: "%{text}<br>gates=%{x}<br>time=%{y:.3s} us<extra></extra>",
            },
          ]}
          layout={{ xaxis: { title: "model gates", type: "log" }, yaxis: { title: "preprocessing time (us)", type: "log" } }}
        />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Most aggressive reductions" title="Lowest all-node factors" />
        <CompactTable
          headers={["Model", "Flag", "KN", "XOR", "Factor"]}
          rows={topCompression.map((row) => [
            row.modelName,
            row.compileFlag,
            row.noKn ? "off" : "on",
            row.noXor ? "off" : "on",
            formatNumber(row.compressionAllNodes, 3),
          ])}
        />
      </div>
    </section>
  );
}

function ModelExplorer({ data, onPreviewAsset }: { data: SiteData; onPreviewAsset: (asset: PlotAsset) => void }) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(data.models[0]?.modelId ?? "");
  const [loadedTraceModel, setLoadedTraceModel] = useState("");
  const [modelTraces, setModelTraces] = useState<TraceSeries[]>([]);
  const filteredModels = useMemo(
    () =>
      data.models.filter((model) => {
        const q = query.toLowerCase();
        return !q || model.modelName.toLowerCase().includes(q) || model.family.toLowerCase().includes(q);
      }),
    [data.models, query],
  );
  const selected = data.models.find((model) => model.modelId === selectedId) ?? filteredModels[0] ?? data.models[0];
  const modelRuns = useMemo(() => data.performance.filter((run) => run.modelId === selected.modelId), [data.performance, selected.modelId]);
  const modelCompiler = useMemo(() => data.compiler.rows.filter((row) => row.modelId === selected.modelId), [data.compiler.rows, selected.modelId]);
  const modelAssets = useMemo(() => data.artifacts.plots.filter((asset) => asset.modelId === selected.modelId).slice(0, 6), [data.artifacts.plots, selected.modelId]);
  const trace = loadedTraceModel === selected.modelId ? modelTraces[0] : undefined;

  useEffect(() => {
    let cancelled = false;
    setLoadedTraceModel("");
    setModelTraces([]);
    loadJson<TraceModelData>(`traces/${selected.modelId}.json`)
      .then((payload) => {
        if (cancelled) return;
        setLoadedTraceModel(selected.modelId);
        setModelTraces(payload.series.sort((a, b) => b.setId.localeCompare(a.setId)));
      })
      .catch(() => {
        if (cancelled) return;
        setLoadedTraceModel(selected.modelId);
        setModelTraces([]);
      });
    return () => {
      cancelled = true;
    };
  }, [selected.modelId]);

  return (
    <section className="section-grid">
      <div className="panel model-list">
        <SectionTitle eyebrow="Models" title="Search the archive" />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search model or family" />
        <div className="model-buttons">
          {filteredModels.map((model) => (
            <button
              key={model.modelId}
              className={classNames("model-button", selected.modelId === model.modelId && "active")}
              onClick={() => setSelectedId(model.modelId)}
            >
              <strong>{model.modelName}</strong>
              <span>{model.family} · {formatInteger(model.performanceRunCount)} runs · {formatInteger(model.compilerRunCount)} compiler rows</span>
            </button>
          ))}
        </div>
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow={selected.family} title={selected.modelName} text="Reference probabilities, best observed performance, and compiler reduction summary." />
        <div className="metric-grid">
          <Metric label="Oracle probability" value={formatNumber(selected.reference?.oracleP, 4)} />
          <Metric label="Rare-event estimate" value={formatNumber(selected.reference?.rareP, 4)} />
          <Metric label="Best bits/s" value={formatNumber(selected.bestBitsPerSecond, 3)} />
          <Metric label="Best relative error" value={formatNumber(selected.bestRelError, 3)} />
          <Metric label="Max PDAG nodes" value={formatInteger(selected.maxNodes)} />
          <Metric label="Best compression" value={formatNumber(selected.bestCompressionAllNodes, 3)} detail={selected.bestCompileFlag ?? undefined} />
        </div>
      </div>
      <div className="panel span-2">
        <SectionTitle
          eyebrow="Convergence trace"
          title={trace ? `${trace.setId} · ${trace.timestamp}` : loadedTraceModel === selected.modelId ? "No trace available" : "Loading trace"}
        />
        {trace ? (
          <TracePlot trace={trace} />
        ) : (
          <EmptyState text={loadedTraceModel === selected.modelId ? "No sampled convergence trace was found for this model." : "Loading sampled convergence traces for this model..."} />
        )}
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Run detail" title="Convergence rows" />
        <CompactTable
          headers={["Device", "Flag", "Time", "Bits/s", "Rel err"]}
          rows={modelRuns.map((run) => [
            run.deviceName,
            run.compileFlag,
            formatDurationMs(run.convergenceTimeMs),
            formatNumber(run.bitsPerSecond, 3),
            formatNumber(run.relError, 3),
          ])}
        />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Compiler detail" title="Compression configurations" />
        <CompactTable
          headers={["Flag", "KN", "XOR", "PDAG gates", "Factor"]}
          rows={modelCompiler.slice(0, 60).map((row) => [
            row.compileFlag,
            row.noKn ? "off" : "on",
            row.noXor ? "off" : "on",
            formatInteger(row.pdagGates),
            formatNumber(row.compressionAllNodes, 3),
          ])}
        />
      </div>
      <div className="panel span-2">
        <SectionTitle eyebrow="Generated figures" title="Per-model plot artifacts" />
        {modelAssets.length ? (
          <div className="asset-strip compact">
            {modelAssets.map((asset) => (
              <AssetCard key={asset.id} asset={asset} onPreview={onPreviewAsset} />
            ))}
          </div>
        ) : (
          <EmptyState text="No copied PNG/SVG plot matched this model id." />
        )}
      </div>
    </section>
  );
}

function ArtifactGallery({ artifacts, onPreviewAsset }: { artifacts: Artifacts; onPreviewAsset: (asset: PlotAsset) => void }) {
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const categories = useMemo(() => ["all", ...Array.from(new Set(artifacts.plots.map((plot) => plot.category))).sort()], [artifacts.plots]);
  const plots = useMemo(
    () =>
      artifacts.plots.filter((plot) => {
        const q = query.toLowerCase();
        return (category === "all" || plot.category === category) && (!q || plot.title.toLowerCase().includes(q) || plot.source.toLowerCase().includes(q));
      }),
    [artifacts.plots, category, query],
  );

  return (
    <section className="section-grid">
      <div className="panel span-3 controls">
        <Select label="Category" value={category} onChange={setCategory} options={categories} />
        <label>
          Search artifacts
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="throughput, convergence, compiler..." />
        </label>
        <Metric label="Displayed plots" value={formatInteger(plots.length)} />
      </div>
      <div className="panel span-3">
        <SectionTitle eyebrow="PNG/SVG" title="Web plot gallery" text="Browser-friendly images are copied into the Pages artifact; EPS and source scripts remain linked to GitHub." />
        <div className="asset-grid">
          {plots.map((asset) => (
            <AssetCard key={asset.id} asset={asset} onPreview={onPreviewAsset} />
          ))}
        </div>
      </div>
      <div className="panel">
        <SectionTitle eyebrow="Source code" title="Plotting and benchmark scripts" />
        <LinkList artifacts={artifacts.scripts} />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="CSV" title="Canonical data files" />
        <LinkList artifacts={artifacts.dataFiles} />
      </div>
      <div className="panel">
        <SectionTitle eyebrow="EPS" title="Vector figure archive" />
        <LinkList artifacts={artifacts.vectorFigures} />
      </div>
    </section>
  );
}

function ThroughputPlot({ runs }: { runs: PerformanceRun[] }) {
  const devices = Array.from(new Set(runs.map((run) => run.deviceName))).filter(Boolean);
  const traces = devices.map((device) => {
    const subset = runs.filter((run) => run.deviceName === device);
    return {
      type: "scattergl",
      mode: "markers",
      name: device,
      x: subset.map((run) => run.numNodes),
      y: subset.map((run) => run.bitsPerSecond),
      text: subset.map((run) => `${run.modelName}<br>${run.compileFlag}<br>${formatDurationMs(run.convergenceTimeMs)}`),
      marker: { size: 8, opacity: 0.75, line: { color: inkPalette.paper, width: 0.5 } },
      hovertemplate: "%{text}<br>nodes=%{x}<br>bits/s=%{y:.3s}<extra></extra>",
    };
  });
  return <Plot data={traces} layout={{ xaxis: { title: "PDAG nodes", type: "log" }, yaxis: { title: "bits per second", type: "log" } }} />;
}

function TracePlot({ trace }: { trace: TraceSeries }) {
  const x = trace.pointsData.map((point) => point.trials);
  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "lines",
          name: "p99",
          x,
          y: trace.pointsData.map((point) => point.p99),
          line: { width: 0 },
          showlegend: false,
          hoverinfo: "skip",
        },
        {
          type: "scatter",
          mode: "lines",
          name: "p01\u2013p99 band",
          x,
          y: trace.pointsData.map((point) => point.p01),
          fill: "tonexty",
          fillcolor: "rgba(27, 29, 32, 0.12)",
          line: { width: 0 },
          hoverinfo: "skip",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "mean",
          x,
          y: trace.pointsData.map((point) => point.mean),
          line: { color: inkPalette.ink, width: 1.6 },
          marker: { size: 4, color: inkPalette.ink },
          hovertemplate: "trials=%{x:.3s}<br>mean=%{y:.4e}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines",
          name: "oracle",
          x,
          y: trace.pointsData.map(() => trace.oracleP),
          line: { color: inkPalette.madder, dash: "dot", width: 1.5 },
          hovertemplate: "oracle=%{y:.4e}<extra></extra>",
        },
      ]}
      layout={{ xaxis: { title: "sampled trials", type: "log" }, yaxis: { title: "probability", type: "log" } }}
    />
  );
}

function RunTable({ runs }: { runs: PerformanceRun[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Device</th>
            <th>Flag</th>
            <th>Nodes</th>
            <th>Layers</th>
            <th>Mean</th>
            <th>Oracle</th>
            <th>Rel err</th>
            <th>Time</th>
            <th>Bits/s</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.runId}>
              <td>{run.modelName}</td>
              <td>{run.deviceName}</td>
              <td>{run.compileFlag}</td>
              <td>{formatInteger(run.numNodes)}</td>
              <td>{formatInteger(run.numLayers)}</td>
              <td>{formatNumber(run.mean, 4)}</td>
              <td>{formatNumber(run.oracleP, 4)}</td>
              <td>{formatNumber(run.relError, 3)}</td>
              <td>{formatDurationMs(run.convergenceTimeMs)}</td>
              <td>{formatNumber(run.bitsPerSecond, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AssetCard({ asset, onPreview }: { asset: PlotAsset; onPreview: (asset: PlotAsset) => void }) {
  const assetUrl = `${import.meta.env.BASE_URL}${asset.assetPath}`;

  return (
    <article className="asset-card">
      <a
        className="asset-card-link"
        href={assetUrl}
        onClick={(event) => {
          event.preventDefault();
          onPreview(asset);
        }}
        aria-label={`Preview ${asset.title}`}
      >
        <img src={assetUrl} alt={asset.title} loading="lazy" />
        <div>
          <strong>{asset.title}</strong>
          <span>{asset.category} · {asset.extension.toUpperCase()} · {formatBytes(asset.sizeBytes)}</span>
        </div>
      </a>
    </article>
  );
}

function AssetPreviewModal({ asset, onClose }: { asset: PlotAsset; onClose: () => void }) {
  const assetUrl = `${import.meta.env.BASE_URL}${asset.assetPath}`;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.classList.add("modal-open");
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("modal-open");
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="image-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${asset.title} preview`}
        onClick={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <p className="eyebrow">{asset.category}</p>
            <h2>{asset.title}</h2>
            <p>{asset.source}</p>
          </div>
          <button className="modal-close" type="button" onClick={onClose} aria-label="Close preview">
            Close
          </button>
        </header>
        <div className="modal-image-frame">
          <img src={assetUrl} alt={asset.title} />
        </div>
        <footer>
          <a href={assetUrl}>Open bundled image</a>
          <a href={asset.sourceUrl}>View source path on GitHub</a>
        </footer>
      </section>
    </div>
  );
}

function LinkList({ artifacts }: { artifacts: Array<{ title: string; path: string; url: string; extension: string; sizeBytes?: number; kind?: string }> }) {
  return (
    <ul className="link-list">
      {artifacts.map((artifact) => (
        <li key={artifact.path}>
          <a href={artifact.url}>{artifact.title}</a>
          <span>{artifact.kind ?? artifact.extension.toUpperCase()} · {artifact.path}{artifact.sizeBytes ? ` · ${formatBytes(artifact.sizeBytes)}` : ""}</span>
        </li>
      ))}
    </ul>
  );
}

function CompactTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.join("-")}-${index}`}>
              {row.map((cell, cellIndex) => <td key={`${cell}-${cellIndex}`}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionTitle({ eyebrow, title, text }: { eyebrow: string; title: string; text?: string }) {
  return (
    <div className="section-title">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      {text && <p>{text}</p>}
    </div>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option === "all" ? "All" : option}
          </option>
        ))}
      </select>
    </label>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

function medianOf(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number => value !== null && value !== undefined && !Number.isNaN(value)).sort((a, b) => a - b);
  if (!present.length) return null;
  const middle = Math.floor(present.length / 2);
  return present.length % 2 ? present[middle] : (present[middle - 1] + present[middle]) / 2;
}

export default App;
