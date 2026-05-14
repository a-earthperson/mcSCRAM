import { Fragment, useEffect, useMemo, useState } from "react";
import { Plot, inkColorscale, inkColorway, inkPalette } from "./components/Plot";
import {
  Artifacts,
  CompileLevel,
  CompileLevelStats,
  CompilerData,
  CompilerSummary,
  ConnectiveEffectGroup,
  ConnectiveSwitch,
  ConvergenceView,
  DeviceSummary,
  ExactResult,
  HardwareData,
  HardwareDevice,
  LogFit,
  ModelCard,
  NarrativeCard,
  NumberSummary,
  PerformanceRun,
  PlotAsset,
  SetIndexEntry,
  SiteData,
  SpeedupData,
  Summary,
  Tallies,
  TraceIndexData,
  TraceModelData,
  TraceSeries,
} from "./types";
import {
  classNames,
  formatBytes,
  formatDurationMs,
  formatInteger,
  formatNumber,
} from "./lib/format";

type Tab = "overview" | "hardware" | "performance" | "compiler" | "convergence" | "models" | "artifacts";

const tabs: Array<{ id: Tab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "hardware", label: "Hardware" },
  { id: "performance", label: "Performance" },
  { id: "compiler", label: "Compiler" },
  { id: "convergence", label: "Convergence" },
  { id: "models", label: "Models" },
  { id: "artifacts", label: "Artifacts" },
];

const HOST_SHORT = "Core i7 10700";

const DEVICE_ORDER = [
  HOST_SHORT,
  "UHD 630",
  "GTX 1660 S",
  "RTX 4090",
  "RTX 5090",
  "L40S",
];

const DEVICE_COLOR: Record<string, string> = {
  [HOST_SHORT]: inkPalette.paynes,
  "UHD 630": inkPalette.ochre,
  "GTX 1660 S": inkPalette.indigo,
  "RTX 4090": inkPalette.madder,
  "RTX 5090": inkPalette.sage,
  "L40S": inkPalette.indigoDeep,
};

const FLAG_COLOR: Record<string, string> = {
  C0: inkPalette.paynes,
  C1: inkPalette.indigo,
  C2: inkPalette.madder,
  C3: inkPalette.sage,
  C4: inkPalette.ochre,
  C8: inkPalette.indigoDeep,
};

const ACTIVE_COMPILER_FLAGS = ["C0", "C1", "C2", "C3", "C4", "C8"];

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
      loadJson<NarrativeCard[]>("narrative.json"),
      loadJson<ModelCard[]>("models.json"),
      loadJson<PerformanceRun[]>("performance.json"),
      loadJson<DeviceSummary[]>("device-summaries.json"),
      loadJson<SpeedupData>("speedups.json"),
      loadJson<CompilerData>("compiler.json"),
      loadJson<ConvergenceView>("convergence.json"),
      loadJson<TraceIndexData>("traces-index.json"),
      loadJson<Record<string, SetIndexEntry>>("set-index.json"),
      loadJson<Artifacts>("artifacts.json"),
      loadJson<Tallies>("tallies.json"),
      loadJson<ExactResult[]>("exact-results.json"),
      loadJson<HardwareData>("hardware.json"),
    ])
      .then(([summary, narrative, models, performance, devices, speedups, compiler, convergence, traces, setIndex, artifacts, tallies, exactResults, hardware]) => {
        setData({ summary, narrative, models, performance, devices, speedups, compiler, convergence, traces, setIndex, artifacts, tallies, exactResults, hardware });
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
      {tab === "hardware" && <HardwareDashboard data={data} />}
      {tab === "performance" && <PerformanceDashboard data={data} />}
      {tab === "compiler" && <CompilerDashboard data={data} />}
      {tab === "convergence" && <ConvergenceDashboard data={data} />}
      {tab === "models" && <ModelExplorer data={data} onPreviewAsset={setPreviewAsset} />}
      {tab === "artifacts" && <ArtifactGallery artifacts={data.artifacts} onPreviewAsset={setPreviewAsset} />}
      {previewAsset && <AssetPreviewModal asset={previewAsset} onClose={() => setPreviewAsset(null)} />}
    </main>
  );
}

// ---------------------------------------------------------------------------
// Hero & narrative
// ---------------------------------------------------------------------------

function Hero({ data }: { data: SiteData }) {
  const { summary } = data;
  const peakRow = useMemo(() => {
    return data.performance.reduce<PerformanceRun | null>((best, run) => {
      if (run.bitsPerSecond === null) return best;
      if (!best || (best.bitsPerSecond ?? 0) < (run.bitsPerSecond ?? 0)) return run;
      return best;
    }, null);
  }, [data.performance]);

  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">Aralia Fault Tree Dataset · mcSCRAM</p>
        <h1>Benchmarks for a GPU Monte-Carlo PRA engine</h1>
        <p className="lede">
          mcSCRAM is a Monte-Carlo successor to the SCRAM probabilistic risk-assessment toolkit that compiles
          fault-tree models into a propagation DAG (<em>PDAG</em>) and samples them on GPUs via SYCL.
          This page distills <strong>{formatInteger(summary.counts.performanceRuns)}</strong>{" "}
          cross-device runs over <strong>{summary.counts.devices}</strong> devices,{" "}
          <strong>{formatInteger(summary.counts.compilerRows)}</strong> compiler ablation rows, and{" "}
          <strong>{formatInteger(summary.counts.traceFiles)}</strong> sampled convergence traces into one
          interactive archive — for {summary.counts.models} models, 43 of which are from the Aralia Fault Tree dataset.
        </p>
        <div className="hero-actions">
          <a href={summary.mcScramRepo}>mcSCRAM</a>
          <a href={summary.sourceRepo}>Archive</a>
          <a href={summary.doi}>Zenodo DOI</a>
        </div>
      </div>
      <div className="hero-card">
        <Metric
          label="Peak throughput"
          value={peakRow ? `${formatNumber(peakRow.bitsPerSecond, 3)} bits/s` : "n/a"}
          detail={peakRow ? `${peakRow.modelName} · ${peakRow.deviceShort}` : undefined}
        />
        <Metric
          label="Median GPU bits/s"
          value={formatNumber(data.devices.find((d) => d.deviceShort === "GTX 1660 S")?.bitsPerSecond.median ?? null, 3)}
          detail="across 973 GTX 1660 SUPER runs"
        />
        <Metric
          label="GPU vs CPU host"
          value={`${formatNumber(data.speedups.perDevice["GTX 1660 S"]?.median ?? null, 2)}x`}
          detail="matched-configuration median speedup"
        />
        <Metric
          label="Convergence exponent"
          value={formatNumber(data.convergence.exponentSummary.median, 2)}
          detail="median slope of log rel-err vs log trials"
        />
      </div>
    </header>
  );
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

function Overview({ data, onPreviewAsset }: { data: SiteData; onPreviewAsset: (asset: PlotAsset) => void }) {
  const orderedDevices = useMemo(() => orderDevices(data.devices), [data.devices]);
  const speedupsByDevice = useMemo(() => groupBy(data.speedups.rows, (row) => row.deviceShort), [data.speedups.rows]);
  const errorVsTrials = useMemo(
    () => buildErrorVsTrialsByDevice(data.performance),
    [data.performance],
  );

  return (
    <section className="section-grid">
      <div className="panel span-3 narrative-strip">
        {data.narrative.map((card) => (
          <article key={card.title} className="narrative-card">
            <p className="eyebrow">{card.detail}</p>
            <h3>{card.title}</h3>
            <strong className="narrative-metric">{card.metric}</strong>
            <p>{card.body}</p>
          </article>
        ))}
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Device leaderboard"
          title="Throughput per device — three orders of magnitude separate the stack"
          text="Each bar shows the full distribution of measured bits/s: the bar height is the median, the high cap is p95, and the marker on top is the single-run peak. Note the log y-axis."
        />
        <Plot
          data={[
            {
              type: "bar",
              x: orderedDevices.map((d) => d.deviceShort),
              y: orderedDevices.map((d) => d.bitsPerSecond.median ?? null),
              name: "median",
              marker: { color: orderedDevices.map((d) => DEVICE_COLOR[d.deviceShort] ?? inkPalette.ink) },
              error_y: {
                type: "data",
                symmetric: false,
                array: orderedDevices.map((d) => Math.max(0, (d.bitsPerSecond.p95 ?? 0) - (d.bitsPerSecond.median ?? 0))),
                arrayminus: orderedDevices.map((d) => Math.max(0, (d.bitsPerSecond.median ?? 0) - (d.bitsPerSecond.p05 ?? 0))),
                color: inkPalette.ink,
                thickness: 1.2,
              },
              hovertemplate: "%{x}<br>median=%{y:.3s}<br>p05–p95 errorbars<extra></extra>",
            },
            {
              type: "scatter",
              mode: "markers",
              x: orderedDevices.map((d) => d.deviceShort),
              y: orderedDevices.map((d) => d.bitsPerSecond.max ?? null),
              name: "peak",
              marker: { color: inkPalette.ink, size: 11, symbol: "diamond-open", line: { color: inkPalette.ink, width: 1 } },
              hovertemplate: "%{x} peak=%{y:.3s}<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { title: "device", tickangle: -15 },
            yaxis: { title: "bits per second", type: "log" },
            showlegend: true,
            legend: { y: 1.08, x: 0, xanchor: "left", orientation: "h" },
            margin: { l: 70, r: 24, t: 22, b: 96 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Matched speedup" title="Per-device speedup over the 8-core Core i7 10700 CPU host" />
        <Plot
          data={DEVICE_ORDER
            .filter((d) => d !== HOST_SHORT && speedupsByDevice[d]?.length)
            .map((dev) => ({
              type: "box",
              y: speedupsByDevice[dev]!.map((row) => row.speedupVsHost),
              name: dev,
              boxpoints: "outliers" as const,
              jitter: 0.4,
              marker: { color: DEVICE_COLOR[dev] ?? inkPalette.ink, size: 4, opacity: 0.6 },
              line: { color: DEVICE_COLOR[dev] ?? inkPalette.ink },
              fillcolor: "rgba(0,0,0,0)",
              hovertemplate: `${dev}<br>speedup=%{y:.2f}x<extra></extra>`,
            }))}
          layout={{
            yaxis: { title: "speedup vs 8-core Core i7 10700 CPU host", type: "log" },
            showlegend: false,
            margin: { l: 64, r: 16, t: 22, b: 80 },
          }}
        />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Scaling"
          title="bits/s vs PDAG nodes — almost perfectly inverse, on every device"
          text="A log-log fit gives the slope of the throughput-size scaling for each device. Slopes ≈ -1 mean throughput drops linearly with graph size — the GPU is fully utilized; latency tracks problem size and not vice versa."
        />
        <Plot
          data={orderedDevices.flatMap((dev) => {
            const subset = data.performance.filter((r) => r.deviceShort === dev.deviceShort && r.numNodes && r.bitsPerSecond);
            if (!subset.length) return [];
            const xs = subset.map((r) => r.numNodes!) as number[];
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            return [
              {
                type: "scattergl",
                mode: "markers",
                name: `${dev.deviceShort} (slope ${formatFitSlope(dev.throughputVsNodesFit)})`,
                x: subset.map((r) => r.numNodes),
                y: subset.map((r) => r.bitsPerSecond),
                text: subset.map((r) => `${r.modelName}<br>${r.deviceShort}<br>flag ${r.compileFlag}`),
                marker: { color: DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, size: 7, opacity: 0.72, line: { color: inkPalette.paper, width: 0.4 } },
                hovertemplate: "%{text}<br>nodes=%{x}<br>bits/s=%{y:.3s}<extra></extra>",
              } as Record<string, unknown>,
              fitLine(dev.throughputVsNodesFit, DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, `${dev.deviceShort} fit`, [xMin, xMax]),
            ].filter(Boolean) as Record<string, unknown>[];
          })}
          layout={{
            xaxis: { title: "PDAG nodes", type: "log" },
            yaxis: { title: "bits per second", type: "log" },
            legend: { y: -0.22, x: 0, xanchor: "left", orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 120 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Compiler" title="Compression by (flag, kn, xor) — median ± p05/p95" />
        <CompressionMatrixPlot summaries={data.compiler.summaries} />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Convergence"
          title="Relative error vs trials, by device"
          text="Each marker is a final-state run. The empirical Monte-Carlo rate is ≈ 1/√N (slope −0.5 on log-log). GPUs reach the same accuracy budget with the same trial count — they're not 'less accurate', they just sample faster."
        />
        <Plot
          data={Object.entries(errorVsTrials).map(([dev, series]) => ({
            type: "scattergl",
            mode: "markers",
            name: dev,
            x: series.x,
            y: series.y,
            text: series.text,
            marker: { color: DEVICE_COLOR[dev] ?? inkPalette.ink, size: 6, opacity: 0.7 },
            hovertemplate: "%{text}<br>trials=%{x:.3s}<br>rel err=%{y:.3e}<extra></extra>",
          }))}
          layout={{
            xaxis: { title: "total bits / iterations", type: "log" },
            yaxis: { title: "relative error", type: "log" },
            legend: { y: -0.22, x: 0, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 110 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Probability of dominant cut-set" title="Reference probabilities across the Aralia models" text="Reference (oracle / rare-event / MCUB) probabilities span twelve orders of magnitude. Rare-event tails are the hardest for Monte-Carlo." />
        <ReferenceProbabilityPlot models={data.models} />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Exact corner" title="BDD / CCF baselines" text="Where exact computation is tractable, mcSCRAM agrees with the SCRAM BDD baseline; these CCF test cases are used as sanity oracles." />
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

      <div className="panel span-3">
        <SectionTitle eyebrow="Artifacts" title="Pre-rendered plots and artifacts" text="Copies of the original MATLAB-generated figures. EPS originals and source scripts are linked in the Artifacts tab." />
        <div className="asset-strip">
          {data.artifacts.plots
            .filter((asset) => ["Throughput", "Compiler", "Device speedup", "Information theory", "Convergence", "Graph structure"].includes(asset.category))
            .slice(0, 8)
            .map((asset) => (
              <AssetCard key={asset.id} asset={asset} onPreview={onPreviewAsset} />
            ))}
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

function HardwareDashboard({ data }: { data: SiteData }) {
  const ordered = useMemo(() => orderHardware(data.hardware.devices), [data.hardware.devices]);
  const fleetSpan = useMemo(() => {
    const lanes = ordered.map((d) => d.lanes);
    const bandwidth = ordered.map((d) => d.bandwidthGBs);
    const peak = ordered.map((d) => d.measuredPeakBps ?? 0).filter((v) => v > 0);
    return {
      laneRatio: Math.max(...lanes) / Math.min(...lanes),
      bandwidthRatio: Math.max(...bandwidth) / Math.min(...bandwidth),
      peakRatio: Math.max(...peak) / Math.min(...peak),
      yearSpan: [Math.min(...ordered.map((d) => d.year)), Math.max(...ordered.map((d) => d.year))] as [number, number],
    };
  }, [ordered]);

  return (
    <section className="section-grid">
      <div className="panel span-3">
        <SectionTitle
          eyebrow="Fleet at a glance"
          title="Six devices across five vendor generations"
          text="mcSCRAM runs the same SYCL kernel against an Intel Comet Lake CPU host, an Intel Gen9 integrated GPU, a Turing consumer GPU, two Ada Lovelace cards (one consumer, one datacenter), and a Blackwell consumer flagship. The spread is roughly 3 orders of magnitude in compute lanes and a 40× span in memory bandwidth, which makes the fleet a clean instrument for separating compute-bound from memory-bound behaviour."
        />
        <div className="metric-grid">
          <Metric label="Devices" value={formatInteger(ordered.length)} detail={`${fleetSpan.yearSpan[0]} → ${fleetSpan.yearSpan[1]}`} />
          <Metric label="Compute-lane span" value={`${fleetSpan.laneRatio.toFixed(0)}×`} detail="UHD 630 → RTX 5090" />
          <Metric label="Bandwidth span" value={`${fleetSpan.bandwidthRatio.toFixed(0)}×`} detail="45.8 GB/s → 1.79 TB/s" />
          <Metric label="Measured peak span" value={`${fleetSpan.peakRatio.toFixed(0)}×`} detail="Core i7 vs GTX 1660 S" />
        </div>
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Specification ledger"
          title="Vendor-published specifications for the fleet"
          text="Values are normalized to the conventions in benchmarks/hardware-setup.md: cache totals in KiB (NVIDIA's 1024-byte convention), CUDA cores aggregated as SMs × cores/SM, resident-warp slots from the architectural cap (CC 7.5 → 32, CC 8.9 and 12.x → 48). Each row links to the original vendor specification page."
        />
        <HardwareSpecTable devices={ordered} />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Architecture roofline"
          title="Where does measured throughput come from?"
          text={`A six-point log-log regression of median bits/s against three hardware quantities. Bandwidth dominates: med ∝ BW^${formatFitSlope(data.hardware.bandwidthMedianFit)} (R² = ${formatNumber(data.hardware.bandwidthMedianFit.r2, 2)}). Lane scaling is sublinear (slope ${formatFitSlope(data.hardware.laneMedianFit)}, R² = ${formatNumber(data.hardware.laneMedianFit.r2, 2)}), so adding CUDA cores past the bandwidth wall does not buy throughput. We use the median rather than the peak because the GTX 1660 S has 23× more measured runs than the RTX 4090 — peak ordering would be sample-size-biased.`}
        />
        <div className="roofline-grid">
          <ScalingScatter
            devices={ordered}
            xField="bandwidthGBs"
            yField="measuredMedianBps"
            xLabel="vendor bandwidth (GB/s)"
            yLabel="median bits/s"
            fit={data.hardware.bandwidthMedianFit}
            title="vs bandwidth"
          />
          <ScalingScatter
            devices={ordered}
            xField="lanes"
            yField="measuredMedianBps"
            xLabel="compute lanes"
            yLabel="median bits/s"
            fit={data.hardware.laneMedianFit}
            title="vs lanes"
          />
          <ScalingScatter
            devices={ordered}
            xField="fp32TFlops"
            yField="measuredMedianBps"
            xLabel="FP32 TFLOP/s"
            yLabel="median bits/s"
            fit={data.hardware.flopsMedianFit}
            title="vs FP32 FLOPS"
          />
        </div>
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Compute hierarchy" title="Lane count per device" text="Log y-axis; the UHD 630 sits 100× below the consumer GPUs, the Blackwell flagship 100× above them again." />
        <Plot
          data={[
            {
              type: "bar",
              x: ordered.map((d) => d.deviceShort),
              y: ordered.map((d) => d.lanes),
              text: ordered.map((d) => d.lanesNote),
              marker: { color: ordered.map((d) => DEVICE_COLOR[d.deviceShort] ?? inkPalette.ink) },
              hovertemplate: "%{x}<br>%{y} %{customdata}<br>%{text}<extra></extra>",
              customdata: ordered.map((d) => d.lanesLabel),
            },
          ]}
          layout={{
            xaxis: { title: "device", tickangle: -15 },
            yaxis: { title: "lanes", type: "log" },
            margin: { l: 70, r: 16, t: 22, b: 80 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Memory hierarchy" title="L1 / L2 / global memory capacity" text="Aggregate per-device on a log axis. The Ada/Blackwell L2 cache (≥72 MiB) is the dominant new architectural lever: it holds entire PDAGs that previously spilled to VRAM." />
        <MemoryHierarchyPlot devices={ordered} />
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="Throughput efficiency"
          title="Bits/s per compute lane"
          text="Peak bits/s divided by the device's lane count. Smaller GPUs are dramatically more efficient per lane — they spend less of their fleet idling on memory."
        />
        <Plot
          data={[
            {
              type: "bar",
              x: ordered.map((d) => d.deviceShort),
              y: ordered.map((d) => d.bpsPerLane ?? 0),
              marker: { color: ordered.map((d) => DEVICE_COLOR[d.deviceShort] ?? inkPalette.ink) },
              hovertemplate: "%{x}<br>%{y:.3s} bits/s/lane<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { tickangle: -15 },
            yaxis: { title: "peak bits/s per lane", type: "log" },
            margin: { l: 70, r: 16, t: 22, b: 80 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="Bandwidth efficiency"
          title="Bits/s emitted per GB/s of available bandwidth"
          text="Throughput divided by vendor bandwidth — the workload's effective utilization ratio. Higher means the device is closer to its memory wall; lower means there is headroom that smaller PDAGs would consume."
        />
        <Plot
          data={[
            {
              type: "bar",
              x: ordered.map((d) => d.deviceShort),
              y: ordered.map((d) => d.bpsPerGBs ?? 0),
              marker: { color: ordered.map((d) => DEVICE_COLOR[d.deviceShort] ?? inkPalette.ink) },
              hovertemplate: "%{x}<br>%{y:.3s} bits/s / (GB/s)<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { tickangle: -15 },
            yaxis: { title: "peak bits/s ÷ vendor GB/s", type: "log" },
            margin: { l: 70, r: 16, t: 22, b: 80 },
          }}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Calculation checks"
          title="Independent cross-checks against vendor materials"
          text="Selected derivations from the hardware specification table — sanity checks for resident-warp counts, peak FP32, and memory bandwidth."
        />
        <CalcChecks devices={ordered} />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="SYCL terminology"
          title="Glossary: work-item, sub-group, warp, SM, EU"
          text="The vendor mapping for SYCL primitives is not symmetric across the fleet.  This glossary captures the mapping that benchmark numbers depend on."
        />
        <dl className="glossary">
          {data.hardware.glossary.map((entry) => (
            <div key={entry.term}>
              <dt>{entry.term}</dt>
              <dd>
                <p>{entry.summary}</p>
                <p className="glossary-mapping">{entry.mapping}</p>
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="SYCL ↔ hardware"
          title="What the runtime observes"
          text="At kernel launch mcSCRAM reads SYCL's device descriptors. These are the values that flowed into every benchmark row alongside the published spec."
        />
        <RuntimeObservedTable devices={ordered} />
      </div>

      <div className="panel span-3">
        <SectionTitle eyebrow="Normalization notes" title="How the specification table was assembled" />
        <ul className="footnote-list">
          {data.hardware.normalizationNotes.map((note, idx) => (
            <li key={idx}>{note}</li>
          ))}
        </ul>
        <p className="provenance">
          Source markdown: <a href={data.hardware.sourceUrl}>{data.hardware.source}</a>
        </p>
      </div>
    </section>
  );
}

function HardwareSpecTable({ devices }: { devices: HardwareDevice[] }) {
  return (
    <div className="table-wrap spec-table">
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>Architecture · year</th>
            <th>Compute</th>
            <th>Resident warps</th>
            <th>Clock (boost)</th>
            <th>Bandwidth</th>
            <th>L1 · L2</th>
            <th>Global memory</th>
            <th>FP32 peak</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((d) => (
            <tr key={d.deviceShort}>
              <td>
                <strong>{d.deviceShort}</strong>
                <span className="cell-sub">{d.deviceLong}</span>
                <a className="cell-link" href={d.sourceUrl}>vendor spec</a>
              </td>
              <td>
                {d.architecture}
                <span className="cell-sub">{d.cudaCompute ? `CC ${d.cudaCompute} · ` : ""}{d.process} · {d.year}</span>
              </td>
              <td>
                {formatInteger(d.lanes)} {d.lanesLabel}
                <span className="cell-sub">{d.computeUnits} {d.computeUnitsLabel}{d.smCount ? ` · ${d.smCount} SMs` : ""}</span>
              </td>
              <td>{d.residentWarps ? `${formatInteger(d.residentWarps)} (${d.smCount} × ${(d.residentWarps / (d.smCount ?? 1)).toFixed(0)})` : "—"}</td>
              <td>
                {formatNumber(d.boostClockMHz / 1000, 2)} GHz
                <span className="cell-sub">base {formatNumber(d.baseClockMHz / 1000, 2)} GHz</span>
              </td>
              <td>
                {formatNumber(d.bandwidthGBs, 3)} GB/s
                <span className="cell-sub">{formatNumber(d.memoryRateGbps, 2)} Gb/s × {d.memoryBusBits}-bit</span>
              </td>
              <td>
                {d.l1KiB ? `${formatInteger(d.l1KiB)} KiB` : "n/a"} · {d.l2KiB ? `${formatInteger(d.l2KiB)} KiB` : "n/a"}
                <span className="cell-sub">{d.l1Note}</span>
              </td>
              <td>
                {d.globalMemMiB >= 1024 ? `${(d.globalMemMiB / 1024).toFixed(0)} GiB` : `${d.globalMemMiB} MiB`}
                <span className="cell-sub">{d.memoryType}</span>
              </td>
              <td>
                {formatNumber(d.fp32TFlops, 3)} TFLOP/s
                <span className="cell-sub">{d.fp32Note}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MemoryHierarchyPlot({ devices }: { devices: HardwareDevice[] }) {
  const buildTrace = (key: "l1KiB" | "l2KiB" | "globalKiB", name: string, color: string) => ({
    type: "bar" as const,
    name,
    x: devices.map((d) => d.deviceShort),
    y: devices.map((d) => {
      if (key === "globalKiB") return d.globalMemMiB * 1024;
      return d[key] ?? null;
    }),
    marker: { color },
    hovertemplate: `%{x} ${name}=%{y:.3s} KiB<extra></extra>`,
  });
  return (
    <Plot
      data={[
        buildTrace("l1KiB", "L1 / shared", inkPalette.indigo),
        buildTrace("l2KiB", "L2", inkPalette.madder),
        buildTrace("globalKiB", "global", inkPalette.sage),
      ]}
      layout={{
        barmode: "group",
        xaxis: { tickangle: -15 },
        yaxis: { title: "capacity (KiB)", type: "log" },
        legend: { y: -0.22, orientation: "h" },
        margin: { l: 70, r: 16, t: 22, b: 100 },
      }}
    />
  );
}

function ScalingScatter({
  devices,
  xField,
  yField,
  xLabel,
  yLabel,
  fit,
  title,
}: {
  devices: HardwareDevice[];
  xField: keyof HardwareDevice;
  yField: keyof HardwareDevice;
  xLabel: string;
  yLabel: string;
  fit: LogFit;
  title?: string;
}) {
  const data = devices
    .map((d) => ({ device: d, x: d[xField] as number | null, y: d[yField] as number | null }))
    .filter((p): p is { device: HardwareDevice; x: number; y: number } => typeof p.x === "number" && typeof p.y === "number" && p.x > 0 && p.y > 0);
  const xMin = Math.min(...data.map((p) => p.x));
  const xMax = Math.max(...data.map((p) => p.x));
  const traces: Record<string, unknown>[] = [
    {
      type: "scatter",
      mode: "markers+text",
      x: data.map((p) => p.x),
      y: data.map((p) => p.y),
      text: data.map((p) => p.device.deviceShort),
      textposition: "top right",
      textfont: { family: "Inter Tight, sans-serif", size: 11, color: inkPalette.ink },
      marker: {
        color: data.map((p) => DEVICE_COLOR[p.device.deviceShort] ?? inkPalette.ink),
        size: 12,
        line: { color: inkPalette.ink, width: 0.6 },
      },
      hovertemplate: "%{text}<br>%{x:.3s}<br>%{y:.3s}<extra></extra>",
    },
  ];
  const line = fitLine(fit, inkPalette.ink, "log–log fit", [xMin, xMax]);
  if (line) traces.push(line);
  return (
    <figure className="roofline-figure">
      {title ? <figcaption>{title} · slope {formatFitSlope(fit)} · R² {formatNumber(fit.r2, 2)}</figcaption> : null}
      <Plot
        className="plot roofline-plot"
        data={traces}
        layout={{
          xaxis: { title: xLabel, type: "log" },
          yaxis: { title: yLabel, type: "log" },
          showlegend: false,
          margin: { l: 64, r: 24, t: 12, b: 50 },
        }}
      />
    </figure>
  );
}

function CalcChecks({ devices }: { devices: HardwareDevice[] }) {
  const checks: Array<{ device: string; quantity: string; derivation: string; result: string; check: string }> = [];
  for (const d of devices) {
    if (d.smCount && d.residentWarps) {
      const perSm = d.residentWarps / d.smCount;
      checks.push({
        device: d.deviceShort,
        quantity: "resident warps",
        derivation: `${d.smCount} SMs × ${perSm.toFixed(0)} warps/SM`,
        result: formatInteger(d.residentWarps),
        check: d.cudaCompute ? `CC ${d.cudaCompute} arch cap` : "—",
      });
    }
    if (d.smCount && d.l1KiB) {
      const perSm = d.l1KiB / d.smCount;
      checks.push({
        device: d.deviceShort,
        quantity: "L1 / shared total",
        derivation: `${d.smCount} SMs × ${perSm.toFixed(0)} KiB/SM`,
        result: `${formatInteger(d.l1KiB)} KiB`,
        check: "matches whitepaper",
      });
    }
    if (d.memoryRateGbps && d.memoryBusBits) {
      const bw = (d.memoryRateGbps * d.memoryBusBits) / 8;
      checks.push({
        device: d.deviceShort,
        quantity: "memory bandwidth",
        derivation: `${d.memoryRateGbps} Gb/s × ${d.memoryBusBits}-bit ÷ 8`,
        result: `${formatNumber(bw, 3)} GB/s`,
        check: `vendor lists ${formatNumber(d.bandwidthGBs, 3)} GB/s`,
      });
    }
    if (d.lanes && d.fp32TFlops && d.boostClockMHz) {
      const implied = (d.fp32TFlops * 1e12) / (2 * d.lanes);
      checks.push({
        device: d.deviceShort,
        quantity: "boost clock (FP32 implied)",
        derivation: `${formatNumber(d.fp32TFlops, 3)} TF ÷ (2 FLOP × ${formatInteger(d.lanes)} lanes)`,
        result: `${formatNumber(implied / 1e9, 3)} GHz`,
        check: `published ${formatNumber(d.boostClockMHz / 1000, 2)} GHz`,
      });
    }
  }
  return (
    <div className="table-wrap calc-table">
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>Quantity</th>
            <th>Derivation</th>
            <th>Computed</th>
            <th>Check</th>
          </tr>
        </thead>
        <tbody>
          {checks.map((row, idx) => (
            <tr key={`${row.device}-${idx}`}>
              <td>{row.device}</td>
              <td>{row.quantity}</td>
              <td className="mono-cell">{row.derivation}</td>
              <td className="mono-cell">{row.result}</td>
              <td>{row.check}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RuntimeObservedTable({ devices }: { devices: HardwareDevice[] }) {
  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>SYCL compute units</th>
            <th>SYCL clock (MHz)</th>
            <th>SYCL global mem</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((d) => {
            const o = d.observed ?? {};
            const gmem = typeof o.globalMemBytes === "number" ? formatBytes(o.globalMemBytes) : "—";
            return (
              <tr key={d.deviceShort}>
                <td>{d.deviceShort}</td>
                <td>
                  {formatInteger(o.maxComputeUnits ?? null)}
                  <span className="cell-sub">spec: {d.computeUnits} {d.computeUnitsLabel}</span>
                </td>
                <td>{formatInteger(o.maxClockMHz ?? null)}</td>
                <td>{gmem}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function orderHardware(devices: HardwareDevice[]): HardwareDevice[] {
  return [...devices].sort((a, b) => {
    const ai = DEVICE_ORDER.indexOf(a.deviceShort);
    const bi = DEVICE_ORDER.indexOf(b.deviceShort);
    if (ai === -1 && bi === -1) return a.deviceShort.localeCompare(b.deviceShort);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

function PerformanceDashboard({ data }: { data: SiteData }) {
  const [device, setDevice] = useState<string>("all");
  const [flag, setFlag] = useState<string>("all");
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () =>
      data.performance.filter((run) => {
        const matchesDevice = device === "all" || run.deviceShort === device;
        const matchesFlag = flag === "all" || run.compileFlag === flag;
        const q = query.toLowerCase();
        const matchesQuery = !q || run.modelName.toLowerCase().includes(q) || run.family.toLowerCase().includes(q);
        return matchesDevice && matchesFlag && matchesQuery;
      }),
    [data.performance, device, flag, query],
  );

  const orderedDevices = useMemo(() => orderDevices(data.devices), [data.devices]);
  const topRuns = useMemo(() => [...filtered].sort((a, b) => (b.bitsPerSecond ?? 0) - (a.bitsPerSecond ?? 0)).slice(0, 80), [filtered]);

  return (
    <section className="section-grid">
      <div className="panel span-3 controls">
        <Select label="Device" value={device} onChange={setDevice} options={["all", ...data.summary.facets.deviceShort]} />
        <Select label="Compile flag" value={flag} onChange={setFlag} options={["all", ...data.summary.facets.compileFlags]} />
        <label>
          Search
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="baobab, das, isp..." />
        </label>
        <Metric label="Runs shown" value={formatInteger(filtered.length)} />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Device leaderboard"
          title="Raw and matched per-device statistics"
          text="For each device, the median and p05/p95 are taken over every measured run; the matched-config speedup uses only (model, flag, kn, xor, δ, ccf) tuples that also ran on the 8-core CPU host."
        />
        <DeviceLeaderboard devices={orderedDevices} speedups={data.speedups} />
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Scaling" title="Throughput vs PDAG nodes (with log-log fit)" />
        <Plot
          data={orderedDevices.flatMap((dev) => {
            const subset = filtered.filter((r) => r.deviceShort === dev.deviceShort && r.numNodes && r.bitsPerSecond);
            if (!subset.length) return [];
            const xs = subset.map((r) => r.numNodes!) as number[];
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            return [
              {
                type: "scattergl",
                mode: "markers",
                name: `${dev.deviceShort} (n=${subset.length})`,
                x: subset.map((r) => r.numNodes),
                y: subset.map((r) => r.bitsPerSecond),
                text: subset.map((r) => `${r.modelName}<br>flag ${r.compileFlag} · set ${r.setId}`),
                marker: { color: DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, size: 7, opacity: 0.75 },
                hovertemplate: "%{text}<br>nodes=%{x}<br>bits/s=%{y:.3s}<extra></extra>",
              } as Record<string, unknown>,
              fitLine(dev.throughputVsNodesFit, DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, `${dev.deviceShort} ∝ N^${formatFitSlope(dev.throughputVsNodesFit)}`, [xMin, xMax]),
            ].filter(Boolean) as Record<string, unknown>[];
          })}
          layout={{
            xaxis: { title: "PDAG nodes", type: "log" },
            yaxis: { title: "bits per second", type: "log" },
            legend: { y: -0.22, x: 0, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 130 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Wall time" title="Time to convergence vs PDAG nodes" />
        <Plot
          data={orderedDevices.flatMap((dev) => {
            const subset = filtered.filter((r) => r.deviceShort === dev.deviceShort && r.numNodes && r.convergenceTimeMs);
            if (!subset.length) return [];
            return [
              {
                type: "scattergl",
                mode: "markers",
                name: dev.deviceShort,
                x: subset.map((r) => r.numNodes),
                y: subset.map((r) => r.convergenceTimeMs),
                text: subset.map((r) => `${r.modelName}<br>${r.deviceShort}`),
                marker: { color: DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, size: 6, opacity: 0.7 },
                hovertemplate: "%{text}<br>nodes=%{x}<br>time=%{y:.3s} ms<extra></extra>",
              },
            ];
          })}
          layout={{
            xaxis: { title: "PDAG nodes", type: "log" },
            yaxis: { title: "convergence time (ms)", type: "log" },
            showlegend: false,
            margin: { l: 70, r: 16, t: 22, b: 60 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="Information density"
          title="Iterations/s vs bits/iteration"
          text="Throughput = iterations/s · bits/iteration. GPUs trade iteration rate for huge bit-parallel batches; CPU host trades the other way."
        />
        <Plot
          data={orderedDevices.map((dev) => {
            const subset = filtered.filter((r) => r.deviceShort === dev.deviceShort && r.bitsPerIteration && r.iterationsPerSecond);
            return {
              type: "scattergl",
              mode: "markers",
              name: dev.deviceShort,
              x: subset.map((r) => r.bitsPerIteration),
              y: subset.map((r) => r.iterationsPerSecond),
              text: subset.map((r) => `${r.modelName}<br>${r.deviceShort}<br>flag ${r.compileFlag}`),
              marker: { color: DEVICE_COLOR[dev.deviceShort] ?? inkPalette.ink, size: 6, opacity: 0.7 },
              hovertemplate: "%{text}<br>bits/iter=%{x:.3s}<br>iter/s=%{y:.3s}<extra></extra>",
            };
          })}
          layout={{
            xaxis: { title: "bits per iteration", type: "log" },
            yaxis: { title: "iterations per second", type: "log" },
            legend: { y: -0.22, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 96 },
          }}
        />
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Distribution" title="Matched-configuration speedup distribution per device" />
        <Plot
          data={DEVICE_ORDER.filter((dev) => dev !== HOST_SHORT).map((dev) => {
            const rows = data.speedups.rows.filter((r) => r.deviceShort === dev);
            if (!rows.length) return null;
            return {
              type: "violin",
              y: rows.map((r) => r.speedupVsHost),
              name: dev,
              points: "all",
              jitter: 0.5,
              pointpos: 0,
              meanline: { visible: true, color: inkPalette.ink, width: 1 },
              marker: { color: DEVICE_COLOR[dev] ?? inkPalette.ink, size: 4, opacity: 0.6 },
              line: { color: DEVICE_COLOR[dev] ?? inkPalette.ink },
              fillcolor: "rgba(0,0,0,0)",
              hovertemplate: `${dev}<br>speedup=%{y:.2f}x<extra></extra>`,
            };
          }).filter(Boolean) as Record<string, unknown>[]}
          layout={{
            yaxis: { title: "speedup vs 8-core CPU host", type: "log" },
            showlegend: false,
            margin: { l: 64, r: 16, t: 22, b: 80 },
          }}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle eyebrow="Run detail" title={`${formatInteger(filtered.length)} filtered runs`} />
        <RunTable runs={topRuns} />
      </div>
    </section>
  );
}

function DeviceLeaderboard({ devices, speedups }: { devices: DeviceSummary[]; speedups: SpeedupData }) {
  return (
    <div className="table-wrap leaderboard">
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>Runs</th>
            <th>bits/s p05</th>
            <th>bits/s median</th>
            <th>bits/s p95</th>
            <th>bits/s peak</th>
            <th>iter/s median</th>
            <th>fit slope</th>
            <th>matched speedup (median)</th>
            <th>matched speedup (p95)</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((dev) => {
            const spd = speedups.perDevice[dev.deviceShort];
            return (
              <tr key={dev.deviceShort}>
                <td>
                  <strong>{dev.deviceShort}</strong>
                  <span className="cell-sub">{dev.deviceCategory}</span>
                </td>
                <td>{formatInteger(dev.count)}</td>
                <td>{formatNumber(dev.bitsPerSecond.p05, 3)}</td>
                <td>{formatNumber(dev.bitsPerSecond.median, 3)}</td>
                <td>{formatNumber(dev.bitsPerSecond.p95, 3)}</td>
                <td>{formatNumber(dev.bitsPerSecond.max, 3)}</td>
                <td>{formatNumber(dev.iterationsPerSecond.median, 3)}</td>
                <td>{formatFitSlope(dev.throughputVsNodesFit)}</td>
                <td>{spd ? `${formatNumber(spd.median, 2)}x` : "—"}</td>
                <td>{spd ? `${formatNumber(spd.p95, 2)}x` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

function CompilerDashboard({ data }: { data: SiteData }) {
  const visible = useMemo(
    () => data.compiler.summaries.filter((s) => ACTIVE_COMPILER_FLAGS.includes(s.compileFlag) && s.compressionAllNodes.count > 0),
    [data.compiler.summaries],
  );
  const [flag, setFlag] = useState<string>("all");
  const visibleRows = useMemo(() => {
    return data.compiler.rows.filter((row) => {
      if (!ACTIVE_COMPILER_FLAGS.includes(row.compileFlag)) return false;
      if (flag !== "all" && row.compileFlag !== flag) return false;
      return true;
    });
  }, [data.compiler.rows, flag]);
  const topCompression = useMemo(
    () =>
      [...visibleRows]
        .filter((row) => row.compressionAllNodes !== null)
        .sort((a, b) => (a.compressionAllNodes ?? 1) - (b.compressionAllNodes ?? 1))
        .slice(0, 35),
    [visibleRows],
  );
  const activeLevels = useMemo(
    () => data.compiler.levelStats.filter((s) => ACTIVE_COMPILER_FLAGS.includes(s.compileFlag)),
    [data.compiler.levelStats],
  );
  const activeTaxonomy = useMemo(
    () => data.compiler.taxonomy.filter((entry) => ACTIVE_COMPILER_FLAGS.includes(entry.id)),
    [data.compiler.taxonomy],
  );

  return (
    <section className="section-grid">
      <div className="panel span-3">
        <SectionTitle
          eyebrow="From fault tree to bit-parallel kernel"
          title="The PDAG is the contract between SCRAM's parser and mcSCRAM's GPU sampler"
          text="The SCRAM front end parses an MEF fault tree, normalises identifiers, and hands the result to the compiler as a propagation DAG (PDAG). Every gate becomes one device kernel and every basic event a bit-packed Boolean buffer; one Monte-Carlo iteration evaluates one column of bits across the entire PDAG. So compressing the PDAG is not merely a 'compile-time' nicety — bits/s ≈ (bits per iteration · iterations / s) drops nearly inversely with PDAG node count on every device we measured (Performance tab). The DirectEval preprocessor exposes one knob (compilation_level, the Cn flag) plus two switches (kn / xor) that toggle expansion of threshold and exclusive-or connectives."
        />
        <div className="metric-grid compiler-headline">
          <Metric
            label="Best compression"
            value={formatNumber(activeLevels.find((l) => l.compileFlag === "C2")?.pdagGates.median ?? null, 0)}
            detail={`median gates @ C2 (was ${formatNumber(activeLevels.find((l) => l.compileFlag === "C0")?.pdagGates.median ?? null, 0)} @ C0)`}
          />
          <Metric
            label="C2 preprocessing cost"
            value={`${formatNumber(activeLevels.find((l) => l.compileFlag === "C2")?.preprocessingTimeUs.median ?? null, 0)} µs`}
            detail="median per model on the C2 default"
          />
          <Metric
            label="Cost of full pipeline"
            value={`${formatNumber(activeLevels.find((l) => l.compileFlag === "C8")?.preprocessingTimeUs.median ?? null, 0)} µs`}
            detail="median preprocessing @ C8"
          />
          <Metric
            label="Models with k-of-N"
            value={`${data.compiler.connectiveEffect.kn.modelCount} of ${formatInteger(data.summary.counts.models)}`}
            detail={`${data.compiler.connectiveEffect.xor.modelCount} contain XOR`}
          />
        </div>
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Pipeline"
          title="Where compilation sits in the Monte-Carlo pipeline"
          text="One forward sweep of mcSCRAM. Solid stages always run; the three optional phases (III–V) are gated on compile level and on the kn/xor expansion switches."
        />
        <PipelineFlow />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="DirectEval level taxonomy"
          title="What each Cn flag adds, in pipeline order"
          text="Cumulative levels: Cn runs every pass in C0…Cn−1 then the row's additions. The 'mcSCRAM impact' column scopes the textbook KC rationale to our bit-parallel sampler. Median PDAG size and preprocessing time below come from the 172-row ablation slice — same 43 models across every flag."
        />
        <LevelTaxonomyTable taxonomy={activeTaxonomy} levelStats={activeLevels} />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Cost vs value"
          title="PDAG size and compile time as the level dial turns"
          text="Two-axis log plot: median PDAG nodes (left) and median preprocessing time (right) across the 43-model ablation. C2 is the throughput sweet spot; C3–C4 spend time without shrinking; C8 trades 6× more compile time for a slightly flatter graph."
        />
        <LevelCostValuePlot levels={activeLevels} />
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="Median gate budget per level"
          title="How many AND / OR / k-of-N / XOR survive at each level"
          text="Stacked share of the median model's PDAG after each level. Phase III expansion at C6+ would convert k-of-N → AND/OR, but C5/C6/C7 are not in the production archive (gaps in the bar)."
        />
        <LevelCompositionPlot levels={activeLevels} />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Connective expansion"
          title="kn and xor are not boolean flags on compression — they pick which rewrite to apply"
          text="The CSV's req_no_kn / req_no_xor are the inverses of the compiler's expand_atleast_gates / expand_xor_gates settings. When expansion is enabled, Phase III (or its targeted variants at lower levels) rewrites the connective into its AND/OR form using the rules below. At C6 with either flag = true the compiler is committed to a fully AND/OR PDAG via Phase III."
        />
        <ConnectiveSwitchesPanel switches={data.compiler.connectiveSwitches} />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Connective effect — focused subset"
          title="What the kn / xor switches actually do, on the models that contain them"
          text="The aggregate matrix is dominated by AND/OR-only models that ignore both switches. Here we restrict to the model subset that actually contains the relevant connective in its MEF source: 7 Aralia trees have k-of-N gates and 1 has XOR. Lower bars = smaller PDAG."
        />
        <ConnectiveEffectPanel kn={data.compiler.connectiveEffect.kn} xor={data.compiler.connectiveEffect.xor} />
      </div>

      <div className="panel span-3 controls">
        <Select label="Compile flag" value={flag} onChange={setFlag} options={["all", ...ACTIVE_COMPILER_FLAGS]} />
        <Metric label="Rows" value={formatInteger(visibleRows.length)} />
        <Metric label="Median all-node factor" value={formatNumber(medianOf(visibleRows.map((row) => row.compressionAllNodes)), 3)} />
        <Metric label="Median preprocessing" value={`${formatNumber(medianOf(visibleRows.map((row) => row.preprocessingTimeUs)), 3)} µs`} />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Ablation matrix"
          title="Compression factor by compile flag × (kn, xor)"
          text="The headline measurement: every grouped bar collapses 43 models. compression_factor = |PDAG nodes| / |source-model nodes|; values < 1 mean the compiler shrunk the graph, > 1 mean it expanded helper gates."
        />
        <CompressionMatrixPlot summaries={visible} />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Structure → reduction"
          title="Fan-in vs compression factor (per flag)"
          text="High average fan-in is correlated with stronger compression — wide gates accept more common-argument factoring and associative coalescing. The slope is reported per level."
        />
        <Plot
          data={ACTIVE_COMPILER_FLAGS.flatMap((flagId) => {
            const rows = visibleRows.filter((r) => r.compileFlag === flagId && r.pdagAvgFanIn && r.compressionAllNodes);
            if (!rows.length) return [];
            const fit = visible.find((s) => s.compileFlag === flagId && !s.noKn && !s.noXor)?.fanInVsCompressionFit;
            return [
              {
                type: "scattergl",
                mode: "markers",
                name: `${flagId} (slope ${formatFitSlope(fit)})`,
                x: rows.map((r) => r.pdagAvgFanIn),
                y: rows.map((r) => r.compressionAllNodes),
                text: rows.map((r) => `${r.modelName}<br>flag ${r.compileFlag} · kn:${r.noKn ? "off" : "on"} xor:${r.noXor ? "off" : "on"}`),
                marker: { color: FLAG_COLOR[flagId] ?? inkPalette.ink, size: 6, opacity: 0.7 },
                hovertemplate: "%{text}<br>fan-in=%{x:.2f}<br>compression=%{y:.3f}<extra></extra>",
              },
            ];
          })}
          layout={{
            xaxis: { title: "PDAG average gate fan-in" },
            yaxis: { title: "compression factor (all nodes)" },
            legend: { y: -0.22, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 100 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle
          eyebrow="Gate inventory"
          title="Total PDAG gates by type"
          text="Aralia models are dominated by AND/OR; only a small handful have k-of-N or XOR gates after preprocessing. Log y-axis surfaces the long tail."
        />
        <Plot
          data={(["and", "or", "atleast", "xor"] as const).map((kind, idx) => ({
            type: "bar",
            name: ({ and: "AND", or: "OR", atleast: "k-of-N", xor: "XOR" } as Record<string, string>)[kind],
            x: ACTIVE_COMPILER_FLAGS,
            y: ACTIVE_COMPILER_FLAGS.map((flagId) => {
              const sum = visible
                .filter((s) => s.compileFlag === flagId)
                .reduce((acc, s) => acc + (s[`${kind}Sum` as keyof CompilerSummary] as number ?? 0), 0);
              return sum > 0 ? sum : null;
            }),
            marker: { color: [inkPalette.indigoDeep, inkPalette.indigo, inkPalette.madder, inkPalette.ochre][idx] },
            hovertemplate: "%{x} %{fullData.name}=%{y}<extra></extra>",
          }))}
          layout={{
            barmode: "group",
            xaxis: { title: "compile flag" },
            yaxis: { title: "PDAG gates summed across runs", type: "log" },
            legend: { y: -0.22, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 100 },
          }}
        />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Compiler cost"
          title="Preprocessing time vs model gates (per flag)"
          text="Log-log fit slope close to 1 indicates near-linear preprocessing in model size — Phase II is O(N log N) in practice on these models. Higher flags shift the line up; they do not steepen it."
        />
        <Plot
          data={ACTIVE_COMPILER_FLAGS.flatMap((flagId) => {
            const rows = visibleRows.filter((r) => r.compileFlag === flagId && r.modelGates && r.preprocessingTimeUs);
            if (!rows.length) return [];
            const fit = visible.find((s) => s.compileFlag === flagId && !s.noKn && !s.noXor)?.timeVsGatesFit;
            return [
              {
                type: "scattergl",
                mode: "markers",
                name: `${flagId} ∝ gates^${formatFitSlope(fit)}`,
                x: rows.map((r) => r.modelGates),
                y: rows.map((r) => r.preprocessingTimeUs),
                text: rows.map((r) => `${r.modelName}<br>flag ${r.compileFlag}`),
                marker: { color: FLAG_COLOR[flagId] ?? inkPalette.ink, size: 6, opacity: 0.65 },
                hovertemplate: "%{text}<br>gates=%{x}<br>time=%{y:.3s} µs<extra></extra>",
              },
            ];
          })}
          layout={{
            xaxis: { title: "model gates", type: "log" },
            yaxis: { title: "preprocessing time (µs)", type: "log" },
            legend: { y: -0.22, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 130 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Top reductions" title="Most aggressive compressions in the archive" text="The smallest compression_factor rows — these are the models that respond best to the compiler. The kn / xor columns identify whether the saving came from connective expansion." />
        <CompactTable
          headers={["Model", "Flag", "kn", "xor", "Factor"]}
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

// ---------------------------------------------------------------------------
// Compiler sub-components
// ---------------------------------------------------------------------------

const PIPELINE_STAGES: Array<{
  kind: "source" | "compile-stage" | "compile-optional" | "runtime";
  title: string;
  subtitle: string;
  detail?: string;
}> = [
  { kind: "source", title: "MEF XML", subtitle: "fault-tree source", detail: "SCRAM parser" },
  { kind: "source", title: "PDAG build", subtitle: "raw connectives + signed edges", detail: "construction" },
  { kind: "compile-stage", title: "Phase I", subtitle: "cleanup", detail: "NOT/NAND/NOR → signed edges; remove null/constants" },
  { kind: "compile-stage", title: "Phase II", subtitle: "structural compression", detail: "CSE · modules · coalesce · common-args · distributivity · decomposition" },
  { kind: "compile-optional", title: "Phase III", subtitle: "AND/OR normalize", detail: "XOR & k-of-N expansion (gated by kn / xor)" },
  { kind: "compile-optional", title: "Phase IV", subtitle: "complement propagate", detail: "De Morgan push toward leaves" },
  { kind: "compile-optional", title: "Phase V", subtitle: "aggressive coalesce", detail: "flatten ∧/∨ across modules" },
  { kind: "runtime", title: "Device blocks", subtitle: "SYCL kernel layers", detail: "1 layer = 1 kernel launch" },
  { kind: "runtime", title: "Bit-parallel trials", subtitle: "Monte-Carlo sampling", detail: "trial-bits per iteration" },
];

function PipelineFlow() {
  return (
    <div className="pipeline-flow" role="list">
      {PIPELINE_STAGES.map((stage, idx) => (
        <Fragment key={stage.title}>
          <article className={classNames("pipeline-stage", `pipeline-stage--${stage.kind}`)} role="listitem">
            <strong>{stage.title}</strong>
            <span>{stage.subtitle}</span>
            {stage.detail ? <small>{stage.detail}</small> : null}
          </article>
          {idx < PIPELINE_STAGES.length - 1 ? (
            <span className="pipeline-arrow" aria-hidden="true">›</span>
          ) : null}
        </Fragment>
      ))}
    </div>
  );
}

function LevelTaxonomyTable({
  taxonomy,
  levelStats,
}: {
  taxonomy: CompileLevel[];
  levelStats: CompileLevelStats[];
}) {
  const statsById = useMemo(() => {
    const map = new Map<string, CompileLevelStats>();
    for (const entry of levelStats) map.set(entry.compileFlag, entry);
    return map;
  }, [levelStats]);
  const baseline = statsById.get("C0");
  const baselineGates = baseline?.pdagGates.median ?? null;
  return (
    <div className="table-wrap level-taxonomy">
      <table>
        <thead>
          <tr>
            <th>Cn</th>
            <th>Stage</th>
            <th>Adds passes (incremental)</th>
            <th>mcSCRAM impact</th>
            <th>Median PDAG gates</th>
            <th>Δ vs C0</th>
            <th>Median compile time</th>
          </tr>
        </thead>
        <tbody>
          {taxonomy.map((entry) => {
            const stats = statsById.get(entry.id);
            const gates = stats?.pdagGates.median ?? null;
            const deltaPct =
              gates !== null && baselineGates && baselineGates > 0
                ? ((gates - baselineGates) / baselineGates) * 100
                : null;
            return (
              <tr key={entry.id}>
                <td>
                  <strong>{entry.id}</strong>
                  <span className="cell-sub">level {entry.level} · {entry.name}</span>
                </td>
                <td>
                  <span className={`stage-tag stage-tag--${entry.stage}`}>{entry.stage}</span>
                </td>
                <td className="pass-cell">
                  {entry.passes.length ? (
                    <ul>
                      {entry.passes.map((pass) => (
                        <li key={pass.name}>
                          <code>{pass.name}</code>
                          <span>{pass.what}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <em className="cell-sub">no preprocessing passes</em>
                  )}
                  {entry.rewrites?.length ? (
                    <div className="rewrite-list">
                      {entry.rewrites.map((rw) => (
                        <div key={rw.from} className="rewrite">
                          <code>{rw.from}</code>
                          <span className="rewrite-arrow">→</span>
                          <code>{rw.to}</code>
                          <small>{rw.note}</small>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </td>
                <td className="impact-cell">{entry.mcImpact}</td>
                <td className="mono-cell">{stats ? formatNumber(gates, 0) : "—"}</td>
                <td className="mono-cell">
                  {deltaPct === null ? "—" : `${deltaPct > 0 ? "+" : ""}${deltaPct.toFixed(1)}%`}
                </td>
                <td className="mono-cell">{stats ? `${formatNumber(stats.preprocessingTimeUs.median, 0)} µs` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LevelCostValuePlot({ levels }: { levels: CompileLevelStats[] }) {
  const ordered = [...levels].sort((a, b) => (a.level ?? 0) - (b.level ?? 0));
  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "lines+markers",
          name: "median PDAG gates",
          x: ordered.map((l) => l.compileFlag),
          y: ordered.map((l) => l.pdagGates.median ?? null),
          line: { color: inkPalette.indigoDeep, width: 2 },
          marker: { color: inkPalette.indigoDeep, size: 10 },
          yaxis: "y",
          hovertemplate: "%{x}<br>median %{y:.0f} PDAG gates<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "median preprocessing (µs)",
          x: ordered.map((l) => l.compileFlag),
          y: ordered.map((l) => l.preprocessingTimeUs.median ?? null),
          line: { color: inkPalette.madder, width: 2, dash: "dash" },
          marker: { color: inkPalette.madder, size: 10, symbol: "diamond" },
          yaxis: "y2",
          hovertemplate: "%{x}<br>median %{y:.0f} µs preprocessing<extra></extra>",
        },
      ]}
      layout={{
        xaxis: { title: "compile level" },
        yaxis: { title: "median PDAG gates", type: "log", side: "left" },
        yaxis2: {
          title: "median preprocessing (µs)",
          type: "log",
          side: "right",
          overlaying: "y",
          showgrid: false,
        },
        legend: { y: -0.22, orientation: "h" },
        margin: { l: 70, r: 70, t: 22, b: 90 },
      }}
    />
  );
}

function LevelCompositionPlot({ levels }: { levels: CompileLevelStats[] }) {
  const ordered = [...levels].sort((a, b) => (a.level ?? 0) - (b.level ?? 0));
  const types = [
    { key: "and" as const, label: "AND", color: inkPalette.indigoDeep },
    { key: "or" as const, label: "OR", color: inkPalette.indigo },
    { key: "atleast" as const, label: "k-of-N", color: inkPalette.madder },
    { key: "xor" as const, label: "XOR", color: inkPalette.ochre },
  ];
  return (
    <Plot
      data={types.map((t) => ({
        type: "bar",
        name: t.label,
        x: ordered.map((l) => l.compileFlag),
        y: ordered.map((l) => l.compositionShare[t.key] ?? 0),
        marker: { color: t.color },
        hovertemplate: `%{x} ${t.label}=%{y:.1%}<extra></extra>`,
      }))}
      layout={{
        barmode: "stack",
        xaxis: { title: "compile level" },
        yaxis: { title: "PDAG gate-type share", tickformat: ".0%" },
        legend: { y: -0.22, orientation: "h" },
        margin: { l: 70, r: 16, t: 22, b: 90 },
      }}
    />
  );
}

function ConnectiveSwitchesPanel({ switches }: { switches: ConnectiveSwitch[] }) {
  return (
    <div className="connective-switches">
      {switches.map((sw) => (
        <article key={sw.id} className="connective-switch">
          <header>
            <p className="eyebrow">{sw.cliFlag} · expand when {sw.expandsWhen}</p>
            <h3>{sw.title}</h3>
            <p>{sw.subtitle}</p>
          </header>
          <div className="formula">
            <span className="formula-label">rewrite</span>
            <code>{sw.formula}</code>
          </div>
          <dl>
            <div>
              <dt>bases</dt>
              <dd>{sw.bases}</dd>
            </div>
            <div>
              <dt>graph effect</dt>
              <dd>{sw.growth}</dd>
            </div>
            <div>
              <dt>mcSCRAM impact</dt>
              <dd>{sw.mcImpact}</dd>
            </div>
          </dl>
        </article>
      ))}
    </div>
  );
}

function ConnectiveEffectPanel({
  kn,
  xor,
}: {
  kn: ConnectiveEffectGroup;
  xor: ConnectiveEffectGroup;
}) {
  return (
    <div className="connective-effect-grid">
      <ConnectiveEffectChart label="k-of-N (req_no_kn)" group={kn} />
      <ConnectiveEffectChart label="XOR (req_no_xor)" group={xor} />
    </div>
  );
}

function ConnectiveEffectChart({
  label,
  group,
}: {
  label: string;
  group: ConnectiveEffectGroup;
}) {
  const ordered = useMemo(
    () => group.levels.filter((entry) => entry.expandOn.pdagGates.median || entry.expandOff.pdagGates.median),
    [group.levels],
  );
  if (!ordered.length || !group.modelCount) {
    return (
      <div className="connective-effect-card">
        <header>
          <p className="eyebrow">{label}</p>
          <h3>no eligible models</h3>
        </header>
        <EmptyState text="The Aralia corpus has no models with this connective type." />
      </div>
    );
  }
  return (
    <div className="connective-effect-card">
      <header>
        <p className="eyebrow">{label}</p>
        <h3>{group.modelCount} model{group.modelCount === 1 ? "" : "s"} · {formatInteger(group.rowCount)} rows</h3>
        <p className="cell-sub">{group.models.join(" · ")}</p>
      </header>
      <Plot
        className="plot connective-plot"
        data={[
          {
            type: "bar",
            name: "expand on",
            x: ordered.map((entry) => entry.compileFlag),
            y: ordered.map((entry) => entry.expandOn.pdagGates.median ?? null),
            marker: { color: inkPalette.indigoDeep },
            hovertemplate: "%{x} expand on · median %{y:.0f} gates<extra></extra>",
          },
          {
            type: "bar",
            name: "expand off",
            x: ordered.map((entry) => entry.compileFlag),
            y: ordered.map((entry) => entry.expandOff.pdagGates.median ?? null),
            marker: { color: inkPalette.madder },
            hovertemplate: "%{x} expand off · median %{y:.0f} gates<extra></extra>",
          },
        ]}
        layout={{
          barmode: "group",
          xaxis: { title: "compile level" },
          yaxis: { title: "median PDAG gates" },
          legend: { y: -0.25, orientation: "h" },
          margin: { l: 70, r: 16, t: 14, b: 80 },
        }}
      />
      <table className="connective-ratio-table">
        <thead>
          <tr>
            <th>Level</th>
            <th>on / off ratio</th>
            <th>median Δ</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((entry) => {
            const onMed = entry.expandOn.pdagGates.median ?? null;
            const offMed = entry.expandOff.pdagGates.median ?? null;
            const ratio = entry.pdagRatioOnVsOff;
            const delta = onMed !== null && offMed !== null ? onMed - offMed : null;
            return (
              <tr key={entry.compileFlag}>
                <td>{entry.compileFlag}</td>
                <td className="mono-cell">{ratio === null ? "—" : ratio.toFixed(2)}</td>
                <td className="mono-cell">{delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(0)}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CompressionMatrixPlot({ summaries }: { summaries: CompilerSummary[] }) {
  const flags = Array.from(new Set(summaries.map((s) => s.compileFlag))).sort();
  const variants: Array<{ label: string; noKn: boolean; noXor: boolean; color: string }> = [
    { label: "kn:on  xor:on",  noKn: false, noXor: false, color: inkPalette.indigoDeep },
    { label: "kn:on  xor:off", noKn: false, noXor: true,  color: inkPalette.indigo },
    { label: "kn:off xor:on",  noKn: true,  noXor: false, color: inkPalette.madder },
    { label: "kn:off xor:off", noKn: true,  noXor: true,  color: inkPalette.ochre },
  ];
  const traces = variants.map((variant) => {
    const xs: string[] = [];
    const ys: Array<number | null> = [];
    const errPlus: number[] = [];
    const errMinus: number[] = [];
    for (const flag of flags) {
      const s = summaries.find((row) => row.compileFlag === flag && row.noKn === variant.noKn && row.noXor === variant.noXor);
      xs.push(flag);
      ys.push(s?.compressionAllNodes.median ?? null);
      errPlus.push(s ? Math.max(0, (s.compressionAllNodes.p95 ?? 0) - (s.compressionAllNodes.median ?? 0)) : 0);
      errMinus.push(s ? Math.max(0, (s.compressionAllNodes.median ?? 0) - (s.compressionAllNodes.p05 ?? 0)) : 0);
    }
    return {
      type: "bar",
      name: variant.label,
      x: xs,
      y: ys,
      marker: { color: variant.color },
      error_y: { type: "data", symmetric: false, array: errPlus, arrayminus: errMinus, color: inkPalette.ink, thickness: 1 },
      hovertemplate: `${variant.label}<br>%{x} median=%{y:.3f}<extra></extra>`,
    };
  });
  return (
    <Plot
      data={traces}
      layout={{
        barmode: "group",
        xaxis: { title: "compile flag", tickangle: 0 },
        yaxis: { title: "compression factor (all nodes)" },
        legend: { y: -0.22, orientation: "h" },
        margin: { l: 70, r: 16, t: 22, b: 110 },
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------

function ConvergenceDashboard({ data }: { data: SiteData }) {
  const [exemplarModels, setExemplarModels] = useState<string[]>(() => {
    const deep = data.convergence.deepTraceModels ?? [];
    const families = new Set<string>();
    const picked: string[] = [];
    for (const entry of deep) {
      if (families.has(entry.family)) continue;
      families.add(entry.family);
      picked.push(entry.modelId);
      if (picked.length >= 6) break;
    }
    if (picked.length < 6) {
      for (const entry of deep) {
        if (picked.includes(entry.modelId)) continue;
        picked.push(entry.modelId);
        if (picked.length >= 6) break;
      }
    }
    return picked;
  });
  const [tracesByModel, setTracesByModel] = useState<Record<string, TraceSeries[]>>({});

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      exemplarModels.map((modelId) =>
        loadJson<TraceModelData>(`traces/${modelId}.json`)
          .then((payload) => [modelId, payload.series] as const)
          .catch(() => [modelId, [] as TraceSeries[]] as const),
      ),
    ).then((entries) => {
      if (cancelled) return;
      const next: Record<string, TraceSeries[]> = {};
      for (const [modelId, series] of entries) {
        next[modelId] = series;
      }
      setTracesByModel(next);
    });
    return () => {
      cancelled = true;
    };
  }, [exemplarModels]);

  const exponentSamples = data.convergence.exponentSamples;
  const trialsByTarget = data.convergence.trialsToTargetSamples;

  const filteredExponents = useMemo(
    () => exponentSamples.filter((value) => Number.isFinite(value) && value < 0.4 && value > -1.5),
    [exponentSamples],
  );
  const exponentPeakCount = useMemo(() => {
    const buckets = new Map<number, number>();
    for (const value of filteredExponents) {
      const key = Math.round(value / 0.05);
      buckets.set(key, (buckets.get(key) ?? 0) + 1);
    }
    let peak = 0;
    for (const value of buckets.values()) if (value > peak) peak = value;
    return Math.max(peak, 10);
  }, [filteredExponents]);

  return (
    <section className="section-grid">
      <div className="panel span-3">
        <SectionTitle
          eyebrow="Distribution"
          title="Empirical convergence-rate exponent across 700+ traces"
          text="Slope of log10(rel_error) vs log10(trials) for each trace.  The peak sits near −0.5 (the theoretical Monte-Carlo rate); the long tail toward 0 is dominated by traces that hit their stop-on-convergence target early and so plateau."
        />
        <Plot
          data={[
            {
              type: "histogram",
              x: filteredExponents,
              xbins: { start: -1.4, end: 0.4, size: 0.05 },
              marker: { color: inkPalette.indigo, line: { color: inkPalette.ink, width: 0.5 } },
              hovertemplate: "slope=%{x:.2f}<br>count=%{y}<extra></extra>",
            },
            {
              type: "scatter",
              mode: "lines",
              x: [-0.5, -0.5],
              y: [0, exponentPeakCount],
              line: { color: inkPalette.madder, dash: "dot", width: 2 },
              name: "−1/2 (Monte-Carlo theory)",
              showlegend: true,
              hoverinfo: "skip",
              yaxis: "y",
            },
          ]}
          layout={{
            xaxis: { title: "convergence-rate exponent" },
            yaxis: { title: "trace count" },
            legend: { y: -0.22, orientation: "h" },
            margin: { l: 70, r: 16, t: 22, b: 100 },
          }}
        />
      </div>

      <div className="panel span-2">
        <SectionTitle
          eyebrow="Trial budget"
          title="How many trials does it take to hit ε targets?"
          text="Box plot of trials required to first reach a given relative error (rel_error or epsilon proxy).  Log scale: each decade of accuracy costs roughly an order of magnitude more samples — the 1/N² Monte-Carlo cost."
        />
        <Plot
          data={Object.entries(trialsByTarget).map(([key, values]) => ({
            type: "box",
            y: values,
            name: `ε ≤ 1e−${key.replace("trialsTo", "")}`,
            boxpoints: "outliers" as const,
            jitter: 0.4,
            marker: { color: inkPalette.indigoDeep, size: 4, opacity: 0.6 },
            line: { color: inkPalette.ink },
            fillcolor: "rgba(0,0,0,0)",
            hovertemplate: `${key}<br>trials=%{y:.3s}<extra></extra>`,
          }))}
          layout={{
            yaxis: { title: "trials to threshold", type: "log" },
            showlegend: false,
            margin: { l: 70, r: 16, t: 22, b: 60 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Information yield" title="info_bits_total at trace endpoint" text="Total information-theoretic bits accumulated by the end of each trace. Heavier-tailed models accumulate more rare-event information before convergence." />
        <Plot
          data={[
            {
              type: "histogram",
              x: data.traces.index.map((t) => t.infoBitsTotal).filter((value): value is number => typeof value === "number"),
              marker: { color: inkPalette.sage, line: { color: inkPalette.ink, width: 0.5 } },
              hovertemplate: "info=%{x:.2f} bits<br>count=%{y}<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { title: "total information bits (final iteration)" },
            yaxis: { title: "trace count" },
            margin: { l: 70, r: 16, t: 22, b: 60 },
          }}
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow="Exemplar traces"
          title="rel_error vs trials for canonical models"
          text="A small set of representative Aralia models. Each thumbnail overlays every trace measured for that model; the dashed indigo line is the −1/2 reference slope from a single matched anchor point."
        />
        <ExemplarSelector data={data} value={exemplarModels} onChange={setExemplarModels} />
        <div className="trace-multiples">
          {exemplarModels.map((modelId) => (
            <TraceThumbnail key={modelId} modelId={modelId} traces={tracesByModel[modelId] ?? []} />
          ))}
        </div>
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Run sets" title="Replicate batches in the archive" text="Each row is one set_*/ directory in benchmarks/runs/. Sets sweep target accuracy, time budget, and CCF on/off." />
        <CompactTable
          headers={["Set", "Target ε", "Conf", "Timeout"]}
          rows={Object.values(data.setIndex)
            .sort((a, b) => a.setId.localeCompare(b.setId))
            .map((entry) => [
              entry.setId,
              formatNumber(entry.delta, 3) ?? "—",
              formatNumber(entry.confidence, 3) ?? "—",
              entry.timeoutSec && entry.timeoutSec > 0 ? `${entry.timeoutSec}s` : "—",
            ])}
        />
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Monitored tallies" title="Marginal event probabilities (300 s on RTX 4090)" text="Histogram of log10(mean) for all 15,143 monitored marginal tallies. The double peak around −5 and 0 separates rare-event marginals from the dominant cut-set events." />
        <Plot
          data={[
            {
              type: "bar",
              x: data.tallies.logMeanHistogram.map((bucket) => `1e${bucket.bucket}`),
              y: data.tallies.logMeanHistogram.map((bucket) => bucket.count),
              marker: { color: inkPalette.indigo, line: { color: inkPalette.ink, width: 0.5 } },
              hovertemplate: "bucket=%{x}<br>count=%{y}<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { title: "log10(mean) bucket" },
            yaxis: { title: "monitored events", type: "log" },
            margin: { l: 70, r: 16, t: 22, b: 60 },
          }}
        />
      </div>
    </section>
  );
}

function ExemplarSelector({
  data,
  value,
  onChange,
}: {
  data: SiteData;
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const candidates = useMemo(() => {
    const deep = data.convergence.deepTraceModels ?? [];
    return deep.slice(0, 24);
  }, [data.convergence.deepTraceModels]);
  const valueSet = new Set(value);
  return (
    <div className="chip-row">
      {candidates.map((entry) => {
        const active = valueSet.has(entry.modelId);
        return (
          <button
            key={entry.modelId}
            type="button"
            className={classNames("chip", active && "active")}
            onClick={() => {
              if (active) onChange(value.filter((v) => v !== entry.modelId));
              else if (value.length < 8) onChange([...value, entry.modelId]);
            }}
            title={`${formatInteger(entry.points)} sampled trace points`}
          >
            {entry.modelId}
          </button>
        );
      })}
    </div>
  );
}

function TraceThumbnail({ modelId, traces }: { modelId: string; traces: TraceSeries[] }) {
  if (!traces.length) {
    return (
      <article className="trace-thumb empty">
        <header>{modelId}</header>
        <div>no trace</div>
      </article>
    );
  }
  const sortedTraces = [...traces].sort((a, b) => b.setId.localeCompare(a.setId));
  const series = sortedTraces.map((trace, idx) => {
    const xs = trace.pointsData.map((p) => p.trials).filter((v): v is number => typeof v === "number" && v > 0);
    const ys = trace.pointsData.map((p) => (p.rel_error ?? p.epsilon)).map((v) => (v !== null && v !== undefined && v > 0 ? v : null));
    return {
      type: "scatter",
      mode: "lines",
      name: trace.setId,
      x: xs,
      y: ys,
      line: { color: inkColorway[idx % inkColorway.length], width: 1.2 },
      hovertemplate: `${trace.setId}<br>trials=%{x:.2s}<br>err=%{y:.2e}<extra></extra>`,
    };
  });
  const oracle = traces[0]?.oracleP;
  return (
    <article className="trace-thumb">
      <header>
        <strong>{modelId}</strong>
        {oracle ? <span>oracle p = {formatNumber(oracle, 4)}</span> : null}
      </header>
      <Plot
        className="plot trace-plot"
        data={series}
        layout={{
          showlegend: false,
          xaxis: { title: "trials", type: "log", showgrid: false },
          yaxis: { title: "rel error", type: "log", showgrid: false },
          margin: { l: 56, r: 8, t: 8, b: 36 },
        }}
      />
    </article>
  );
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

function ModelExplorer({ data, onPreviewAsset }: { data: SiteData; onPreviewAsset: (asset: PlotAsset) => void }) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(data.models[0]?.modelId ?? "");
  const [sort, setSort] = useState<"family" | "size" | "speed">("family");
  const [loadedTraceModel, setLoadedTraceModel] = useState("");
  const [modelTraces, setModelTraces] = useState<TraceSeries[]>([]);

  const filteredModels = useMemo(() => {
    const q = query.toLowerCase();
    let rows = data.models.filter((model) => !q || model.modelName.toLowerCase().includes(q) || model.family.toLowerCase().includes(q));
    if (sort === "family") rows = [...rows].sort((a, b) => `${a.family}-${a.modelName}`.localeCompare(`${b.family}-${b.modelName}`));
    if (sort === "size") rows = [...rows].sort((a, b) => (b.maxPdagGates ?? 0) - (a.maxPdagGates ?? 0));
    if (sort === "speed") rows = [...rows].sort((a, b) => (b.bestBitsPerSecond ?? 0) - (a.bestBitsPerSecond ?? 0));
    return rows;
  }, [data.models, query, sort]);

  const selected = data.models.find((model) => model.modelId === selectedId) ?? filteredModels[0] ?? data.models[0];
  const modelRuns = useMemo(() => data.performance.filter((run) => run.modelId === selected.modelId), [data.performance, selected.modelId]);
  const modelCompiler = useMemo(() => data.compiler.rows.filter((row) => row.modelId === selected.modelId), [data.compiler.rows, selected.modelId]);
  const modelAssets = useMemo(() => data.artifacts.plots.filter((asset) => asset.modelId === selected.modelId).slice(0, 6), [data.artifacts.plots, selected.modelId]);

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
        <div className="model-controls">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search model or family" />
          <Select label="Sort" value={sort} onChange={(value) => setSort(value as "family" | "size" | "speed")} options={["family", "size", "speed"]} />
        </div>
        <div className="model-buttons">
          {filteredModels.map((model) => (
            <button
              key={model.modelId}
              className={classNames("model-button", selected.modelId === model.modelId && "active")}
              onClick={() => setSelectedId(model.modelId)}
            >
              <strong>{model.modelName}</strong>
              <span>
                {model.family} · {formatInteger(model.maxPdagGates)} gates · best {formatNumber(model.bestBitsPerSecond, 3)} bits/s
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow={selected.family} title={selected.modelName} text="Reference probabilities, best observed performance, and compiler reduction summary." />
        <div className="metric-grid">
          <Metric label="Oracle probability" value={formatNumber(selected.reference?.oracleP, 4)} />
          <Metric label="Rare-event approx" value={formatNumber(selected.reference?.rareP, 4)} />
          <Metric label="MCUB" value={formatNumber(selected.reference?.mcubP, 4)} />
          <Metric label="Min cut sets" value={formatNumber(selected.reference?.mcs ?? null, 3)} />
          <Metric label="Best bits/s" value={formatNumber(selected.bestBitsPerSecond, 3)} detail={selected.bestDevice ?? undefined} />
          <Metric label="Best rel error" value={formatNumber(selected.bestRelError, 3)} />
          <Metric label="PDAG gates max" value={formatInteger(selected.maxPdagGates)} />
          <Metric label="Basic events" value={formatInteger(selected.basicEvents)} />
        </div>
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Per-device performance" title="Best bits/s across the available backends" />
        <Plot
          data={[
            {
              type: "bar",
              orientation: "h",
              x: DEVICE_ORDER.map((dev) => selected.perDeviceBestBps[dev] ?? null),
              y: DEVICE_ORDER,
              marker: { color: DEVICE_ORDER.map((dev) => DEVICE_COLOR[dev] ?? inkPalette.ink) },
              hovertemplate: "%{y}<br>peak=%{x:.3s} bits/s<extra></extra>",
            },
          ]}
          layout={{
            xaxis: { title: "best bits per second", type: "log" },
            yaxis: { autorange: "reversed" },
            margin: { l: 130, r: 16, t: 22, b: 48 },
          }}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Convergence trace" title={loadedTraceModel === selected.modelId && modelTraces.length ? `${modelTraces.length} sampled traces` : loadedTraceModel === selected.modelId ? "No trace available" : "Loading..."} />
        {modelTraces.length ? (
          <ModelTraceStack traces={modelTraces} />
        ) : (
          <EmptyState text={loadedTraceModel === selected.modelId ? "No sampled trace was archived for this model." : "Loading sampled convergence traces..."} />
        )}
      </div>

      <div className="panel span-2">
        <SectionTitle eyebrow="Run detail" title="Convergence rows" />
        <CompactTable
          headers={["Set", "Device", "Flag", "kn", "xor", "δ", "Time", "Bits/s", "rel err"]}
          rows={modelRuns.map((run) => [
            run.setId,
            run.deviceShort,
            run.compileFlag,
            run.noKn ? "off" : "on",
            run.noXor ? "off" : "on",
            formatNumber(run.delta, 3),
            formatDurationMs(run.convergenceTimeMs),
            formatNumber(run.bitsPerSecond, 3),
            formatNumber(run.relError, 3),
          ])}
        />
      </div>

      <div className="panel">
        <SectionTitle eyebrow="Compiler detail" title="Compression configurations" />
        <CompactTable
          headers={["Flag", "kn", "xor", "PDAG gates", "Factor"]}
          rows={modelCompiler.slice(0, 60).map((row) => [
            row.compileFlag,
            row.noKn ? "off" : "on",
            row.noXor ? "off" : "on",
            formatInteger(row.pdagGates),
            formatNumber(row.compressionAllNodes, 3),
          ])}
        />
      </div>

      <div className="panel span-3">
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

function ModelTraceStack({ traces }: { traces: TraceSeries[] }) {
  const ordered = [...traces].slice(0, 6);
  return (
    <Plot
      data={ordered.flatMap((trace, idx) => {
        const xs = trace.pointsData.map((p) => p.trials);
        const ys = trace.pointsData.map((p) => p.rel_error ?? p.epsilon ?? null);
        return [
          {
            type: "scatter",
            mode: "lines",
            name: `${trace.setId} · ${trace.timestamp}`,
            x: xs,
            y: ys,
            line: { color: inkColorway[idx % inkColorway.length], width: 1.4 },
            hovertemplate: `${trace.setId}<br>trials=%{x:.3s}<br>err=%{y:.3e}<extra></extra>`,
          },
        ];
      })}
      layout={{
        xaxis: { title: "trials", type: "log" },
        yaxis: { title: "rel error / epsilon", type: "log" },
        legend: { y: -0.28, orientation: "h", font: { size: 10 } },
        margin: { l: 70, r: 12, t: 12, b: 130 },
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

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
        <SectionTitle eyebrow="PNG/SVG" title="Web plot gallery" text="Browser-friendly images bundled into the Pages artifact; EPS originals and source scripts are linked." />
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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function ReferenceProbabilityPlot({ models }: { models: ModelCard[] }) {
  const grouped = useMemo(() => {
    return models
      .filter((m) => m.reference?.oracleP)
      .sort((a, b) => (a.reference?.oracleP ?? 0) - (b.reference?.oracleP ?? 0));
  }, [models]);
  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "markers",
          x: grouped.map((m) => m.reference?.oracleP),
          y: grouped.map((_, idx) => idx),
          text: grouped.map((m) => m.modelName),
          marker: {
            color: grouped.map((m) => m.reference?.oracleP ?? 0),
            colorscale: inkColorscale,
            size: 10,
            line: { color: inkPalette.ink, width: 0.6 },
          },
          hovertemplate: "%{text}<br>p=%{x:.3e}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "markers",
          name: "rare-event approx",
          x: grouped.map((m) => m.reference?.rareP ?? null),
          y: grouped.map((_, idx) => idx),
          marker: { color: inkPalette.madder, size: 6, symbol: "x", line: { color: inkPalette.ink, width: 0.6 } },
          hovertemplate: "rare=%{x:.3e}<extra></extra>",
        },
      ]}
      layout={{
        xaxis: { title: "probability", type: "log" },
        yaxis: { title: "models sorted by oracle p", showticklabels: false },
        showlegend: false,
        margin: { l: 32, r: 16, t: 16, b: 56 },
      }}
    />
  );
}

function buildErrorVsTrialsByDevice(runs: PerformanceRun[]): Record<string, { x: number[]; y: number[]; text: string[] }> {
  const out: Record<string, { x: number[]; y: number[]; text: string[] }> = {};
  for (const run of runs) {
    if (!run.totalBits || !run.relError || run.relError <= 0) continue;
    const bucket = out[run.deviceShort] ?? (out[run.deviceShort] = { x: [], y: [], text: [] });
    bucket.x.push(run.totalBits);
    bucket.y.push(run.relError);
    bucket.text.push(`${run.modelName} · ${run.deviceShort} · ${run.compileFlag}`);
  }
  return out;
}

function fitLine(
  fit: LogFit | undefined,
  color: string,
  name: string,
  xRange?: [number, number],
): Record<string, unknown> | null {
  if (!fit || fit.slope === null || fit.intercept === null) return null;
  const [xMin, xMax] = xRange ?? [10, 100_000];
  const logMin = Math.log10(Math.max(xMin, 1));
  const logMax = Math.log10(Math.max(xMax, xMin * 10));
  const xs: number[] = [];
  const steps = 16;
  for (let i = 0; i <= steps; i += 1) {
    xs.push(Math.pow(10, logMin + ((logMax - logMin) * i) / steps));
  }
  const ys = xs.map((x) => Math.pow(10, fit.slope! * Math.log10(x) + fit.intercept!));
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: xs,
    y: ys,
    line: { color, dash: "dot", width: 1.5 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

function formatFitSlope(fit: LogFit | undefined): string {
  if (!fit || fit.slope === null) return "n/a";
  return fit.slope.toFixed(2);
}

function orderDevices(devices: DeviceSummary[]): DeviceSummary[] {
  return [...devices].sort((a, b) => {
    const ai = DEVICE_ORDER.indexOf(a.deviceShort);
    const bi = DEVICE_ORDER.indexOf(b.deviceShort);
    if (ai === -1 && bi === -1) return a.deviceShort.localeCompare(b.deviceShort);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}

function groupBy<T, K extends string>(rows: T[], pick: (row: T) => K | null | undefined): Record<K, T[]> {
  const out = {} as Record<K, T[]>;
  for (const row of rows) {
    const key = pick(row);
    if (!key) continue;
    if (!out[key]) out[key] = [];
    out[key].push(row);
  }
  return out;
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
            <th>Set</th>
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
              <td>{run.deviceShort}</td>
              <td>{run.compileFlag}</td>
              <td>{run.setId}</td>
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

function CompactTable({ headers, rows }: { headers: string[]; rows: Array<Array<string | number | null | undefined>> }) {
  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.join("-")}-${index}`}>
              {row.map((cell, cellIndex) => <td key={`${cell}-${cellIndex}`}>{cell ?? "—"}</td>)}
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
