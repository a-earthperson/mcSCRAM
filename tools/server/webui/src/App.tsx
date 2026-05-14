import { useEffect, useState } from "react";
import { Hero } from "./app/Hero";
import { loadJson } from "./app/loadJson";
import { AssetPreviewModal } from "./app/ui";
import {
  Artifacts,
  CompilerData,
  ConvergenceView,
  DeviceSummary,
  ExactResult,
  HardwareData,
  ModelCard,
  NarrativeCard,
  PerformanceRun,
  PlotAsset,
  SetIndexEntry,
  SiteData,
  SpeedupData,
  Summary,
  Tallies,
  TraceIndexData,
} from "./types";
import { classNames } from "./lib/format";
import { ArtifactGallery } from "./app/tabs/ArtifactsTab";
import { CompilerDashboard } from "./app/tabs/CompilerTab";
import { ConvergenceDashboard } from "./app/tabs/ConvergenceTab";
import { HardwareDashboard } from "./app/tabs/HardwareTab";
import { ModelExplorer } from "./app/tabs/ModelsTab";
import { Overview } from "./app/tabs/OverviewTab";
import { PerformanceDashboard } from "./app/tabs/PerformanceTab";

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

export default App;
