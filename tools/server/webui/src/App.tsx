import { Suspense, useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { Hero } from "./app/shell/Hero";
import { loadJson } from "./app/lib/loadJson";
import {
  ArtifactsTab,
  CompilerTab,
  ConvergenceTab,
  defaultSection,
  HardwareTab,
  LiveDemoTab,
  PlaygroundTab,
  ModelsTab,
  OverviewTab,
  PerformanceTab,
  sectionHref,
  sections,
} from "./app/tabs/lazyTabs";
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
} from "./app/types";
import { classNames } from "./app/lib/format";

import "./app/styles/toc.css";

function TabFallback() {
  return (
    <div className="tab-fallback" aria-busy="true">
      <div className="pulse" />
    </div>
  );
}

function App() {
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
        <p>Loading mcSCRAM benchmarks...</p>
      </main>
    );
  }

  return (
    <>
      <main className="shell" style={{ paddingBottom: 0 }}>
        <Hero data={data} />
      </main>
      <nav className="tabs" aria-label="Benchmark site sections">
        {sections.map((item) => (
          <NavLink
            key={item.path}
            to={sectionHref(item.path)}
            className={({ isActive }) => classNames("tab", isActive && "active")}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <main className="shell">
        <Suspense fallback={<TabFallback />}>
          <Routes>
            <Route path="/" element={<Navigate to={sectionHref(defaultSection)} replace />} />
            <Route path="/live-demo" element={<LiveDemoTab />} />
            <Route path="/playground" element={<PlaygroundTab />} />
            <Route path="/overview" element={<OverviewTab data={data} onPreviewAsset={setPreviewAsset} />} />
            <Route path="/hardware" element={<HardwareTab data={data} />} />
            <Route path="/performance" element={<PerformanceTab data={data} />} />
            <Route path="/compiler" element={<CompilerTab data={data} />} />
            <Route path="/convergence" element={<ConvergenceTab data={data} />} />
            <Route path="/models" element={<ModelsTab data={data} onPreviewAsset={setPreviewAsset} />} />
            <Route path="/artifacts" element={<ArtifactsTab artifacts={data.artifacts} onPreviewAsset={setPreviewAsset} />} />
            <Route path="*" element={<Navigate to={sectionHref(defaultSection)} replace />} />
          </Routes>
        </Suspense>
        {previewAsset && <AssetPreviewModal asset={previewAsset} onClose={() => setPreviewAsset(null)} />}
      </main>
    </>
  );
}

export default App;
