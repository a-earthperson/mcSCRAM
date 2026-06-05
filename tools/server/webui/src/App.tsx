import { Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router";
import { Hero } from "./app/shell/Hero";
import { loadSiteData } from "./app/lib/loadSiteData";
import { ClientOnly } from "./app/components/ui/ClientOnly";
import {
  ArtifactsTab,
  CompilerTab,
  ConvergenceTab,
  HardwareTab,
  LiveDemoTab,
  PlaygroundTab,
  ModelsTab,
  OverviewTab,
  PerformanceTab,
  sectionHref,
  sections,
} from "./app/tabs/lazyTabs";
import { AssetPreviewModal } from "./app/components";
import type { PlotAsset, SiteData } from "./app/types";
import { classNames } from "./app/lib/format";

import "./app/styles/toc.css";

function TabFallback() {
  return (
    <div className="tab-fallback" aria-busy="true">
      <div className="pulse" />
    </div>
  );
}

type AppProps = {
  initialData?: SiteData | null;
};

function App({ initialData = null }: AppProps) {
  const [data, setData] = useState<SiteData | null>(initialData);
  const [error, setError] = useState<string | null>(null);
  const [previewAsset, setPreviewAsset] = useState<PlotAsset | null>(null);

  useEffect(() => {
    if (initialData) return;
    loadSiteData()
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, [initialData]);

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
            <Route path="/" element={<LiveDemoTab />} />
            <Route path="/live-demo" element={<LiveDemoTab />} />
            <Route
              path="/playground"
              element={(
                <ClientOnly fallback={<TabFallback />}>
                  <PlaygroundTab />
                </ClientOnly>
              )}
            />
            <Route path="/overview" element={<OverviewTab data={data} onPreviewAsset={setPreviewAsset} />} />
            <Route path="/hardware" element={<HardwareTab data={data} />} />
            <Route path="/performance" element={<PerformanceTab data={data} />} />
            <Route path="/compiler" element={<CompilerTab data={data} />} />
            <Route path="/convergence" element={<ConvergenceTab data={data} />} />
            <Route path="/models" element={<ModelsTab data={data} onPreviewAsset={setPreviewAsset} />} />
            <Route path="/artifacts" element={<ArtifactsTab artifacts={data.artifacts} onPreviewAsset={setPreviewAsset} />} />
            <Route path="*" element={<LiveDemoTab />} />
          </Routes>
        </Suspense>
        {previewAsset && <AssetPreviewModal asset={previewAsset} onClose={() => setPreviewAsset(null)} />}
      </main>
    </>
  );
}

export default App;
