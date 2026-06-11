import { Suspense, useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes } from "react-router";
import { Hero } from "./app/shell/Hero";
import { loadSiteData } from "./app/lib/loadSiteData";
import { BenchmarksLayout } from "./app/tabs/benchmarks/BenchmarksLayout";
import { PlaygroundRouter } from "./app/tabs/playground/PlaygroundRoutes";
import { benchmarksBasePath } from "./app/tabs/benchmarks/routes";
import { playgroundPath } from "./app/tabs/playground/routes";
import { topLevelNavSections } from "./app/tabs/routes";
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
      <nav className="tabs tabs-primary" aria-label="Site Sections">
        {topLevelNavSections.map((item) => (
          <NavLink
            key={item.href}
            to={item.href}
            className={({ isActive }) => classNames("tab", isActive && "active")}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <Suspense fallback={<TabFallback />}>
        <Routes>
          <Route path="/" element={<Navigate to={playgroundPath} replace />} />
          <Route
            path={`${playgroundPath}/*`}
            element={(
              <main className="shell">
                <PlaygroundRouter fallback={<TabFallback />} />
              </main>
            )}
          />
          <Route
            path={`${benchmarksBasePath}/*`}
            element={<BenchmarksLayout data={data} onPreviewAsset={setPreviewAsset} />}
          />
          <Route path="*" element={<Navigate to={playgroundPath} replace />} />
        </Routes>
      </Suspense>
      {previewAsset && <AssetPreviewModal asset={previewAsset} onClose={() => setPreviewAsset(null)} />}
    </>
  );
}

export default App;
