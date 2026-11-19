import { Suspense } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router";
import { Hero } from "./app/shell/Hero";
import { classNames } from "./app/lib/format";
import { BenchmarksRouter } from "./app/tabs/benchmarks/BenchmarksRoutes";
import { PlaygroundRouter } from "./app/tabs/playground/PlaygroundRoutes";
import { benchmarksPath } from "./app/tabs/benchmarks/routes";
import { playgroundPath } from "./app/tabs/playground/routes";
import { topLevelNavSections } from "./app/tabs/routes";

function TabFallback() {
  return (
    <div className="tab-fallback" aria-busy="true">
      <div className="pulse" />
    </div>
  );
}

function App() {
  return (
    <>
      <main className="shell" style={{ paddingBottom: 0 }}>
        <Hero />
      </main>
      <nav className="tabs tabs-primary" aria-label="Site sections">
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
            path={`${benchmarksPath}/*`}
            element={(
              <main className="shell">
                <BenchmarksRouter fallback={<TabFallback />} />
              </main>
            )}
          />
          <Route path="*" element={<Navigate to={playgroundPath} replace />} />
        </Routes>
      </Suspense>
    </>
  );
}

export default App;
