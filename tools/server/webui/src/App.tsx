import { Suspense } from "react";
import { Navigate, Route, Routes } from "react-router";
import { Hero } from "./app/shell/Hero";
import { PlaygroundRouter } from "./app/tabs/playground/PlaygroundRoutes";
import { playgroundPath } from "./app/tabs/playground/routes";

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
          <Route path="*" element={<Navigate to={playgroundPath} replace />} />
        </Routes>
      </Suspense>
    </>
  );
}

export default App;
