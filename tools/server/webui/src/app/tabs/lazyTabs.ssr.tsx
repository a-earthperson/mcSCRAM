import { lazy } from "react";

/** Eager tab imports for SSR — Playground stays lazy (WebGL/MathBox is client-only). */
// export { LiveDemo as LiveDemoTab } from "./live-demo/LiveDemoTab";
export { Overview as OverviewTab } from "./benchmarks/OverviewTab";
export { HardwareDashboard as HardwareTab } from "./benchmarks/HardwareTab";
export { PerformanceDashboard as PerformanceTab } from "./benchmarks/PerformanceTab";
export { CompilerDashboard as CompilerTab } from "./benchmarks/CompilerTab";
export { ConvergenceDashboard as ConvergenceTab } from "./benchmarks/ConvergenceTab";
export { ModelExplorer as ModelsTab } from "./benchmarks/ModelsTab";
export { ArtifactGallery as ArtifactsTab } from "./benchmarks/ArtifactsTab";

export const PlaygroundTab = lazy(() =>
  import("./playground/Playground").then((m) => ({ default: m.Playground })),
);
