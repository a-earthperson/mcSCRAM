import { lazy } from "react";

/** Eager tab imports for SSR — Playground stays lazy (WebGL/MathBox is client-only). */
export { LiveDemo as LiveDemoTab } from "./live-demo/LiveDemoTab";
export { Overview as OverviewTab } from "./overview/OverviewTab";
export { HardwareDashboard as HardwareTab } from "./hardware/HardwareTab";
export { PerformanceDashboard as PerformanceTab } from "./performance/PerformanceTab";
export { CompilerDashboard as CompilerTab } from "./compiler/CompilerTab";
export { ConvergenceDashboard as ConvergenceTab } from "./convergence/ConvergenceTab";
export { ModelExplorer as ModelsTab } from "./models/ModelsTab";
export { ArtifactGallery as ArtifactsTab } from "./artifacts/ArtifactsTab";

export const PlaygroundTab = lazy(() =>
  import("./playground/Playground").then((m) => ({ default: m.Playground })),
);

export { sections, defaultSection, sectionHref, type SectionPath } from "./lazyTabs";
