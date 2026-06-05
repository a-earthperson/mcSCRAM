import { lazy } from "react";

/** Code-split tab views — each chunk loads only when its route is visited. */
export const LiveDemoTab = lazy(() =>
  import("./live-demo/LiveDemoTab").then((m) => ({ default: m.LiveDemo })),
);
export const PlaygroundTab = lazy(() =>
  import("./playground/Playground").then((m) => ({ default: m.Playground })),
);
export const OverviewTab = lazy(() =>
  import("./overview/OverviewTab").then((m) => ({ default: m.Overview })),
);
export const HardwareTab = lazy(() =>
  import("./hardware/HardwareTab").then((m) => ({ default: m.HardwareDashboard })),
);
export const PerformanceTab = lazy(() =>
  import("./performance/PerformanceTab").then((m) => ({ default: m.PerformanceDashboard })),
);
export const CompilerTab = lazy(() =>
  import("./compiler/CompilerTab").then((m) => ({ default: m.CompilerDashboard })),
);
export const ConvergenceTab = lazy(() =>
  import("./convergence/ConvergenceTab").then((m) => ({ default: m.ConvergenceDashboard })),
);
export const ModelsTab = lazy(() =>
  import("./models/ModelsTab").then((m) => ({ default: m.ModelExplorer })),
);
export const ArtifactsTab = lazy(() =>
  import("./artifacts/ArtifactsTab").then((m) => ({ default: m.ArtifactGallery })),
);

export const sections = [
  { path: "live-demo", label: "Live Demo" },
  { path: "playground", label: "Playground" },
  { path: "overview", label: "Overview" },
  { path: "hardware", label: "Hardware" },
  { path: "performance", label: "Performance" },
  { path: "compiler", label: "Compiler" },
  { path: "convergence", label: "Convergence" },
  { path: "models", label: "Models" },
  { path: "artifacts", label: "Artifacts" },
] as const;

export type SectionPath = (typeof sections)[number]["path"];

export const defaultSection: SectionPath = "live-demo";

export function sectionHref(path: SectionPath): string {
  return `/${path}`;
}
