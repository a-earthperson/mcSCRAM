import { lazy } from "react";

/** Code-split tab views — each chunk loads only when its route is visited. */
// export const LiveDemoTab = lazy(() =>
//   import("./live-demo/LiveDemoTab").then((m) => ({ default: m.LiveDemo })),
// );
export const PlaygroundTab = lazy(() =>
  import("./playground/Playground").then((m) => ({ default: m.Playground })),
);
export const OverviewTab = lazy(() =>
  import("./benchmarks/OverviewTab").then((m) => ({ default: m.Overview })),
);
export const HardwareTab = lazy(() =>
  import("./benchmarks/HardwareTab").then((m) => ({ default: m.HardwareDashboard })),
);
export const PerformanceTab = lazy(() =>
  import("./benchmarks/PerformanceTab").then((m) => ({ default: m.PerformanceDashboard })),
);
export const CompilerTab = lazy(() =>
  import("./benchmarks/CompilerTab").then((m) => ({ default: m.CompilerDashboard })),
);
export const ConvergenceTab = lazy(() =>
  import("./benchmarks/ConvergenceTab").then((m) => ({ default: m.ConvergenceDashboard })),
);
export const ModelsTab = lazy(() =>
  import("./benchmarks/ModelsTab").then((m) => ({ default: m.ModelExplorer })),
);
export const ArtifactsTab = lazy(() =>
  import("./benchmarks/ArtifactsTab").then((m) => ({ default: m.ArtifactGallery })),
);
