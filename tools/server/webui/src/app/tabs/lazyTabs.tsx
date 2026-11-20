import { lazy } from "react";

/** Code-split the playground — WebGL/MathBox is client-only. */
export const PlaygroundTab = lazy(() =>
  import("./playground/Playground").then((m) => ({ default: m.Playground })),
);

export const BenchmarksTab = lazy(() =>
  import("./benchmarks/Benchmarks").then((m) => ({ default: m.Benchmarks })),
);
