import { useSyncExternalStore } from "react";

export type SamplingRunMetrics = {
  trialsPerSecond: number | null;
  completedPasses: number | null;
  /** Live GBAS projected pass budget; populated while streaming when early-stop is on. */
  gbasProjectedPasses: number | null;
  elapsedSeconds: number | null;
  timeToConvergenceSeconds: number | null;
  epsilonConverged: boolean;
};

export type SamplingRunMetricsPatch = Partial<SamplingRunMetrics>;

export type SamplingRunMetricsStore = {
  getSnapshot: () => SamplingRunMetrics;
  subscribe: (listener: () => void) => () => void;
  setMetrics: (metrics: SamplingRunMetricsPatch) => void;
  resetMetrics: () => void;
};

export const EMPTY_SAMPLING_RUN_METRICS: SamplingRunMetrics = {
  trialsPerSecond: null,
  completedPasses: null,
  gbasProjectedPasses: null,
  elapsedSeconds: null,
  timeToConvergenceSeconds: null,
  epsilonConverged: false,
};

function metricsEqual(left: SamplingRunMetrics, right: SamplingRunMetrics): boolean {
  return (
    left.trialsPerSecond === right.trialsPerSecond &&
    left.completedPasses === right.completedPasses &&
    left.gbasProjectedPasses === right.gbasProjectedPasses &&
    left.elapsedSeconds === right.elapsedSeconds &&
    left.timeToConvergenceSeconds === right.timeToConvergenceSeconds &&
    left.epsilonConverged === right.epsilonConverged
  );
}

export function createSamplingRunMetricsStore(
  initialMetrics: SamplingRunMetrics = EMPTY_SAMPLING_RUN_METRICS,
): SamplingRunMetricsStore {
  let snapshot = initialMetrics;
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setMetrics: (metrics) => {
      const next = { ...snapshot, ...metrics };
      if (metricsEqual(snapshot, next)) return;
      snapshot = next;
      listeners.forEach((listener) => listener());
    },
    resetMetrics: () => {
      if (metricsEqual(snapshot, EMPTY_SAMPLING_RUN_METRICS)) return;
      snapshot = EMPTY_SAMPLING_RUN_METRICS;
      listeners.forEach((listener) => listener());
    },
  };
}

export function useSamplingRunMetrics(store: SamplingRunMetricsStore): SamplingRunMetrics {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}
