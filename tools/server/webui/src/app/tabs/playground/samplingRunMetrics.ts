import { useSyncExternalStore } from "react";

export type SamplingRunMetrics = {
  trialsPerSecond: number | null;
  completedPasses: number | null;
};

export type SamplingRunMetricsStore = {
  getSnapshot: () => SamplingRunMetrics;
  subscribe: (listener: () => void) => () => void;
  setMetrics: (metrics: SamplingRunMetrics) => void;
};

export const EMPTY_SAMPLING_RUN_METRICS: SamplingRunMetrics = {
  trialsPerSecond: null,
  completedPasses: null,
};

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
      if (
        snapshot.trialsPerSecond === metrics.trialsPerSecond &&
        snapshot.completedPasses === metrics.completedPasses
      ) {
        return;
      }
      snapshot = metrics;
      listeners.forEach((listener) => listener());
    },
  };
}

export function useSamplingRunMetrics(store: SamplingRunMetricsStore): SamplingRunMetrics {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}
