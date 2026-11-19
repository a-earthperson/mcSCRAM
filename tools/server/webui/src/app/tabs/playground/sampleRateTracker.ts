export type SampleRateObservation = {
  cumulativeTrials: number;
  cumulativeTimeSeconds: number;
  trialsPerSecond: number;
};

export type SampleRateTracker = {
  start: () => void;
  clear: () => void;
  observe: (trials: number, timestampSeconds: number) => SampleRateObservation | null;
};

export function createSampleRateTracker(): SampleRateTracker {
  let anchor: { trials: number; timestampSeconds: number } | null = null;

  const reset = () => {
    anchor = null;
  };

  return {
    start: reset,
    clear: reset,
    observe(trials, timestampSeconds) {
      if (!Number.isFinite(trials) || trials <= 0 || !Number.isFinite(timestampSeconds)) {
        return null;
      }
      if (!anchor) {
        anchor = { trials, timestampSeconds };
        return null;
      }
      const cumulativeTrials = trials - anchor.trials;
      const cumulativeTimeSeconds = timestampSeconds - anchor.timestampSeconds;
      if (cumulativeTimeSeconds <= 0 || cumulativeTrials < 0) {
        return null;
      }
      return {
        cumulativeTrials,
        cumulativeTimeSeconds,
        trialsPerSecond: cumulativeTrials / cumulativeTimeSeconds,
      };
    },
  };
}
