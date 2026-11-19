export type SampleTimingObservation = {
  elapsedSeconds: number;
  timeToConvergenceSeconds: number | null;
  epsilonConverged: boolean;
};

export type SampleTimingTracker = {
  start: () => void;
  clear: () => void;
  observe: (timestampSeconds: number, epsilonConverged: boolean) => SampleTimingObservation | null;
};

export function createSampleTimingTracker(): SampleTimingTracker {
  let sessionStartTimestamp: number | null = null;
  let timeToConvergenceSeconds: number | null = null;
  let epsilonConverged = false;

  const reset = () => {
    sessionStartTimestamp = null;
    timeToConvergenceSeconds = null;
    epsilonConverged = false;
  };

  return {
    start: reset,
    clear: reset,
    observe(timestampSeconds, sampleEpsilonConverged) {
      if (!Number.isFinite(timestampSeconds)) return null;
      if (!sessionStartTimestamp) sessionStartTimestamp = timestampSeconds;
      const elapsedSeconds = timestampSeconds - sessionStartTimestamp;
      if (!epsilonConverged && sampleEpsilonConverged) {
        epsilonConverged = true;
        timeToConvergenceSeconds = elapsedSeconds;
      }
      return {
        elapsedSeconds,
        timeToConvergenceSeconds,
        epsilonConverged,
      };
    },
  };
}
