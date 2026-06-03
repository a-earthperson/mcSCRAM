import type { SamplingControls } from "./useSamplingControls";

type SamplerTransportControlsProps = Pick<
  SamplingControls,
  | "resetPending"
  | "samplingControlPending"
  | "samplingActive"
  | "controlsLocked"
  | "clientPaused"
  | "resetCalculation"
  | "toggleSampling"
  | "setPaused"
>;

export function SamplerTransportControls({
  resetPending,
  samplingControlPending,
  samplingActive,
  controlsLocked,
  clientPaused,
  resetCalculation,
  toggleSampling,
  setPaused,
}: SamplerTransportControlsProps) {
  return (
    <div className="sampler-transport-controls">
      <div className="sampler-transport-controls__row" role="toolbar" aria-label="Sampling transport">
        <button
          type="button"
          className="control-button control-reset"
          onClick={resetCalculation}
          disabled={controlsLocked}
        >
          {resetPending ? "Resetting" : "Reset"}
        </button>
        <button
          type="button"
          className="control-button control-sampling"
          onClick={toggleSampling}
          disabled={controlsLocked}
        >
          {samplingControlPending
            ? (samplingControlPending === "start" ? "Starting" : "Stopping")
            : samplingActive
              ? "Stop"
              : "Start"}
        </button>
        <button
          type="button"
          className="pause-play-button control-pause"
          onClick={() => setPaused(!clientPaused)}
          aria-label={clientPaused ? "Resume client updates" : "Pause client updates"}
          title={clientPaused ? "Unpause client updates" : "Pause client updates"}
        >
          {clientPaused ? (
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>
          )}
        </button>
      </div>
      <p className="sampler-transport-controls__hint">
        {clientPaused ? "Client updates paused" : "Reset and start/stop reach the server; pause is local only"}
      </p>
    </div>
  );
}
