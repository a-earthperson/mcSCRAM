import type { SamplingControls } from "./useSamplingControls";

type SamplerTransportControlsProps = Pick<
  SamplingControls,
  | "resetPending"
  | "samplingControlPending"
  | "samplingActive"
  | "streamPhase"
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
  streamPhase,
  controlsLocked,
  clientPaused,
  resetCalculation,
  toggleSampling,
  setPaused,
}: SamplerTransportControlsProps) {
  const showStarting = samplingControlPending === "start" || streamPhase === "arming";
  const showStopping = samplingControlPending === "stop" || streamPhase === "draining";
  const showStop = !showStarting && !showStopping && samplingActive;
  const showTransitional = showStarting || showStopping;
  const pigmentClass = showTransitional
    ? "control-transition"
    : showStop
      ? "control-stop"
      : "control-sampling";
  const label = showStarting
    ? "Starting"
    : showStopping
      ? "Stopping"
      : showStop
        ? "Stop"
        : "Start";

  return (
    <div className="sampler-transport-controls__row" role="toolbar" aria-label="Sampling transport">
      <button
        type="button"
        className={`control-button sampler-transport-controls__start ${pigmentClass}`}
        onClick={toggleSampling}
        disabled={controlsLocked}
      >
        {label}
      </button>
      <button
        type="button"
        style={{ display: "none" }}
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
  );
}
