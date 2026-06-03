import { useCallback, useRef, useState } from "react";
import { mcscramHttpUrl } from "../../../mcscramClient";
import type { SamplerRunOptions } from "./samplerRunOptions";
import type { SamplingControlResponse, SamplingStatus } from "./samplingStatus";

export type SamplingControls = ReturnType<typeof useSamplingControls>;

export type UseSamplingControlsOptions = {
  /** Called when reset is requested (before the HTTP call). */
  onReset?: () => void;
  /** Called when sampling is started (before the HTTP call). */
  onBeforeStart?: () => void;
  /** Sampler options applied on /reset and /start. */
  getRunOptions?: () => SamplerRunOptions;
};

async function postSamplingControl(path: "/reset" | "/start" | "/stop", runOptions?: SamplerRunOptions) {
  const init: RequestInit = { method: "POST" };
  if (path !== "/stop" && runOptions) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(runOptions);
  }
  const response = await fetch(mcscramHttpUrl(path), init);
  if (!response.ok) throw new Error(`${path.slice(1)} failed: HTTP ${response.status}`);
  return response.json() as Promise<SamplingControlResponse>;
}

export function useSamplingControls(options?: UseSamplingControlsOptions) {
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const [status, setStatus] = useState<SamplingStatus | null>(null);
  const [resetPending, setResetPending] = useState(false);
  const [samplingControlPending, setSamplingControlPending] = useState<"start" | "stop" | null>(null);
  const [clientPaused, setClientPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const clientPausedRef = useRef(false);

  const samplingActive = status?.alive === true || status?.status === "running";
  const controlsLocked = resetPending || samplingControlPending !== null;

  const setPaused = useCallback((paused: boolean) => {
    clientPausedRef.current = paused;
    setClientPaused(paused);
  }, []);

  const resetCalculation = useCallback(async () => {
    if (resetPending) return;
    setResetPending(true);
    optionsRef.current?.onReset?.();
    try {
      const payload = await postSamplingControl("/reset", optionsRef.current?.getRunOptions?.());
      if (payload.status) setStatus(payload.status);
      if (!payload.accepted) setError(payload.reason ?? "Reset request was ignored.");
      else setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setResetPending(false);
    }
  }, [resetPending]);

  const toggleSampling = useCallback(async () => {
    const shouldStop = status?.alive === true || status?.status === "running";
    const action = shouldStop ? "stop" : "start";
    if (samplingControlPending) return;
    setSamplingControlPending(action);
    if (!shouldStop) optionsRef.current?.onBeforeStart?.();
    try {
      const payload = action === "stop"
        ? await postSamplingControl("/stop")
        : await postSamplingControl("/start", optionsRef.current?.getRunOptions?.());
      if (payload.status) setStatus(payload.status);
      if (!payload.accepted) setError(payload.reason ?? `${action} request was ignored.`);
      else setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSamplingControlPending(null);
    }
  }, [samplingControlPending, status?.alive, status?.status]);

  return {
    status,
    setStatus,
    error,
    setError,
    resetPending,
    samplingControlPending,
    clientPaused,
    clientPausedRef,
    samplingActive,
    controlsLocked,
    resetCalculation,
    toggleSampling,
    setPaused,
  };
}
