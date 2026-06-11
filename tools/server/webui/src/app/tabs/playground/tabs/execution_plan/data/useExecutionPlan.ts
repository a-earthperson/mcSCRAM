import { useEffect, useState } from "react";
import {
  fetchExecutionPlanForModel,
  type CompileDagOptions,
  type ExecutionPlanPayload,
  type SampleShapeOptions,
} from "../../../../../api/faultTreeApi";

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export type UseExecutionPlanResult = {
  plan: ExecutionPlanPayload | null;
  loading: boolean;
  error: string | null;
};

/**
 * Fetch the native DirectEval execution plan for the current model + compile
 * switches + sampling shape. Cancels in-flight requests on dependency changes,
 * and surfaces both transport errors and the payload's own `error` field
 * (e.g. when the native extension is unavailable).
 */
export function useExecutionPlan({
  selectedId,
  compileOptions,
  sampleOptions,
}: {
  selectedId: string;
  compileOptions: CompileDagOptions;
  sampleOptions: SampleShapeOptions;
}): UseExecutionPlanResult {
  const [plan, setPlan] = useState<ExecutionPlanPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { numTrials, requestedTrialsPerPass } = sampleOptions;

  useEffect(() => {
    if (!selectedId) {
      setPlan(null);
      setError(null);
      setLoading(false);
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchExecutionPlanForModel(selectedId, compileOptions, { numTrials, requestedTrialsPerPass })
      .then((payload) => {
        if (cancelled) return;
        if (payload.error) {
          setPlan(null);
          setError(payload.error);
          return;
        }
        setPlan(payload);
        setError(null);
      })
      .catch((reason) => {
        if (cancelled) return;
        setPlan(null);
        setError(errorMessage(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId, compileOptions, numTrials, requestedTrialsPerPass]);

  return { plan, loading, error };
}
