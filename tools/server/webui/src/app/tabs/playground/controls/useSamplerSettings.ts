import { useCallback, useMemo, useState } from "react";
import { parseZodField } from "../../../../lib/parseZodField";
import {
  ciPercentToConfidence,
  confidenceCiPercentSchema,
  DEFAULT_CONFIDENCE_CI_PERCENT,
} from "./samplerCI";
import { DEFAULT_DELTA, deltaSchema } from "./samplerDelta";
import { DEFAULT_OVERHEAD_RATIO, overheadRatioSchema } from "./samplerOverheadRatio";
import {
  DEFAULT_TRIALS_SCALAR,
  trialsScalarSchema,
  trialsScalarToCount,
} from "./samplerTrials";

export type SamplerSettings = {
  trialsScalar: number;
  trialsCount: number;
  setTrialsScalar: (scalar: number) => void;
  overheadRatio: number;
  setOverheadRatio: (ratio: number) => void;
  stopOnConvergence: boolean;
  setStopOnConvergence: (enabled: boolean) => void;
  confidenceCiPercent: number;
  setConfidenceCiPercent: (ciPercent: number) => void;
  /** mcSCRAM `ci_confidence` fraction derived from {@link confidenceCiPercent}. */
  confidence: number;
  delta: number;
  setDelta: (delta: number) => void;
};

export function useSamplerSettings(): SamplerSettings {
  const [trialsScalar, setTrialsScalarState] = useState(DEFAULT_TRIALS_SCALAR);
  const [overheadRatio, setOverheadRatioState] = useState(DEFAULT_OVERHEAD_RATIO);
  const [stopOnConvergence, setStopOnConvergence] = useState(true);
  const [confidenceCiPercent, setConfidenceCiPercentState] = useState(DEFAULT_CONFIDENCE_CI_PERCENT);
  const [delta, setDeltaState] = useState(DEFAULT_DELTA);

  const setTrialsScalar = useCallback((scalar: number) => {
    const parsed = parseZodField(trialsScalarSchema, scalar);
    if (parsed.ok) setTrialsScalarState(parsed.value);
  }, []);

  const setOverheadRatio = useCallback((ratio: number) => {
    const parsed = parseZodField(overheadRatioSchema, ratio);
    if (parsed.ok) setOverheadRatioState(parsed.value);
  }, []);

  const setConfidenceCiPercent = useCallback((ciPercent: number) => {
    const parsed = parseZodField(confidenceCiPercentSchema, ciPercent);
    if (parsed.ok) setConfidenceCiPercentState(parsed.value);
  }, []);

  const setDelta = useCallback((value: number) => {
    const parsed = parseZodField(deltaSchema, value);
    if (parsed.ok) setDeltaState(parsed.value);
  }, []);

  const trialsCount = useMemo(() => trialsScalarToCount(trialsScalar), [trialsScalar]);
  const confidence = useMemo(() => ciPercentToConfidence(confidenceCiPercent), [confidenceCiPercent]);

  return {
    trialsScalar,
    trialsCount,
    setTrialsScalar,
    overheadRatio,
    setOverheadRatio,
    stopOnConvergence,
    setStopOnConvergence,
    confidenceCiPercent,
    setConfidenceCiPercent,
    confidence,
    delta,
    setDelta,
  };
}
