import { useCallback, useMemo, useState } from "react";
import { parseZodField } from "../../../lib/parseZodField";
import {
  ciPercentToConfidence,
  confidenceCiPercentSchema,
  DEFAULT_CONFIDENCE_CI_PERCENT,
} from "./samplerCI";
import { DEFAULT_DELTA, deltaSchema } from "./samplerDelta";
import { DEFAULT_OVERHEAD_RATIO, overheadRatioSchema } from "./samplerOverheadRatio";
import {
  createTrialsMantissaSchema,
  DEFAULT_TRIALS_MANTISSA,
  DEFAULT_TRIALS_SCALE_EXPONENT,
  effectiveTrialsCount,
  type TrialsScaleExponent,
} from "./samplerTrials";

export type SamplerSettings = {
  trialsMantissa: number;
  trialsScaleExponent: TrialsScaleExponent;
  trialsCount: number;
  setTrialsMantissa: (mantissa: number) => void;
  setTrialsScaleExponent: (scaleExponent: TrialsScaleExponent) => void;
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
  const [trialsMantissa, setTrialsMantissaState] = useState(DEFAULT_TRIALS_MANTISSA);
  const [trialsScaleExponent, setTrialsScaleExponentState] = useState(DEFAULT_TRIALS_SCALE_EXPONENT);
  const [overheadRatio, setOverheadRatioState] = useState(DEFAULT_OVERHEAD_RATIO);
  const [stopOnConvergence, setStopOnConvergence] = useState(true);
  const [confidenceCiPercent, setConfidenceCiPercentState] = useState(DEFAULT_CONFIDENCE_CI_PERCENT);
  const [delta, setDeltaState] = useState(DEFAULT_DELTA);

  const setTrialsMantissa = useCallback((mantissa: number) => {
    const parsed = parseZodField(createTrialsMantissaSchema(trialsScaleExponent), mantissa);
    if (parsed.ok) setTrialsMantissaState(parsed.value);
  }, [trialsScaleExponent]);

  const setTrialsScaleExponent = useCallback((scaleExponent: TrialsScaleExponent) => {
    setTrialsScaleExponentState(scaleExponent);
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

  const trialsCount = useMemo(
    () => effectiveTrialsCount(trialsMantissa, trialsScaleExponent),
    [trialsMantissa, trialsScaleExponent],
  );
  const confidence = useMemo(() => ciPercentToConfidence(confidenceCiPercent), [confidenceCiPercent]);

  return {
    trialsMantissa,
    trialsScaleExponent,
    trialsCount,
    setTrialsMantissa,
    setTrialsScaleExponent,
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
