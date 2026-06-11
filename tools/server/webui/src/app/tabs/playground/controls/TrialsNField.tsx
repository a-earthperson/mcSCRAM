import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import { parseZodField } from "../../../lib/parseZodField";
import {
  createTrialsMantissaSchema,
  formatTrialsDescription,
  formatTrialsMantissa,
  isTrialsScaleExponent,
  stepTrialsMantissa,
  MIN_TRIALS_MANTISSA,
  TRIALS_MANTISSA_STEP,
  TRIALS_N_FIELD_HINT,
  TRIALS_SCALE_EXPONENTS,
  TRIALS_SCALE_LABELS,
  type TrialsScaleExponent,
} from "./samplerTrials";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function TrialsNField({
  mantissa,
  scaleExponent,
  onMantissaChange,
  onScaleExponentChange,
  stopOnConvergence = false,
  disabled = false,
}: {
  mantissa: number;
  scaleExponent: TrialsScaleExponent;
  onMantissaChange: (mantissa: number) => void;
  onScaleExponentChange: (scaleExponent: TrialsScaleExponent) => void;
  stopOnConvergence?: boolean;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const scaleId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatTrialsMantissa(mantissa));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatTrialsMantissa(mantissa);
  const describedBy = [error ? errorId : null, hintId, scaleId].filter(Boolean).join(" ") || undefined;
  const fieldLabel = stopOnConvergence ? "Max Trials" : "Trials";

  const validate = (raw: unknown, exponent: TrialsScaleExponent) => {
    const parsed = parseZodField(createTrialsMantissaSchema(exponent), raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return null;
    }
    setError(null);
    return parsed.value;
  };

  const commit = (raw: unknown, exponent = scaleExponent) => {
    if (raw === "" || raw === null || raw === undefined || raw === "auto") {
      setError(null);
      onMantissaChange(0);
      setDraft("");
      return true;
    }
    const next = validate(raw, exponent);
    if (next === null) return false;
    onMantissaChange(next);
    setDraft(formatTrialsMantissa(next));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const candidate = stepTrialsMantissa(mantissa, direction);
    if (!commit(candidate)) return;
    setFocused(false);
  };

  const handleScaleChange = (raw: string) => {
    const nextScale = Number(raw);
    if (!isTrialsScaleExponent(nextScale) || nextScale === scaleExponent) return;
    onScaleExponentChange(nextScale);
    validate(mantissa, nextScale);
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--trials">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          {fieldLabel}
        </label>
        <FieldHintTrigger label={`${fieldLabel} help`} content={TRIALS_N_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease trials mantissa"
        incrementLabel="Increase trials mantissa"
        onStep={applyStep}
        disableDecrement={stepTrialsMantissa(mantissa, -1) === mantissa}
        disableIncrement={stepTrialsMantissa(mantissa, 1) === mantissa}
        disabled={disabled}
        suffix={(
          <select
            id={scaleId}
            className="sticky-panel-group__number-field-scale sticky-panel-group__number-field-scale-select"
            value={scaleExponent}
            disabled={disabled}
            aria-label="Trials magnitude unit"
            onChange={(event) => handleScaleChange(event.currentTarget.value)}
          >
            {TRIALS_SCALE_EXPONENTS.map((exponent) => (
              <option key={exponent} value={exponent}>
                {TRIALS_SCALE_LABELS[exponent]}
              </option>
            ))}
          </select>
        )}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_TRIALS_MANTISSA,
          step: TRIALS_MANTISSA_STEP,
          placeholder: "auto",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatTrialsMantissa(mantissa));
            setFocused(true);
          },
          onChange: (event) => {
            const nextText = event.currentTarget.value;
            setDraft(nextText);
            if (error) setError(null);
            if (!focused) commit(nextText);
          },
          onBlur: (event) => {
            const ok = commit(event.currentTarget.value);
            if (!ok) setDraft(formatTrialsMantissa(mantissa));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatTrialsDescription(mantissa, scaleExponent, stopOnConvergence)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
