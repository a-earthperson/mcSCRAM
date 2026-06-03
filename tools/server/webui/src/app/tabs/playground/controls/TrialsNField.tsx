import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { parseZodField } from "../../../../lib/parseZodField";
import {
  formatTrialsDescription,
  formatTrialsScalar,
  MAX_TRIALS_SCALAR,
  MIN_TRIALS_SCALAR,
  TRIALS_N_FIELD_HINT,
  stepTrialsScalar,
  trialsScalarSchema,
} from "./samplerTrials";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function TrialsNField({
  value,
  onChange,
  stopOnConvergence = false,
  disabled = false,
}: {
  value: number;
  onChange: (scalar: number) => void;
  stopOnConvergence?: boolean;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatTrialsScalar(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatTrialsScalar(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;
  const fieldLabel = stopOnConvergence ? "MAX TRIALS (N)" : "Trials (N)";

  const commit = (raw: unknown) => {
    const parsed = parseZodField(trialsScalarSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatTrialsScalar(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepTrialsScalar(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatTrialsScalar(next));
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
        decrementLabel="Decrease trials multiplier"
        incrementLabel="Increase trials multiplier"
        onStep={applyStep}
        disableDecrement={stepTrialsScalar(value, -1) === value}
        disableIncrement={stepTrialsScalar(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">×10⁹</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_TRIALS_SCALAR,
          max: MAX_TRIALS_SCALAR,
          step: "any",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatTrialsScalar(value));
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
            if (!ok) setDraft(formatTrialsScalar(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatTrialsDescription(value, stopOnConvergence)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
