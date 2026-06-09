import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import { parseZodField } from "../../../lib/parseZodField";
import {
  formatRequestedTrialsPerPassDescription,
  formatRequestedTrialsPerPassInput,
  MAX_REQUESTED_TRIALS_PER_PASS,
  MIN_REQUESTED_TRIALS_PER_PASS,
  REQUESTED_TRIALS_PER_PASS_FIELD_HINT,
  requestedTrialsPerPassSchema,
  stepRequestedTrialsPerPass,
} from "./samplerTrialsPerPass";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function TrialsPerPassField({
  value,
  onChange,
  visible = true,
  disabled = false,
}: {
  value: number;
  onChange: (trials: number) => void;
  visible?: boolean;
  disabled?: boolean;
}) {
  if (!visible) return null;
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatRequestedTrialsPerPassInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatRequestedTrialsPerPassInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    if (raw === "" || raw === null || raw === undefined || raw === "auto") {
      setError(null);
      onChange(0);
      setDraft("");
      return true;
    }
    const parsed = parseZodField(requestedTrialsPerPassSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatRequestedTrialsPerPassInput(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepRequestedTrialsPerPass(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatRequestedTrialsPerPassInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--trials-per-pass">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Trials / pass
        </label>
        <FieldHintTrigger label="Trials per pass help" content={REQUESTED_TRIALS_PER_PASS_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease trials per pass"
        incrementLabel="Increase trials per pass"
        onStep={applyStep}
        disableDecrement={stepRequestedTrialsPerPass(value, -1) === value}
        disableIncrement={stepRequestedTrialsPerPass(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">N</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_REQUESTED_TRIALS_PER_PASS,
          max: MAX_REQUESTED_TRIALS_PER_PASS,
          step: "any",
          placeholder: "auto",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatRequestedTrialsPerPassInput(value));
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
            if (!ok) setDraft(formatRequestedTrialsPerPassInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatRequestedTrialsPerPassDescription(value)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
