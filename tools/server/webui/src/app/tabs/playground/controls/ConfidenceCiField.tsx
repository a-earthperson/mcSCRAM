import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { parseZodField } from "../../../../lib/parseZodField";
import {
  CONFIDENCE_CI_FIELD_HINT,
  confidenceCiPercentSchema,
  formatCiDescription,
  formatCiInput,
  MAX_CONFIDENCE_CI_PERCENT,
  MIN_CONFIDENCE_CI_PERCENT,
  stepConfidenceCiPercent,
} from "./samplerCI";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function ConfidenceCiField({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (ciPercent: number) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatCiInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatCiInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    const parsed = parseZodField(confidenceCiPercentSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatCiInput(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepConfidenceCiPercent(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatCiInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--confidence-ci">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Confidence (CI)
        </label>
        <FieldHintTrigger label="Confidence (CI) help" content={CONFIDENCE_CI_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease confidence CI"
        incrementLabel="Increase confidence CI"
        onStep={applyStep}
        disableDecrement={stepConfidenceCiPercent(value, -1) === value}
        disableIncrement={stepConfidenceCiPercent(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">% CI</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_CONFIDENCE_CI_PERCENT,
          max: MAX_CONFIDENCE_CI_PERCENT,
          step: "any",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatCiInput(value));
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
            if (!ok) setDraft(formatCiInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatCiDescription(value)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
