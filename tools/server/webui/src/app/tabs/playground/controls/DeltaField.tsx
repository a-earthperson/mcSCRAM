import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { parseZodField } from "../../../../lib/parseZodField";
import {
  DELTA_FIELD_HINT,
  deltaSchema,
  formatDeltaDescription,
  formatDeltaInput,
  MAX_DELTA,
  MIN_DELTA,
  stepDelta,
} from "./samplerDelta";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function DeltaField({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (delta: number) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatDeltaInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatDeltaInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    const parsed = parseZodField(deltaSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatDeltaInput(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepDelta(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatDeltaInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--delta">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Delta (δ)
        </label>
        <FieldHintTrigger label="Delta (δ) help" content={DELTA_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease delta"
        incrementLabel="Increase delta"
        onStep={applyStep}
        disableDecrement={stepDelta(value, -1) === value}
        disableIncrement={stepDelta(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">δ</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_DELTA,
          max: MAX_DELTA,
          step: "any",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatDeltaInput(value));
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
            if (!ok) setDraft(formatDeltaInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatDeltaDescription(value)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
