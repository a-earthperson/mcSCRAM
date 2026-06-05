import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { parseZodField } from "../../../lib/parseZodField";
import {
  formatOverheadRatioDescription,
  formatOverheadRatioInput,
  MAX_OVERHEAD_RATIO,
  MIN_OVERHEAD_RATIO,
  OVERHEAD_RATIO_FIELD_HINT,
  overheadRatioSchema,
  stepOverheadRatio,
} from "./samplerOverheadRatio";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function OverheadRatioField({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (ratio: number) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatOverheadRatioInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatOverheadRatioInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    const parsed = parseZodField(overheadRatioSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatOverheadRatioInput(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepOverheadRatio(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatOverheadRatioInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--overhead-ratio">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Overhead ratio (r)
        </label>
        <FieldHintTrigger label="Overhead ratio help" content={OVERHEAD_RATIO_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease overhead ratio"
        incrementLabel="Increase overhead ratio"
        onStep={applyStep}
        disableDecrement={stepOverheadRatio(value, -1) === value}
        disableIncrement={stepOverheadRatio(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">r</span>}
        inputProps={{
          id: fieldId,
          type: "number",
          min: MIN_OVERHEAD_RATIO,
          max: MAX_OVERHEAD_RATIO,
          step: "any",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatOverheadRatioInput(value));
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
            if (!ok) setDraft(formatOverheadRatioInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatOverheadRatioDescription(value)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
