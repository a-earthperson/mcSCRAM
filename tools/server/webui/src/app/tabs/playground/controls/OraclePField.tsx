import { useId, useState } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { parseZodField } from "../../../../lib/parseZodField";
import {
  formatOraclePDescription,
  formatOraclePInput,
  ORACLE_P_FIELD_HINT,
  ORACLE_P_OFF,
  oraclePValueSchema,
  stepOracleP,
} from "./oracleP";
import { SteppedNumberInputRow } from "./SteppedNumberInputRow";

export function OraclePField({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (oracleP: number) => void;
  disabled?: boolean;
}) {
  const fieldId = useId();
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const [focused, setFocused] = useState(false);
  const [draft, setDraft] = useState(() => formatOraclePInput(value));
  const [error, setError] = useState<string | null>(null);

  const displayValue = focused ? draft : formatOraclePInput(value);
  const describedBy = [error ? errorId : null, hintId].filter(Boolean).join(" ") || undefined;

  const commit = (raw: unknown) => {
    if (raw === "" || raw === null || raw === undefined) {
      setError(null);
      onChange(ORACLE_P_OFF);
      setDraft("");
      return true;
    }
    const parsed = parseZodField(oraclePValueSchema, raw);
    if (!parsed.ok) {
      setError(parsed.message);
      return false;
    }
    setError(null);
    onChange(parsed.value);
    setDraft(formatOraclePInput(parsed.value));
    return true;
  };

  const applyStep = (direction: 1 | -1) => {
    const next = stepOracleP(value, direction);
    setError(null);
    onChange(next);
    setDraft(formatOraclePInput(next));
    setFocused(false);
  };

  return (
    <div className="sticky-panel-group__number-field sticky-panel-group__number-field--oracle-p">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={fieldId}>
          Oracle P
        </label>
        <FieldHintTrigger label="Oracle P help" content={ORACLE_P_FIELD_HINT} />
      </div>
      <SteppedNumberInputRow
        decrementLabel="Decrease oracle p"
        incrementLabel="Increase oracle p"
        onStep={applyStep}
        disableDecrement={stepOracleP(value, -1) === value}
        disableIncrement={stepOracleP(value, 1) === value}
        disabled={disabled}
        suffix={<span className="sticky-panel-group__number-field-scale" aria-hidden="true">µ</span>}
        inputProps={{
          id: fieldId,
          type: "text",
          inputMode: "decimal",
          placeholder: "off",
          value: displayValue,
          "aria-invalid": error ? true : undefined,
          "aria-describedby": describedBy,
          onFocus: () => {
            setDraft(formatOraclePInput(value));
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
            if (!ok) setDraft(formatOraclePInput(value));
            setFocused(false);
          },
        }}
      />
      <p id={hintId} className="sticky-panel-group__number-field-detail" aria-live="polite">
        {formatOraclePDescription(value)}
      </p>
      {error ? (
        <p id={errorId} className="sticky-panel-group__number-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
