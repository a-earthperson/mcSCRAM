import type { ComponentPropsWithoutRef, ReactNode } from "react";

export function SteppedNumberInputRow({
  decrementLabel,
  incrementLabel,
  onStep,
  disableDecrement = false,
  disableIncrement = false,
  disabled = false,
  suffix,
  inputProps,
}: {
  decrementLabel: string;
  incrementLabel: string;
  onStep: (direction: 1 | -1) => void;
  disableDecrement?: boolean;
  disableIncrement?: boolean;
  disabled?: boolean;
  suffix?: ReactNode;
  inputProps: ComponentPropsWithoutRef<"input">;
}) {
  const { onKeyDown, ...restInputProps } = inputProps;

  return (
    <div className="sticky-panel-group__number-field-input-row sticky-panel-group__number-field-input-row--stepped">
      <NumberFieldStepButton
        direction={-1}
        label={decrementLabel}
        disabled={disabled || disableDecrement}
        onClick={() => onStep(-1)}
      />
      <input
        {...restInputProps}
        disabled={disabled || restInputProps.disabled}
        onKeyDown={(event) => {
          onKeyDown?.(event);
          if (event.defaultPrevented) return;
          if (event.key === "ArrowUp") {
            event.preventDefault();
            onStep(1);
          } else if (event.key === "ArrowDown") {
            event.preventDefault();
            onStep(-1);
          }
        }}
      />
      <NumberFieldStepButton
        direction={1}
        label={incrementLabel}
        disabled={disabled || disableIncrement}
        onClick={() => onStep(1)}
      />
      {suffix}
    </div>
  );
}

function NumberFieldStepButton({
  direction,
  label,
  disabled,
  onClick,
}: {
  direction: 1 | -1;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="sticky-panel-group__number-step"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
    >
      {direction < 0 ? "−" : "+"}
    </button>
  );
}
