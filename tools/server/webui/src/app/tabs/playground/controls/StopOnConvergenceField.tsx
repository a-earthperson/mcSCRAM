import { FieldHintTrigger } from "../../../components/HintTooltip";
import { StickyPanelCheckbox } from "../../../components/StickyPanel";

export const STOP_ON_CONVERGENCE_FIELD_HINT =
  "When enabled, sampling stops once the Wald CI policy converges, using Confidence (CI) and Delta (δ) for the stopping band (ε = δ·p̂). MAX TRIALS (N) is an upper bound (upto N). When disabled, the run draws exactly N trials; CI and δ are ignored.";

export function StopOnConvergenceField({
  checked,
  onChange,
  disabled = false,
}: {
  checked: boolean;
  onChange: (enabled: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="sticky-panel-group__toggle-row">
      <StickyPanelCheckbox
        id="playground-stop-on-convergence"
        label="STOP ON CONVERGENCE"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
      />
      <FieldHintTrigger label="STOP ON CONVERGENCE help" content={STOP_ON_CONVERGENCE_FIELD_HINT} />
    </div>
  );
}
