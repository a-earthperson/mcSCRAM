import { FieldHintTrigger } from "../../../components/ui/HintTooltip";
import { StickyPanelCheckbox } from "../../../components/ui/StickyPanel";

export const STOP_ON_CONVERGENCE_FIELD_HINT =
  "On: stop as soon as the estimate meets the Confidence and Delta (δ) target, or at Max Trials, whichever comes first. Off: run all Max Trials and ignore Confidence and δ. With Max Trials on auto, a run only ends when it converges or you stop it.";

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
