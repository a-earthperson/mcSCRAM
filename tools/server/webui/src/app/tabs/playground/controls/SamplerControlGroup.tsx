import { StickyPanelGroup, StickyPanelSlot, type StickyPanelSpan } from "../../../components/ui/StickyPanel";
import { ConfidenceCiField } from "./ConfidenceCiField";
import { DeltaField } from "./DeltaField";
import { TrialsPerPassField } from "./TrialsPerPassField";
import { TrialsNField } from "./TrialsNField";
import type { SamplerSettings } from "./useSamplerSettings";

export function SamplerControlGroup({
  settings,
  disabled = false,
  hSpan,
  vSpan,
}: {
  settings: SamplerSettings;
  disabled?: boolean;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Sampler" className="sticky-panel-group--sampler" hSpan={hSpan} vSpan={vSpan}>
      <div className="pg-fields">
        <TrialsNField
          mantissa={settings.trialsMantissa}
          scaleExponent={settings.trialsScaleExponent}
          onMantissaChange={settings.setTrialsMantissa}
          onScaleExponentChange={settings.setTrialsScaleExponent}
          stopOnConvergence={settings.stopOnConvergence}
          disabled={disabled}
        />
        <TrialsPerPassField
          value={settings.requestedTrialsPerPass}
          onChange={settings.setRequestedTrialsPerPass}
          disabled={disabled}
        />
        <ConfidenceCiField
          value={settings.confidenceCiPercent}
          onChange={settings.setConfidenceCiPercent}
          disabled={disabled}
        />
        <DeltaField
          value={settings.delta}
          onChange={settings.setDelta}
          disabled={disabled}
        />
      </div>
    </StickyPanelGroup>
  );
}
