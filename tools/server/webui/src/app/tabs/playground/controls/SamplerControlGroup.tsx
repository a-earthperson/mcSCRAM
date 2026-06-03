import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/StickyPanel";
import { ConfidenceCiField } from "./ConfidenceCiField";
import { DeltaField } from "./DeltaField";
import { OverheadRatioField } from "./OverheadRatioField";
import { StopOnConvergenceField } from "./StopOnConvergenceField";
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
      <StopOnConvergenceField
        checked={settings.stopOnConvergence}
        onChange={settings.setStopOnConvergence}
        disabled={disabled}
      />
      <TrialsNField
        value={settings.trialsScalar}
        onChange={settings.setTrialsScalar}
        stopOnConvergence={settings.stopOnConvergence}
        disabled={disabled}
      />
      <OverheadRatioField
        value={settings.overheadRatio}
        onChange={settings.setOverheadRatio}
        disabled={disabled}
      />
      <ConfidenceCiField
        value={settings.confidenceCiPercent}
        onChange={settings.setConfidenceCiPercent}
        disabled={disabled || !settings.stopOnConvergence}
      />
      <DeltaField
        value={settings.delta}
        onChange={settings.setDelta}
        disabled={disabled || !settings.stopOnConvergence}
      />
    </StickyPanelGroup>
  );
}
