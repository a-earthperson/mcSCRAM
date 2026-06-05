import type { StickyPanelSpan } from "../../../components/ui/StickyPanel";
import { StickyPanelGroup } from "../../../components/ui/StickyPanel";
import { Select } from "../../../components";
import { OraclePField } from "./OraclePField";
import type { InputSettings } from "./useInputSettings";

export function InputControlGroup({
  selectedId,
  onSelectedIdChange,
  modelIds,
  blocked,
  disabled = false,
  inputSettings,
  hSpan,
  vSpan,
}: {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  modelIds: string[];
  blocked: boolean;
  disabled?: boolean;
  inputSettings: InputSettings;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Input" className="sticky-panel-group--input" hSpan={hSpan} vSpan={vSpan}>
      <InputControlFields
        selectedId={selectedId}
        onSelectedIdChange={onSelectedIdChange}
        modelIds={modelIds}
        blocked={blocked}
        disabled={disabled}
        inputSettings={inputSettings}
      />
    </StickyPanelGroup>
  );
}

export function InputControlFields({
  selectedId,
  onSelectedIdChange,
  modelIds,
  blocked,
  disabled = false,
  inputSettings,
}: {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  modelIds: string[];
  blocked: boolean;
  disabled?: boolean;
  inputSettings: InputSettings;
}) {
  return (
    <>
      <Select
        label="Model"
        value={selectedId}
        onChange={onSelectedIdChange}
        options={modelIds}
        disabled={disabled || blocked || modelIds.length === 0}
      />
      <OraclePField
        value={inputSettings.oracleP}
        onChange={inputSettings.setOracleP}
        disabled={disabled || blocked}
      />
    </>
  );
}
