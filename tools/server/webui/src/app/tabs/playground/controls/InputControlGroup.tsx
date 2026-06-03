import type { StickyPanelSpan } from "../../../components/StickyPanel";
import { StickyPanelGroup } from "../../../components/StickyPanel";
import { Select } from "../../../ui";
import { OraclePField } from "./OraclePField";
import type { InputSettings } from "./useInputSettings";

export function InputControlGroup({
  selectedId,
  onSelectedIdChange,
  modelIds,
  blocked,
  inputSettings,
  hSpan,
  vSpan,
}: {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  modelIds: string[];
  blocked: boolean;
  inputSettings: InputSettings;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Input" className="sticky-panel-group--input" hSpan={hSpan} vSpan={vSpan}>
      <Select
        label="Model"
        value={selectedId}
        onChange={onSelectedIdChange}
        options={modelIds}
        disabled={blocked || modelIds.length === 0}
      />
      <OraclePField
        value={inputSettings.oracleP}
        onChange={inputSettings.setOracleP}
        disabled={blocked}
      />
    </StickyPanelGroup>
  );
}
