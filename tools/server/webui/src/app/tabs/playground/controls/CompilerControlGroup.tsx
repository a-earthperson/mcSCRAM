import type { Dispatch, SetStateAction } from "react";
import {
  StickyPanelCheckbox,
  StickyPanelCheckboxGroup,
  StickyPanelGroup,
  type StickyPanelSpan,
} from "../../../components/StickyPanel";
import type { CompileDagOptions } from "../../../faultTreeApi";

export function CompilerControlGroup({
  compileOptions,
  onCompileOptionsChange,
  hSpan,
  vSpan,
}: {
  compileOptions: CompileDagOptions;
  onCompileOptionsChange: Dispatch<SetStateAction<CompileDagOptions>>;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  return (
    <StickyPanelGroup legend="Compiler" className="sticky-panel-group--compiler" hSpan={hSpan} vSpan={vSpan}>
      <CompileLevelSelect
        value={compileOptions.compilationLevel}
        onChange={(compilationLevel) => onCompileOptionsChange((current) => ({ ...current, compilationLevel }))}
      />
      <div className="sticky-panel-group__fields-row">
        <StickyPanelCheckboxGroup legend="Expand" inline>
          <StickyPanelCheckbox
            id="playground-expand-atleast"
            label="K/N"
            checked={compileOptions.expandAtleastGates}
            onChange={(expandAtleastGates) => onCompileOptionsChange((current) => ({ ...current, expandAtleastGates }))}
          />
          <StickyPanelCheckbox
            id="playground-expand-xor"
            label="XOR"
            checked={compileOptions.expandXorGates}
            onChange={(expandXorGates) => onCompileOptionsChange((current) => ({ ...current, expandXorGates }))}
          />
        </StickyPanelCheckboxGroup>
        <StickyPanelCheckboxGroup legend="Merge" inline>
          <StickyPanelCheckbox
            id="playground-merge-atleast"
            label="K/N"
            checked={compileOptions.recognizeAtleastGroups}
            onChange={(recognizeAtleastGroups) => onCompileOptionsChange((current) => ({ ...current, recognizeAtleastGroups }))}
          />
          <StickyPanelCheckbox
            id="playground-merge-xor"
            label="XOR"
            checked={compileOptions.recognizeXorGroups}
            onChange={(recognizeXorGroups) => onCompileOptionsChange((current) => ({ ...current, recognizeXorGroups }))}
          />
        </StickyPanelCheckboxGroup>
      </div>
    </StickyPanelGroup>
  );
}

function CompileLevelSelect({
  value,
  onChange,
}: {
  value: number;
  onChange: (level: number) => void;
}) {
  return (
    <label>
      Level
      <select value={String(value)} onChange={(event) => onChange(Number(event.target.value))}>
        {Array.from({ length: 9 }, (_, level) => (
          <option key={level} value={level}>
            C{level}
          </option>
        ))}
      </select>
    </label>
  );
}
