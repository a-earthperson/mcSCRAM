import { useId, useMemo } from "react";
import type { StickyPanelSpan } from "../../../components/ui/StickyPanel";
import { StickyPanelGroup } from "../../../components/ui/StickyPanel";
import { classNames } from "../../../lib/format";
import { OraclePField } from "./OraclePField";
import type { InputSettings } from "./useInputSettings";
import {
  buildModelPathIndex,
  displayedModelSelection,
  formatExtensionBadge,
  modelIdAfterFormatChange,
  modelIdAfterSystemChange,
  type CatalogModelRef,
} from "./modelPath";

export function InputControlGroup({
  selectedId,
  onSelectedIdChange,
  models,
  blocked,
  disabled = false,
  inputSettings,
  hSpan,
  vSpan,
}: {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  models: readonly CatalogModelRef[];
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
        models={models}
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
  models,
  blocked,
  disabled = false,
  inputSettings,
}: {
  selectedId: string;
  onSelectedIdChange: (id: string) => void;
  models: readonly CatalogModelRef[];
  blocked: boolean;
  disabled?: boolean;
  inputSettings: InputSettings;
}) {
  const index = useMemo(() => buildModelPathIndex(models), [models]);
  const selection = useMemo(() => displayedModelSelection(index, selectedId), [index, selectedId]);
  const empty = models.length === 0;
  const selectsDisabled = disabled || blocked || empty;
  const systemId = useId();
  const formatId = useId();
  const modelSelectId = useId();
  const extensionBadge = formatExtensionBadge(selection.extension);
  const formatExtTitle = selection.formatExtensions.length
    ? selection.formatExtensions.map((ext) => ext.toUpperCase()).join(" · ")
    : undefined;

  return (
    <>
      <LabeledSelect
        id={systemId}
        label="System"
        value={empty ? "" : selection.system}
        disabled={selectsDisabled}
        options={selection.systems.map((system) => ({ value: system, label: system }))}
        onChange={(system) => onSelectedIdChange(modelIdAfterSystemChange(index, selectedId, system))}
      />
      <LabeledSelect
        id={formatId}
        label="Format"
        value={empty ? "" : selection.format}
        disabled={selectsDisabled}
        options={selection.formats.map((format) => ({ value: format, label: format }))}
        onChange={(format) => onSelectedIdChange(modelIdAfterFormatChange(index, selectedId, format))}
        badge={extensionBadge || undefined}
        badgeTitle={formatExtTitle}
      />
      <LabeledSelect
        id={modelSelectId}
        label="Model"
        className="pg-input-select--model"
        value={empty || !selection.models.some((option) => option.modelId === selectedId) ? "" : selectedId}
        disabled={selectsDisabled}
        options={selection.models.map((option) => ({ value: option.modelId, label: option.label }))}
        onChange={onSelectedIdChange}
      />
      <OraclePField
        value={inputSettings.oracleP}
        onChange={inputSettings.setOracleP}
        disabled={disabled || blocked}
      />
    </>
  );
}

function LabeledSelect({
  id,
  label,
  value,
  options,
  onChange,
  disabled,
  badge,
  badgeTitle,
  className,
}: {
  id: string;
  label: string;
  value: string;
  options: readonly { value: string; label: string }[];
  onChange: (value: string) => void;
  disabled: boolean;
  badge?: string;
  badgeTitle?: string;
  className?: string;
}) {
  return (
    <div className={classNames("sticky-panel-group__number-field", "pg-input-select", className)}>
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor={id}>
          {label}
        </label>
        {badge ? (
          <span className="pg-input-format-ext" id={`${id}-ext`} title={badgeTitle}>
            {badge}
          </span>
        ) : null}
      </div>
      <select
        id={id}
        value={value}
        disabled={disabled}
        aria-describedby={badge ? `${id}-ext` : undefined}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
