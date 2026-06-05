import type { ChangeEvent, Dispatch, SetStateAction } from "react";
import { FieldHintTrigger } from "../../../components/HintTooltip";
import { StickyPanelGroup, type StickyPanelSpan } from "../../../components/StickyPanel";
import type { CompileDagOptions } from "../../../api/faultTreeApi";
import {
  clampCompilationLevel,
  COMPILATION_LEVEL_FIELD_HINT,
  COMPILATION_LEVEL_OPTIONS,
  compilationLevelOption,
} from "./compilationLevel";

export function CompilerControlGroup({
  compileOptions,
  onCompileOptionsChange,
  disabled = false,
  hSpan,
  vSpan,
}: {
  compileOptions: CompileDagOptions;
  onCompileOptionsChange: Dispatch<SetStateAction<CompileDagOptions>>;
  disabled?: boolean;
  hSpan: StickyPanelSpan;
  vSpan: StickyPanelSpan;
}) {
  const patch = (key: keyof CompileDagOptions) => (event: ChangeEvent<HTMLInputElement>) =>
    onCompileOptionsChange((current) => ({ ...current, [key]: event.target.checked }));

  return (
    <StickyPanelGroup legend="Compiler" className="sticky-panel-group--compiler" hSpan={hSpan} vSpan={vSpan}>
      <CompileLevelSelect
        value={compileOptions.compilationLevel}
        onChange={(compilationLevel) => onCompileOptionsChange((current) => ({ ...current, compilationLevel }))}
        disabled={disabled}
      />
      <div className="pg-matrix" role="group" aria-label="Connective normalization">
        <span className="pg-matrix__caption" aria-hidden="true">Nodes</span>
        <span className="pg-matrix__col" aria-hidden="true">K/N</span>
        <span className="pg-matrix__col" aria-hidden="true">XOR</span>
        <span className="pg-matrix__col" aria-hidden="true">NUL</span>
        <span className="pg-matrix__col" aria-hidden="true">NOT</span>

        <span className="pg-matrix__row" aria-hidden="true">Replace</span>
        <MatrixCheckbox
          label="Replace K/N (ATLEAST) gates"
          checked={compileOptions.compilationLevel >= 6 || compileOptions.expandAtleastGates}
          disabled={disabled || compileOptions.compilationLevel >= 6}
          onChange={patch("expandAtleastGates")}
        />
        <MatrixCheckbox
          label="Replace XOR gates"
          checked={compileOptions.compilationLevel >= 6 || compileOptions.expandXorGates}
          disabled={disabled || compileOptions.compilationLevel >= 6}
          onChange={patch("expandXorGates")}
        />
        <MatrixCheckbox
          label="Replace NULL gates (remove from PDAG)"
          disabled={disabled || compileOptions.compilationLevel >= 1 || compileOptions.expandAtleastGates || compileOptions.expandXorGates}
          checked={!compileOptions.keepNullGates || compileOptions.compilationLevel >= 1 || compileOptions.expandAtleastGates || compileOptions.expandXorGates}
          onChange={(event) =>
            onCompileOptionsChange((current) => ({ ...current, keepNullGates: !event.target.checked }))
          }
        />
        <MatrixCheckbox
          label="Replace NOT gates"
          disabled={true}
          checked={compileOptions.compilationLevel >= 1 || compileOptions.expandAtleastGates || compileOptions.expandXorGates}
          onChange={() => {}}
        />

        <span className="pg-matrix__row" aria-hidden="true">Merge</span>
        <MatrixCheckbox
          label="Merge K/N (ATLEAST) groups"
          checked={compileOptions.compilationLevel < 6 ? compileOptions.recognizeAtleastGroups : false}
          disabled={disabled || compileOptions.compilationLevel >= 6}
          onChange={patch("recognizeAtleastGroups")}
        />
        <MatrixCheckbox
          label="Merge XOR groups"
          checked={compileOptions.compilationLevel < 6 ? compileOptions.recognizeXorGroups : false}
          disabled={disabled || compileOptions.compilationLevel >= 6}
          onChange={patch("recognizeXorGroups")}
        />
        <MatrixCheckbox
          label="Merge NULL groups (not applicable)"
          checked={false}
          disabled
          onChange={() => {}}
        />
        <MatrixCheckbox label="Merge NOT groups (not applicable)" disabled />
      </div>
    </StickyPanelGroup>
  );
}

type MatrixCheckboxProps =
  | {
      label: string;
      disabled?: boolean;
      checked: boolean;
      onChange: (event: ChangeEvent<HTMLInputElement>) => void;
    }
  | {
      label: string;
      disabled?: boolean;
      checked?: never;
      onChange?: never;
    };

function MatrixCheckbox({ label, disabled = false, ...props }: MatrixCheckboxProps) {
  const wired = "checked" in props;

  return (
    <label className="pg-matrix__cell">
      <input
        type="checkbox"
        disabled={disabled}
        aria-label={label}
        {...(wired
          ? { checked: props.checked, onChange: props.onChange }
          : { defaultChecked: false, readOnly: true })}
      />
    </label>
  );
}

function CompileLevelSelect({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (level: number) => void;
  disabled?: boolean;
}) {
  const level = clampCompilationLevel(value);
  const selected = compilationLevelOption(level);

  return (
    <div className="sticky-panel-group__number-field pg-compile-level">
      <div className="sticky-panel-group__number-field-label-row">
        <label className="sticky-panel-group__number-field-label" htmlFor="playground-compilation-level">
          Level
        </label>
        <FieldHintTrigger label="Compilation level help" content={COMPILATION_LEVEL_FIELD_HINT} />
      </div>
      <div className="pg-compile-level__row">
        <select
          id="playground-compilation-level"
          value={String(level)}
          aria-describedby="playground-compilation-level-summary"
          disabled={disabled}
          onChange={(event) => onChange(clampCompilationLevel(Number(event.target.value)))}
        >
          {COMPILATION_LEVEL_OPTIONS.map((option) => (
            <option key={option.level} value={option.level}>
              {option.label}
            </option>
          ))}
        </select>
        <FieldHintTrigger
          label={`Summary for ${selected.label}`}
          content={selected.summary}
        />
        <span id="playground-compilation-level-summary" className="visually-hidden">
          {selected.summary}
        </span>
      </div>
    </div>
  );
}
