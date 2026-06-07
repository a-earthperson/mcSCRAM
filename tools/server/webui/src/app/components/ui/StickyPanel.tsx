import type { ComponentPropsWithoutRef, CSSProperties, ReactNode } from "react";
import { classNames } from "../../lib/format";
import "../../styles/flat-controls.css";
import "../../styles/sticky-panel-group.css";

type StickyPanelSlotArea = "title" | "controls" | "controls-grid" | "controls-grid-inner" | "auxiliary" | "items" | "view" | "edit" | "chip";

type StickyPanelProps = ComponentPropsWithoutRef<"aside"> & {
};

export type StickyPanelSpan = {
  row: number;
  col: number;
};

type StickyPanelSlotProps = ComponentPropsWithoutRef<"div"> & {
  area?: StickyPanelSlotArea;
  hSpan?: StickyPanelSpan;
  vSpan?: StickyPanelSpan;
};

export function StickyPanel({ children, className, ...props }: StickyPanelProps) {
  return (
    <aside className={classNames("panel", "sticky-panel", `sticky-panel__controls-grid`, className)} {...props}>
      {children}
    </aside>
  );
}

export function StickyPanelCheckbox({
  id,
  label,
  checked,
  onChange,
  disabled = false,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="sticky-panel-checkbox">
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <label htmlFor={id}>{label}</label>
    </div>
  );
}

/** Scoped control region (fieldset) placed on the sticky-panel grid. */
export function StickyPanelGroup({
  legend,
  legendHint,
  children,
  className,
  hSpan,
  vSpan,
  empty = false,
}: {
  legend: string;
  legendHint?: ReactNode;
  children?: ReactNode;
  className?: string;
  hSpan?: StickyPanelSpan;
  vSpan?: StickyPanelSpan;
  /** Reserve layout space when controls are not wired yet. */
  empty?: boolean;
}) {
  return (
    <StickyPanelSlot className={classNames("sticky-panel-group", empty && "sticky-panel-group--empty", className)} hSpan={hSpan} vSpan={vSpan}>
      <fieldset className="sticky-panel-group__fieldset">
        <legend className="sticky-panel-group__legend">
          <span className="sticky-panel-group__legend-text">{legend}</span>
          {legendHint}
        </legend>
        <div className="sticky-panel-group__body">{empty ? null : children}</div>
      </fieldset>
    </StickyPanelSlot>
  );
}

export function StickyPanelSlot({ area = "chip", children, className, hSpan, vSpan, style, ...props }: StickyPanelSlotProps) {
  const spanStyle: CSSProperties = {
    ...(hSpan && {
      "--sticky-panel-h-col": hSpan.col,
      "--sticky-panel-h-row": hSpan.row,
    }),
    ...(vSpan && {
      "--sticky-panel-v-col": vSpan.col,
      "--sticky-panel-v-row": vSpan.row,
    }),
    ...(area === "controls-grid-inner" && {
      "--sticky-panel-inner-rows": vSpan?.row,
    }),
    ...style,
  };

  return (
    <div className={classNames("sticky-panel__slot", `sticky-panel__${area}`, className)} style={spanStyle} {...props}>
      {children}
    </div>
  );
}
