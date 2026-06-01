import type { ComponentPropsWithoutRef, CSSProperties, ReactNode } from "react";
import { classNames } from "../../lib/format";
import "../../styles/flat-controls.css";

type StickyPanelSlotArea = "title" | "controls" | "controls-grid" | "controls-grid-inner" | "auxiliary" | "items" | "view" | "edit" | "chip";

type StickyPanelProps = ComponentPropsWithoutRef<"aside"> & {
};

type StickyPanelSlotProps = ComponentPropsWithoutRef<"div"> & {
  area?: StickyPanelSlotArea;
  hSpan?:  {
    row: number;
    col: number;
  };
  vSpan?: {
    row: number;
    col: number;
  };
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

export function StickyPanelCheckboxGroup({
  legend,
  children,
  className,
  hSpan,
  vSpan,
  inline = false,
}: {
  legend: string;
  children: ReactNode;
  className?: string;
  hSpan?: StickyPanelSlotProps["hSpan"];
  vSpan?: StickyPanelSlotProps["vSpan"];
  /** When true, omit the grid slot wrapper (for use inside a shared row). */
  inline?: boolean;
}) {
  const body = (
    <>
      <p className="sticky-panel-checkbox-group__legend">{legend}</p>
      <div className="sticky-panel-checkbox-group-container">{children}</div>
    </>
  );

  if (inline) {
    return <div className={classNames("sticky-panel-checkbox-group", className)}>{body}</div>;
  }

  return (
    <StickyPanelSlot className={classNames("sticky-panel-checkbox-group", className)} hSpan={hSpan} vSpan={vSpan}>
      {body}
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
