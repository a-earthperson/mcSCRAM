import type { ComponentPropsWithoutRef, CSSProperties } from "react";
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
