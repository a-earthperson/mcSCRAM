import type { RefObject } from "react";
import { classNames } from "../../lib/format";
import { useElementFullscreen } from "./useElementFullscreen";
import "../canvas/canvas.css";

type DiagramFullscreenButtonProps = {
  targetRef: RefObject<HTMLElement | null>;
  className?: string;
  labels?: { enter: string; exit: string };
};

/** Shared diagram chrome: native fullscreen toggle using canvas control styling. */
export function DiagramFullscreenButton({
  targetRef,
  className,
  labels = { enter: "ZEN", exit: "Exit" },
}: DiagramFullscreenButtonProps) {
  const { active, supported, toggle } = useElementFullscreen(targetRef);
  if (!supported) return null;

  return (
    <button
      type="button"
      className={classNames("fault-tree-control", "zen", className)}
      onClick={() => {
        void toggle();
      }}
      aria-pressed={active}
      aria-label={active ? "Exit zen view" : "Enter zen view"}
    >
      {active ? labels.exit : labels.enter}
    </button>
  );
}
