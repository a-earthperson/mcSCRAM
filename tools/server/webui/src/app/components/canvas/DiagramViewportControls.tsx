import type { RefObject } from "react";
import { DiagramFullscreenButton } from "../DiagramFullscreenButton";
import type { CanvasViewportConfig } from "./canvasViewport";

type DiagramViewportControlsProps = {
  frameRef: RefObject<HTMLDivElement | null>;
  viewport: CanvasViewportConfig;
  ariaLabel: string;
  onFit: () => void;
  onZoom: (factor: number) => void;
  onPan: (dx: number, dy: number) => void;
  onReset: () => void;
};

/** Grouped diagram toolbar: zen · fit · zoom · pan · reset, with chromatic clusters. */
export function DiagramViewportControls({
  frameRef,
  viewport,
  ariaLabel,
  onFit,
  onZoom,
  onPan,
  onReset,
}: DiagramViewportControlsProps) {
  const { zoomInFactor, zoomOutFactor, panStepPx } = viewport.button;

  return (
    <div className="fault-tree-controls" aria-label={ariaLabel}>
      <div className="fault-tree-control-group">
        <DiagramFullscreenButton targetRef={frameRef} />
      </div>
      <div className="fault-tree-control-group">
        <button className="fault-tree-control fit" type="button" onClick={onFit}>
          Fit
        </button>
      </div>
      <div className="fault-tree-control-group">
        <button
          className="fault-tree-control zoom"
          type="button"
          onClick={() => onZoom(zoomInFactor)}
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          className="fault-tree-control zoom"
          type="button"
          onClick={() => onZoom(zoomOutFactor)}
          aria-label="Zoom out"
        >
          −
        </button>
      </div>
      <div className="fault-tree-control-group">
        <button
          className="fault-tree-control pan"
          type="button"
          onClick={() => onPan(0, panStepPx)}
          aria-label="Pan up"
        >
          ↑
        </button>
        <button
          className="fault-tree-control pan"
          type="button"
          onClick={() => onPan(0, -panStepPx)}
          aria-label="Pan down"
        >
          ↓
        </button>
        <button
          className="fault-tree-control pan"
          type="button"
          onClick={() => onPan(panStepPx, 0)}
          aria-label="Pan left"
        >
          ←
        </button>
        <button
          className="fault-tree-control pan"
          type="button"
          onClick={() => onPan(-panStepPx, 0)}
          aria-label="Pan right"
        >
          →
        </button>
      </div>
      <div className="fault-tree-control-group">
        <button className="fault-tree-control reset-view" type="button" onClick={onReset}>
          Reset
        </button>
      </div>
    </div>
  );
}
