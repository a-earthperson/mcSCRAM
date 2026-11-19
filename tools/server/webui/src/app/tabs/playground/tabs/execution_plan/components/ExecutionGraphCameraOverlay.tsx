import { useEffect, useRef, useState } from "react";
import type { CameraFrameState } from "../render/executionGraph";

function fmt3([x, y, z]: [number, number, number], digits = 4): string {
  return `[${x.toFixed(digits)}, ${y.toFixed(digits)}, ${z.toFixed(digits)}]`;
}

/**
 * Live camera framing HUD for tuning `render/executionGraph` constants.
 * Sits beside {@link ExecutionGraphLegend} in the graph HUD stack.
 */
export function ExecutionGraphCameraOverlay({
  readFrame,
  active,
}: {
  readFrame: () => CameraFrameState | null;
  active: boolean;
}) {
  const [visible, setVisible] = useState(false);
  const [frame, setFrame] = useState<CameraFrameState | null>(null);
  const readFrameRef = useRef(readFrame);
  readFrameRef.current = readFrame;

  useEffect(() => {
    if (!active) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.key.toLowerCase() !== "c" || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.closest("input, textarea, select, [contenteditable='true']"))
      ) {
        return;
      }
      setVisible((show) => !show);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active]);

  useEffect(() => {
    if (!active || !visible) {
      setFrame(null);
      return undefined;
    }
    let raf = 0;
    const tick = () => {
      setFrame(readFrameRef.current());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, visible]);

  if (!active || !visible || !frame) return null;

  return (
    <section className="exec-graph__camera-hud" aria-label="Camera framing readout">
      <header className="exec-graph__camera-hud-head">camera frame</header>
      <dl className="exec-graph__camera-hud-grid">
        <div>
          <dt>zoom (distance)</dt>
          <dd>{frame.distance.toFixed(4)}</dd>
        </div>
        <div>
          <dt>zoom ratio</dt>
          <dd>{frame.zoomRatio.toFixed(4)}</dd>
        </div>
        <div>
          <dt>fov</dt>
          <dd>{frame.fov.toFixed(2)}&deg;</dd>
        </div>
        <div>
          <dt>direction</dt>
          <dd>{fmt3(frame.direction)}</dd>
        </div>
        <div>
          <dt>pan (target)</dt>
          <dd>{fmt3(frame.target)}</dd>
        </div>
        <div>
          <dt>position</dt>
          <dd>{fmt3(frame.position)}</dd>
        </div>
      </dl>
      <p className="exec-graph__camera-hud-hint">
        FOV: [ / ] or Alt+wheel (zoom still dollies distance) &middot; c toggles this panel
      </p>
      <pre className="exec-graph__camera-hud-snippet" aria-label="Suggested constants">
        {`export const CAMERA_PERSPECTIVE: CameraPerspectiveConfig = {\n  direction: ${fmt3(frame.direction)},\n  target: ${fmt3(frame.target)},\n  fov: ${frame.fov.toFixed(2)},\n  distance: ${frame.distance.toFixed(4)},\n};`}
      </pre>
    </section>
  );
}
