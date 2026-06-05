import { useEffect, useRef, useState } from "react";
import type { CameraFrameState } from "./mathboxExecutionGraph";

function fmt3([x, y, z]: [number, number, number], digits = 4): string {
  return `[${x.toFixed(digits)}, ${y.toFixed(digits)}, ${z.toFixed(digits)}]`;
}

function fmtDir([x, y, z]: [number, number, number]): string {
  return `new THREE.Vector3(${x.toFixed(4)}, ${y.toFixed(4)}, ${z.toFixed(4)}).normalize()`;
}

/**
 * Live camera framing HUD for tuning `mathboxExecutionGraph` constants.
 * Sits beside {@link ExecutionGraphLegend} in the graph HUD stack.
 */
export function ExecutionGraphCameraOverlay({
  readFrame,
  active,
}: {
  readFrame: () => CameraFrameState | null;
  active: boolean;
}) {
  const [frame, setFrame] = useState<CameraFrameState | null>(null);
  const readFrameRef = useRef(readFrame);
  readFrameRef.current = readFrame;

  useEffect(() => {
    if (!active) {
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
  }, [active]);

  if (!active || !frame) return null;

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
      <p className="exec-graph__camera-hud-hint">FOV: [ / ] or Alt+wheel (zoom still dollies distance)</p>
      <pre className="exec-graph__camera-hud-snippet" aria-label="Suggested constants">
        {`const CAMERA_DIRECTION = ${fmtDir(frame.direction)};\nconst CAMERA_FOV = ${frame.fov.toFixed(2)};\n// after frameField, distance ≈ ${frame.distance.toFixed(4)}\n// controls.target = ${fmt3(frame.target)};`}
      </pre>
    </section>
  );
}
