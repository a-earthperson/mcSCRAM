import { useEffect, useMemo, useRef, useState } from "react";
import { DiagramFullscreenButton } from "../../../../components/DiagramFullscreenButton";
import { usePlotTheme } from "../../../../../components/plotTokens";
import { classNames } from "../../../../../lib/format";
import { ExecutionFallbackMap } from "./ExecutionFallbackMap";
import { ExecutionGraphCameraOverlay } from "./ExecutionGraphCameraOverlay";
import { ExecutionGraphLegend } from "./ExecutionGraphLegend";
import {
  createExecutionGraph,
  type ExecutionGraphController,
} from "../render/executionGraph";
import type { ExecutionScene } from "../scene/executionScene";
import type { ExecutionModel } from "../sycl";

type GraphStatus = "webgl" | "fallback";

export function ExecutionGraph({
  scene,
  model,
  selectedId,
  onSelect,
}: {
  scene: ExecutionScene;
  model: ExecutionModel;
  selectedId: number | null;
  onSelect: (partitionId: number | null) => void;
}) {
  const { palette, scheme } = usePlotTheme();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<ExecutionGraphController | null>(null);
  const selectedIdRef = useRef<number | null>(selectedId);
  const [status, setStatus] = useState<GraphStatus>("webgl");
  const sceneOptions = useMemo(() => ({ palette }), [palette]);
  
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  // (Re)build the MathBox scene when the plan or theme changes. Selection is
  // handled imperatively afterwards so it never triggers a full rebuild.
  useEffect(() => {
    const container = stageRef.current;
    if (!container) return undefined;

    let cancelled = false;
    let controller: ExecutionGraphController | null = null;
    try {
      controller = createExecutionGraph({ container, model, sceneOptions });
    } catch (error) {
      if (import.meta.env.DEV) {
        console.error("[ExecutionGraph] WebGL init failed:", error);
      }
      controller = null;
    }

    if (cancelled) {
      controller?.dispose();
      return undefined;
    }

    controllerRef.current = controller;
    setStatus(controller ? "webgl" : "fallback");
    controller?.resize();
    if (controller && selectedIdRef.current !== null) {
      controller.setSelected(selectedIdRef.current);
    }

    return () => {
      cancelled = true;
      controllerRef.current = null;
      controller?.dispose();
    };
    // Rebuilt when the hydrated model changes; scheme drives light/dark re-theme.
  }, [model, palette, scheme]);

  useEffect(() => {
    if (status !== "webgl") return;
    controllerRef.current?.setSelected(selectedId);
  }, [selectedId, status]);

  useEffect(() => {
    if (status !== "webgl") return undefined;
    const stage = stageRef.current;
    if (!stage) return undefined;
    const observer = new ResizeObserver(() => {
      controllerRef.current?.resize();
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [status]);

  return (
    <div
      ref={rootRef}
      className={classNames("exec-graph", status === "fallback" && "exec-graph--fallback")}
    >
      <div className="exec-graph__stage" ref={stageRef} hidden={status === "fallback"} aria-hidden={status === "fallback"} />
      {status === "fallback" ? (
        <ExecutionFallbackMap scene={scene} selectedId={selectedId} onSelect={onSelect} />
      ) : null}

      <div className="exec-graph__overlay" aria-hidden="true">
        <div className="exec-graph__axis exec-graph__axis--layer">execution layer &rarr;</div>
        <div className="exec-graph__axis exec-graph__axis--work">&uarr; log&#8321;&#8320; launched work-items</div>
      </div>

      <div className="exec-graph__hud">
        <div className="exec-graph__controls" aria-label="Execution graph view controls">
          <DiagramFullscreenButton targetRef={rootRef} />
        </div>
        <ExecutionGraphCameraOverlay
          active={status === "webgl"}
          readFrame={() => controllerRef.current?.getCameraFrame() ?? null}
        />
        <ExecutionGraphLegend kinds={scene.kinds} palette={palette} />
      </div>

      {status === "webgl" ? (
        <p className="exec-graph__hint" aria-hidden="true">drag to orbit &middot; scroll to zoom</p>
      ) : null}
    </div>
  );
}
