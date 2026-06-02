import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./canvas.css";
import { formatInteger } from "../../../lib/format";
import { EmptyState } from "../../ui";
import { FAULT_TREE_CANVAS_VIEWPORT, useCanvasFrameSize } from "./canvasViewport";
import {
  fitLayoutView,
  panViewTransform,
  resetViewTransform,
  wheelZoomAtCanvasPoint,
  zoomAtPoint,
} from "./canvasViewportTransform";
import { drawFaultTreeCanvas } from "./drawFaultTree";
import { layoutFaultTree } from "./layoutFaultTree";
import type { FaultTreePayload, ViewTransform } from "./types";
import { useNonPassiveWheel } from "./useNonPassiveWheel";

const VIEWPORT = FAULT_TREE_CANVAS_VIEWPORT;

export type FaultTreeDiagramProps = {
  tree: FaultTreePayload;
};

export function FaultTreeDiagram({ tree }: FaultTreeDiagramProps) {
  const layout = useMemo(() => layoutFaultTree(tree), [tree]);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);
  const sizeRef = useRef(VIEWPORT.layout.defaultSize);
  const didInitialFitRef = useRef(false);
  const [view, setView] = useState<ViewTransform>(() =>
    resetViewTransform(VIEWPORT.layout.defaultSize, VIEWPORT),
  );
  const { size, hasMeasured } = useCanvasFrameSize(frameRef, VIEWPORT.layout);

  const fitView = useCallback(() => {
    setView(fitLayoutView(layout.width, layout.height, sizeRef.current, VIEWPORT));
  }, [layout.width, layout.height]);

  const resetView = useCallback(() => {
    setView(resetViewTransform(sizeRef.current, VIEWPORT));
  }, []);

  const zoomAtCenter = useCallback((factor: number) => {
    const currentSize = sizeRef.current;
    setView((prev) =>
      zoomAtPoint(prev, currentSize.width / 2, currentSize.height / 2, factor, VIEWPORT),
    );
  }, []);

  const panView = useCallback((dx: number, dy: number) => {
    setView((prev) => panViewTransform(prev, dx, dy));
  }, []);

  const handleWheel = useCallback((event: WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setView((prev) => wheelZoomAtCanvasPoint(prev, canvas, event, VIEWPORT));
  }, []);

  useNonPassiveWheel(canvasRef, handleWheel, layout.nodes.length > 0);

  useEffect(() => {
    sizeRef.current = size;
  }, [size]);

  useEffect(() => {
    didInitialFitRef.current = false;
  }, [tree]);

  useEffect(() => {
    if (didInitialFitRef.current || !hasMeasured) {
      return;
    }
    fitView();
    didInitialFitRef.current = true;
  }, [tree, fitView, hasMeasured, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame) return;
    drawFaultTreeCanvas(canvas, layout, view, size, frame);
  }, [layout, size, view]);

  if (!layout.nodes.length) {
    return <EmptyState text="No renderable fault-tree nodes were found in the active model." />;
  }

  const gateCount = tree.nodes.filter((node) => node.kind === "gate").length;
  const basicEventCount = tree.nodes.filter((node) => node.kind === "basic_event").length;
  const houseEventCount = tree.nodes.filter((node) => node.kind === "house_event").length;
  const eventSummary = houseEventCount
    ? `${formatInteger(basicEventCount)} basic events · ${formatInteger(houseEventCount)} house events`
    : `${formatInteger(basicEventCount)} basic events`;

  return (
    <div className="fault-tree-frame" ref={frameRef}>
      <canvas
        ref={canvasRef}
        className="fault-tree-canvas"
        width={size.width}
        height={size.height}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, y: event.clientY, offsetX: view.offsetX, offsetY: view.offsetY };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          setView((prev) => ({
            ...prev,
            offsetX: drag.offsetX + event.clientX - drag.x,
            offsetY: drag.offsetY + event.clientY - drag.y,
          }));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
      />
      <div className="fault-tree-toolbar">
        <span>
          {formatInteger(gateCount)} gates · {eventSummary} · {formatInteger(tree.edges.length)} edges
        </span>
        <div className="fault-tree-controls" aria-label="Fault tree viewport controls">
          <button className="fault-tree-control fit" type="button" onClick={fitView}>Fit</button>
          <button className="fault-tree-control zoom" type="button" onClick={() => zoomAtCenter(VIEWPORT.button.zoomInFactor)} aria-label="Zoom in">+</button>
          <button className="fault-tree-control zoom" type="button" onClick={() => zoomAtCenter(VIEWPORT.button.zoomOutFactor)} aria-label="Zoom out">−</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(0, VIEWPORT.button.panStepPx)} aria-label="Pan up">↑</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(0, -VIEWPORT.button.panStepPx)} aria-label="Pan down">↓</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(VIEWPORT.button.panStepPx, 0)} aria-label="Pan left">←</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(-VIEWPORT.button.panStepPx, 0)} aria-label="Pan right">→</button>
          <button className="fault-tree-control reset-view" type="button" onClick={resetView}>Reset</button>
        </div>
      </div>
    </div>
  );
}
