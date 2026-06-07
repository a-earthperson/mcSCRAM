import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./canvas.css";
import { DiagramViewportControls } from "./DiagramViewportControls";
import { formatInteger } from "../../lib/format";
import { EmptyState } from "../../components/ui";
import { FAULT_TREE_CANVAS_VIEWPORT } from "./canvasViewport";
import { resolveCanvasFitSize, useCanvasSurfaceSize } from "./canvasSurface";
import {
  fitLayoutView,
  resetViewTransform,
  zoomAtPoint,
} from "./canvasViewportTransform";
import { drawFaultTreeCanvas } from "./drawFaultTree";
import { layoutFaultTree } from "./layoutFaultTree";
import type { FaultTreePayload, ViewTransform, CanvasSize } from "./types";
import { useCanvasViewportGestures } from "./useCanvasViewportGestures";

const VIEWPORT = FAULT_TREE_CANVAS_VIEWPORT;

export type FaultTreeDiagramProps = {
  tree: FaultTreePayload;
};

export function FaultTreeDiagram({ tree }: FaultTreeDiagramProps) {
  const layout = useMemo(() => layoutFaultTree(tree), [tree]);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewRef = useRef<ViewTransform>(resetViewTransform(VIEWPORT.layout.defaultSize, VIEWPORT));
  const sizeRef = useRef<CanvasSize>({ width: 0, height: 0 });
  const didInitialFitRef = useRef(false);
  const [view, setViewState] = useState<ViewTransform>(() =>
    resetViewTransform(VIEWPORT.layout.defaultSize, VIEWPORT),
  );
  const { size, hasMeasured } = useCanvasSurfaceSize(canvasRef, VIEWPORT.layout);

  const setView = useCallback((update: (prev: ViewTransform) => ViewTransform) => {
    setViewState((prev) => {
      const next = update(prev);
      viewRef.current = next;
      return next;
    });
  }, []);

  viewRef.current = view;

  const viewportGestures = useCanvasViewportGestures({
    canvasRef,
    viewport: VIEWPORT,
    getView: () => viewRef.current,
    setView,
    enabled: layout.nodes.length > 0,
  });

  const fitView = useCallback(() => {
    setView(() =>
      fitLayoutView(
        layout.width,
        layout.height,
        resolveCanvasFitSize(canvasRef.current, sizeRef.current),
        VIEWPORT,
      ),
    );
  }, [layout.width, layout.height, setView]);

  const resetView = useCallback(() => {
    setView(() => resetViewTransform(sizeRef.current, VIEWPORT));
  }, [setView]);

  const zoomAtCenter = useCallback((factor: number) => {
    const currentSize = sizeRef.current;
    setView((prev) =>
      zoomAtPoint(prev, currentSize.width / 2, currentSize.height / 2, factor, VIEWPORT),
    );
  }, [setView]);

  const panView = useCallback((dx: number, dy: number) => {
    setView((prev) => ({ ...prev, offsetX: prev.offsetX + dx, offsetY: prev.offsetY + dy }));
  }, [setView]);

  useEffect(() => {
    if (!hasMeasured || size.width <= 0 || size.height <= 0) {
      return;
    }
    sizeRef.current = size;
  }, [hasMeasured, size]);

  useEffect(() => {
    didInitialFitRef.current = false;
  }, [tree]);

  useEffect(() => {
    if (didInitialFitRef.current || !hasMeasured || size.width <= 0 || size.height <= 0) {
      return;
    }
    fitView();
    didInitialFitRef.current = true;
  }, [tree, fitView, hasMeasured, size]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame || !hasMeasured || size.width <= 0 || size.height <= 0) return;
    drawFaultTreeCanvas(canvas, layout, view, size, frame);
  }, [hasMeasured, layout, size, view]);

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
        {...viewportGestures}
      />
      <div className="fault-tree-toolbar">
        <span>
          {formatInteger(gateCount)} gates · {eventSummary} · {formatInteger(tree.edges.length)} edges
        </span>
        <DiagramViewportControls
          frameRef={frameRef}
          viewport={VIEWPORT}
          ariaLabel="Fault tree viewport controls"
          onFit={fitView}
          onZoom={zoomAtCenter}
          onPan={panView}
          onReset={resetView}
        />
      </div>
    </div>
  );
}
