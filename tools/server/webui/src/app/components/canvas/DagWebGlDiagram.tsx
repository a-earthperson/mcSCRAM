import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatInteger } from "../../lib/format";
import { DiagramViewportControls } from "./DiagramViewportControls";
import { EmptyState } from "../../components/ui";
import { DAG_CANVAS_VIEWPORT } from "./canvasViewport";
import { resolveCanvasFitSize, useCanvasSurfaceSize } from "./canvasSurface";
import {
  fitBoundsView,
  resetViewTransform,
  zoomAtPoint,
} from "./canvasViewportTransform";
import { DagWebGlRenderer } from "./drawDagWebGl";
import type { DagPayload, ViewTransform, CanvasSize } from "./types";
import { useCanvasViewportGestures } from "./useCanvasViewportGestures";

const VIEWPORT = DAG_CANVAS_VIEWPORT;

export type DagWebGlDiagramProps = {
  dag: DagPayload;
};

export function DagWebGlDiagram({ dag }: DagWebGlDiagramProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<DagWebGlRenderer | null>(null);
  const rafRef = useRef<number | null>(null);
  const viewRef = useRef<ViewTransform>(resetViewTransform(VIEWPORT.layout.defaultSize, VIEWPORT));
  const sizeRef = useRef<CanvasSize>({ width: 0, height: 0 });
  const didInitialFitRef = useRef(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const { size, hasMeasured } = useCanvasSurfaceSize(canvasRef, VIEWPORT.layout);

  const counts = useMemo(() => {
    const gates = dag.counts?.gate ?? dag.nodes.type.filter((typeIndex) => typeIndex > 1).length;
    const basicEvents = dag.counts?.basic_event ?? dag.nodes.type.filter((typeIndex) => typeIndex === 0).length;
    return { gates, basicEvents };
  }, [dag]);

  const scheduleDraw = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      rendererRef.current?.draw(sizeRef.current, viewRef.current);
    });
  }, []);

  const applyView = useCallback((nextView: ViewTransform) => {
    viewRef.current = nextView;
    scheduleDraw();
  }, [scheduleDraw]);

  const setView = useCallback((update: (prev: ViewTransform) => ViewTransform) => {
    applyView(update(viewRef.current));
  }, [applyView]);

  const viewportGestures = useCanvasViewportGestures({
    canvasRef,
    viewport: VIEWPORT,
    getView: () => viewRef.current,
    setView,
    enabled: dag.nodes.count > 0 && renderError === null,
  });

  const fitView = useCallback(() => {
    applyView(fitBoundsView(dag.bounds, resolveCanvasFitSize(canvasRef.current, sizeRef.current), VIEWPORT));
  }, [applyView, dag.bounds]);

  const resetView = useCallback(() => {
    applyView(resetViewTransform(sizeRef.current, VIEWPORT));
  }, [applyView]);

  const panView = useCallback((dx: number, dy: number) => {
    const view = viewRef.current;
    applyView({ ...view, offsetX: view.offsetX + dx, offsetY: view.offsetY + dy });
  }, [applyView]);

  const zoomAt = useCallback((px: number, py: number, factor: number) => {
    applyView(zoomAtPoint(viewRef.current, px, py, factor, VIEWPORT));
  }, [applyView]);

  const zoomAtCenter = useCallback((factor: number) => {
    const currentSize = sizeRef.current;
    zoomAt(currentSize.width / 2, currentSize.height / 2, factor);
  }, [zoomAt]);

  useEffect(() => {
    if (!hasMeasured || size.width <= 0 || size.height <= 0) {
      return;
    }
    sizeRef.current = size;
    rendererRef.current?.resize(size);
    scheduleDraw();
  }, [hasMeasured, scheduleDraw, size]);

  useEffect(() => {
    const frame = frameRef.current;
    const canvas = canvasRef.current;
    if (!frame || !canvas) return undefined;

    let renderer: DagWebGlRenderer | null = null;
    try {
      renderer = new DagWebGlRenderer(canvas, dag);
      rendererRef.current = renderer;
      renderer.resize(sizeRef.current);
      setRenderError(null);
      scheduleDraw();
    } catch (err) {
      rendererRef.current = null;
      setRenderError(err instanceof Error ? err.message : String(err));
    }

    return () => {
      rendererRef.current = null;
      renderer?.destroy();
    };
  }, [dag, scheduleDraw]);

  useEffect(() => {
    didInitialFitRef.current = false;
  }, [dag]);

  useEffect(() => {
    if (
      didInitialFitRef.current ||
      renderError ||
      !rendererRef.current ||
      !hasMeasured ||
      size.width <= 0 ||
      size.height <= 0
    ) {
      return;
    }
    fitView();
    didInitialFitRef.current = true;
  }, [dag, fitView, hasMeasured, renderError, size]);

  useEffect(() => () => {
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  if (!dag.nodes.count) {
    return <EmptyState text="No renderable DAG vertices were found in the active model." />;
  }

  return (
    <div className="fault-tree-frame dag-webgl-frame" ref={frameRef}>
      {renderError ? <EmptyState text={renderError} /> : null}
      <canvas
        ref={canvasRef}
        className="fault-tree-canvas dag-webgl-canvas"
        hidden={renderError !== null}
        {...viewportGestures}
      />
      <div className="fault-tree-toolbar">
        <span>
          {formatInteger(counts.gates)} gates · {formatInteger(counts.basicEvents)} events · {formatInteger(dag.edges.count)} edges
        </span>
        <DiagramViewportControls
          frameRef={frameRef}
          viewport={VIEWPORT}
          ariaLabel="DAG viewport controls"
          onFit={fitView}
          onZoom={zoomAtCenter}
          onPan={panView}
          onReset={resetView}
        />
      </div>
    </div>
  );
}
