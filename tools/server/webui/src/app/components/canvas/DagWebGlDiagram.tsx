import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatInteger } from "../../lib/format";
import { DiagramViewportControls } from "./DiagramViewportControls";
import { EmptyState } from "../../ui";
import { DAG_CANVAS_VIEWPORT, useCanvasFrameSize } from "./canvasViewport";
import {
  fitBoundsView,
  panViewTransform,
  resetViewTransform,
  wheelZoomAtCanvasPoint,
  zoomAtPoint,
} from "./canvasViewportTransform";
import { DagWebGlRenderer } from "./drawDagWebGl";
import type { DagPayload, ViewTransform } from "./types";
import { useNonPassiveWheel } from "./useNonPassiveWheel";

const VIEWPORT = DAG_CANVAS_VIEWPORT;

export type DagWebGlDiagramProps = {
  dag: DagPayload;
};

export function DagWebGlDiagram({ dag }: DagWebGlDiagramProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<DagWebGlRenderer | null>(null);
  const rafRef = useRef<number | null>(null);
  const sizeRef = useRef(VIEWPORT.layout.defaultSize);
  const viewRef = useRef<ViewTransform>(resetViewTransform(VIEWPORT.layout.defaultSize, VIEWPORT));
  const dragRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);
  const didInitialFitRef = useRef(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const { size, hasMeasured } = useCanvasFrameSize(frameRef, VIEWPORT.layout);

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

  const fitView = useCallback(() => {
    applyView(fitBoundsView(dag.bounds, sizeRef.current, VIEWPORT));
  }, [applyView, dag.bounds]);

  const resetView = useCallback(() => {
    applyView(resetViewTransform(sizeRef.current, VIEWPORT));
  }, [applyView]);

  const panView = useCallback((dx: number, dy: number) => {
    applyView(panViewTransform(viewRef.current, dx, dy));
  }, [applyView]);

  const zoomAt = useCallback((px: number, py: number, factor: number) => {
    applyView(zoomAtPoint(viewRef.current, px, py, factor, VIEWPORT));
  }, [applyView]);

  const zoomAtCenter = useCallback((factor: number) => {
    const currentSize = sizeRef.current;
    zoomAt(currentSize.width / 2, currentSize.height / 2, factor);
  }, [zoomAt]);

  const handleWheel = useCallback((event: WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    applyView(wheelZoomAtCanvasPoint(viewRef.current, canvas, event, VIEWPORT));
  }, [applyView]);

  useNonPassiveWheel(canvasRef, handleWheel, dag.nodes.count > 0 && renderError === null);

  useEffect(() => {
    sizeRef.current = size;
    rendererRef.current?.resize(size);
    scheduleDraw();
  }, [scheduleDraw, size]);

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
    if (didInitialFitRef.current || renderError || !rendererRef.current || !hasMeasured) {
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
        width={size.width}
        height={size.height}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const view = viewRef.current;
          dragRef.current = { x: event.clientX, y: event.clientY, offsetX: view.offsetX, offsetY: view.offsetY };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          applyView({
            ...viewRef.current,
            offsetX: drag.offsetX + event.clientX - drag.x,
            offsetY: drag.offsetY + event.clientY - drag.y,
          });
        }}
        onPointerUp={(event) => {
          dragRef.current = null;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
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
