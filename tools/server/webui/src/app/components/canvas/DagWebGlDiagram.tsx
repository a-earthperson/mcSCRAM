import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatInteger } from "../../../lib/format";
import { EmptyState } from "../../ui";
import { DagWebGlRenderer } from "./drawDagWebGl";
import type { CanvasSize, DagBounds, DagPayload, ViewTransform } from "./types";

const DEFAULT_SIZE: CanvasSize = { width: 900, height: 620 };
const MIN_SCALE = 0.000001;
const MAX_SCALE = 512;
const FIT_PADDING = 1.2;

export type DagWebGlDiagramProps = {
  dag: DagPayload;
  note?: string;
};

export function DagWebGlDiagram({ dag, note }: DagWebGlDiagramProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<DagWebGlRenderer | null>(null);
  const rafRef = useRef<number | null>(null);
  const sizeRef = useRef<CanvasSize>(DEFAULT_SIZE);
  const viewRef = useRef<ViewTransform>(defaultView(DEFAULT_SIZE));
  const dragRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);
  const [size, setSize] = useState<CanvasSize>(DEFAULT_SIZE);
  const [renderError, setRenderError] = useState<string | null>(null);

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
    applyView(fitDagView(dag.bounds, sizeRef.current));
  }, [applyView, dag.bounds]);

  const resetView = useCallback(() => {
    applyView(defaultView(sizeRef.current));
  }, [applyView]);

  const panView = useCallback((dx: number, dy: number) => {
    const view = viewRef.current;
    applyView({ ...view, offsetX: view.offsetX + dx, offsetY: view.offsetY + dy });
  }, [applyView]);

  const zoomAt = useCallback((px: number, py: number, factor: number) => {
    const view = viewRef.current;
    const nextScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.scale * factor));
    const worldX = (px - view.offsetX) / view.scale;
    const worldY = (py - view.offsetY) / view.scale;
    applyView({
      scale: nextScale,
      offsetX: px - worldX * nextScale,
      offsetY: py - worldY * nextScale,
    });
  }, [applyView]);

  const zoomAtCenter = useCallback((factor: number) => {
    const currentSize = sizeRef.current;
    zoomAt(currentSize.width / 2, currentSize.height / 2, factor);
  }, [zoomAt]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(320, entry.contentRect.width),
        height: Math.max(420, entry.contentRect.height),
      });
    });
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

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
      viewRef.current = fitDagView(dag.bounds, sizeRef.current);
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
      <div className="fault-tree-toolbar">
        <span>
          WebGL2 DAG · {formatInteger(counts.gates)} gates · {formatInteger(counts.basicEvents)} events · {formatInteger(dag.edges.count)} edges
        </span>
        <div className="fault-tree-controls" aria-label="WebGL DAG viewport controls">
          <button className="fault-tree-control fit" type="button" onClick={fitView}>Fit</button>
          <button className="fault-tree-control zoom" type="button" onClick={() => zoomAtCenter(1.25)} aria-label="Zoom in">+</button>
          <button className="fault-tree-control zoom" type="button" onClick={() => zoomAtCenter(0.8)} aria-label="Zoom out">−</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(0, 90)} aria-label="Pan up">↑</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(0, -90)} aria-label="Pan down">↓</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(90, 0)} aria-label="Pan left">←</button>
          <button className="fault-tree-control pan" type="button" onClick={() => panView(-90, 0)} aria-label="Pan right">→</button>
          <button className="fault-tree-control reset-view" type="button" onClick={resetView}>Reset</button>
        </div>
      </div>
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
        onWheel={(event) => {
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          zoomAt(event.clientX - rect.left, event.clientY - rect.top, Math.exp(-event.deltaY * 0.0012));
        }}
      />
      <p className="fault-tree-note">
        {note ?? "Instanced WebGL2 renderer: triangles and line buffers are static; pan and zoom update shader uniforms only."}
      </p>
    </div>
  );
}

function defaultView(size: CanvasSize): ViewTransform {
  return { scale: 64, offsetX: size.width / 2, offsetY: 80 };
}

function fitDagView(bounds: DagBounds, size: CanvasSize): ViewTransform {
  const paddedWidth = Math.max(bounds.width + FIT_PADDING * 2, 1);
  const paddedHeight = Math.max(bounds.height + FIT_PADDING * 2, 1);
  const scale = Math.max(
    MIN_SCALE,
    Math.min(MAX_SCALE, Math.min(size.width / paddedWidth, size.height / paddedHeight) * 0.92),
  );
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  return {
    scale,
    offsetX: size.width / 2 - centerX * scale,
    offsetY: size.height / 2 - centerY * scale,
  };
}
