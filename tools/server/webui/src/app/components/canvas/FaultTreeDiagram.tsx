import { useEffect, useMemo, useRef, useState } from "react";
import "./canvas.css";
import { formatInteger } from "../../../lib/format";
import { EmptyState } from "../../ui";
import { drawFaultTreeCanvas } from "./drawFaultTree";
import { layoutFaultTree } from "./layoutFaultTree";
import type { CanvasSize, FaultTreePayload, ViewTransform } from "./types";

const DEFAULT_SIZE: CanvasSize = { width: 900, height: 620 };
const DEFAULT_VIEW: ViewTransform = { scale: 1, offsetX: 72, offsetY: 72 };

export type FaultTreeDiagramProps = {
  tree: FaultTreePayload;
  /** Shown below the canvas when omitted. */
  note?: string;
};

export function FaultTreeDiagram({ tree, note }: FaultTreeDiagramProps) {
  const layout = useMemo(() => layoutFaultTree(tree), [tree]);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);
  const [size, setSize] = useState<CanvasSize>(DEFAULT_SIZE);
  const [view, setView] = useState<ViewTransform>(DEFAULT_VIEW);

  const fitView = () => {
    const scale = Math.max(0.001, Math.min(1.4, Math.min(size.width / layout.width, size.height / layout.height) * 0.92));
    setView({
      scale,
      offsetX: Math.max(24, (size.width - layout.width * scale) / 2),
      offsetY: Math.max(24, (size.height - layout.height * scale) / 2),
    });
  };

  const resetView = () => setView(DEFAULT_VIEW);

  const zoomAtCenter = (factor: number) => {
    const px = size.width / 2;
    const py = size.height / 2;
    setView((prev) => {
      const nextScale = Math.max(0.03, Math.min(2.5, prev.scale * factor));
      const worldX = (px - prev.offsetX) / prev.scale;
      const worldY = (py - prev.offsetY) / prev.scale;
      return {
        scale: nextScale,
        offsetX: px - worldX * nextScale,
        offsetY: py - worldY * nextScale,
      };
    });
  };

  const panView = (dx: number, dy: number) => {
    setView((prev) => ({ ...prev, offsetX: prev.offsetX + dx, offsetY: prev.offsetY + dy }));
  };

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
    fitView();
    // Fit when layout dimensions change; pan/zoom owns subsequent view updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout.width, layout.height]);

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
  const footerNote = note ?? "Canvas DAG renderer: shared nodes are drawn once; drag to pan, wheel to zoom. Labels appear as zoom increases.";

  return (
    <div className="fault-tree-frame" ref={frameRef}>
      <div className="fault-tree-toolbar">
        <span>
          {formatInteger(gateCount)} gates · {formatInteger(basicEventCount)} basic events · {formatInteger(tree.edges.length)} edges
        </span>
        <div className="fault-tree-controls" aria-label="Fault tree viewport controls">
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
        onWheel={(event) => {
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          const px = event.clientX - rect.left;
          const py = event.clientY - rect.top;
          const nextScale = Math.max(0.001, Math.min(2.5, view.scale * Math.exp(-event.deltaY * 0.0012)));
          const worldX = (px - view.offsetX) / view.scale;
          const worldY = (py - view.offsetY) / view.scale;
          setView({
            scale: nextScale,
            offsetX: px - worldX * nextScale,
            offsetY: py - worldY * nextScale,
          });
        }}
      />
      <p className="fault-tree-note">{footerNote}</p>
    </div>
  );
}
