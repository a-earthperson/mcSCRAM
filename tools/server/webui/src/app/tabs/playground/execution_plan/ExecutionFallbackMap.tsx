import { useMemo } from "react";
import { usePlotTheme } from "../../../../components/plotTokens";
import { withAlpha } from "../../../../components/plotTokens";
import {
  compactCount,
  kindColor,
  kindLabel,
  type ExecutionScene,
  type SceneTower,
} from "./mathbox/executionScene";

/**
 * 2D layered fallback for the execution landscape. Rendered when WebGL /
 * MathBox cannot initialise. It is a faithful (if flatter) view of the same
 * scene model: columns are execution layers, nodes are kernel partitions sized
 * by launched work, links are partition dependencies.
 */

const NODE_W = 150;
const NODE_H = 52;
const COL_GAP = 78;
const ROW_GAP = 18;
const PAD = 28;

type PlacedNode = {
  tower: SceneTower;
  col: number;
  cx: number;
  cy: number;
  x: number;
  y: number;
};

export function ExecutionFallbackMap({
  scene,
  selectedId,
  onSelect,
}: {
  scene: ExecutionScene;
  selectedId: number | null;
  onSelect: (partitionId: number | null) => void;
}) {
  const { palette } = usePlotTheme();

  const { nodes, width, height, nodeById } = useMemo(() => {
    const layers = Array.from(new Set(scene.towers.map((t) => t.layer))).sort((a, b) => a - b);
    const colByLayer = new Map<number, number>(layers.map((layer, index) => [layer, index]));
    const byLayer = new Map<number, SceneTower[]>();
    for (const tower of scene.towers) {
      const bucket = byLayer.get(tower.layer);
      if (bucket) bucket.push(tower);
      else byLayer.set(tower.layer, [tower]);
    }
    for (const bucket of byLayer.values()) bucket.sort((a, b) => a.id - b.id);

    const maxRows = Math.max(1, ...Array.from(byLayer.values(), (b) => b.length));
    const contentHeight = maxRows * NODE_H + (maxRows - 1) * ROW_GAP;

    const placed: PlacedNode[] = [];
    for (const [layer, bucket] of byLayer) {
      const col = colByLayer.get(layer) ?? 0;
      const colHeight = bucket.length * NODE_H + (bucket.length - 1) * ROW_GAP;
      const top = PAD + (contentHeight - colHeight) / 2;
      bucket.forEach((tower, row) => {
        const x = PAD + col * (NODE_W + COL_GAP);
        const y = top + row * (NODE_H + ROW_GAP);
        placed.push({ tower, col, x, y, cx: x + NODE_W / 2, cy: y + NODE_H / 2 });
      });
    }

    const lookup = new Map(placed.map((node) => [node.tower.id, node]));
    const cols = layers.length;
    return {
      nodes: placed,
      nodeById: lookup,
      width: PAD * 2 + cols * NODE_W + Math.max(0, cols - 1) * COL_GAP,
      height: PAD * 2 + contentHeight,
    };
  }, [scene]);

  if (!nodes.length) {
    return <div className="exec-fallback__empty">No kernel partitions to display.</div>;
  }

  return (
    <div className="exec-fallback">
      <svg
        className="exec-fallback__svg"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label="Execution plan layered partition map"
      >
        <g className="exec-fallback__edges">
          {scene.edges.map((edge, index) => {
            const source = nodeById.get(edge.source);
            const target = nodeById.get(edge.target);
            if (!source || !target) return null;
            const x1 = source.x + NODE_W;
            const y1 = source.cy;
            const x2 = target.x;
            const y2 = target.cy;
            const mid = (x1 + x2) / 2;
            return (
              <path
                key={`${edge.source}-${edge.target}-${index}`}
                className="exec-fallback__edge"
                d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={withAlpha(palette.inkMute, 0.5)}
                strokeWidth={1.2}
              />
            );
          })}
        </g>

        <g className="exec-fallback__nodes">
          {nodes.map(({ tower, x, y }) => {
            const color = kindColor(tower.colorKey, palette);
            const selected = tower.id === selectedId;
            return (
              <g
                key={tower.id}
                className="exec-fallback__node"
                transform={`translate(${x} ${y})`}
                role="button"
                tabIndex={0}
                aria-pressed={selected}
                aria-label={`${kindLabel(tower.colorKey)} partition ${tower.id}`}
                onClick={() => onSelect(selected ? null : tower.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(selected ? null : tower.id);
                  }
                }}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx={3}
                  fill={withAlpha(color, selected ? 0.26 : 0.14)}
                  stroke={selected ? palette.indigoDeep : withAlpha(color, 0.95)}
                  strokeWidth={selected ? 2 : 1.25}
                />
                <rect x={0} y={0} width={4} height={NODE_H} fill={color} />
                <rect
                  x={10}
                  y={NODE_H - 11}
                  width={(NODE_W - 20) * Math.min(1, tower.height)}
                  height={3}
                  rx={1.5}
                  fill={withAlpha(color, 0.85)}
                />
                <text className="exec-fallback__node-kind" x={12} y={20}>
                  {kindLabel(tower.colorKey)}
                </text>
                <text className="exec-fallback__node-meta" x={12} y={35}>
                  {compactCount(tower.partition.node_count)} nodes &middot; {compactCount(tower.partition.work_item_count)} wi
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
