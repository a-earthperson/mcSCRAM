import type { InkPalette } from "../../../../../components/plotTokens";
import { withAlpha } from "../../../../../components/plotTokens";
import {
  kindColor,
  towerFill,
  type ExecutionScene,
  type SceneTower,
} from "./executionScene";
import { towerBase, towerTop } from "./selectionLayer";
import type { MathBoxView, Point3 } from "./types";

export function buildGround(view: MathBoxView, scene: ExecutionScene, palette: InkPalette): void {
  view.grid({
    axes: [1, 3],
    divideX: Math.min(12, Math.max(2, scene.layerCount)),
    divideY: Math.min(12, Math.max(2, scene.maxLanes + 1)),
    width: 1,
    color: palette.inkFaint,
    opacity: 1,
    zBias: -5,
  });
}

export function buildTowers(view: MathBoxView, scene: ExecutionScene, palette: InkPalette): void {
  const byKind = new Map<string, SceneTower[]>();
  for (const tower of scene.towers) {
    const bucket = byKind.get(tower.colorKey);
    if (bucket) bucket.push(tower);
    else byKind.set(tower.colorKey, [tower]);
  }

  for (const kind of scene.kinds) {
    const towers = byKind.get(kind);
    if (!towers || !towers.length) continue;
    const stroke = kindColor(kind, palette);
    const fill = towerFill(kind, palette);

    const barPoints: Point3[] = [];
    const capPoints: Point3[] = [];
    for (const tower of towers) {
      barPoints.push(towerBase(tower), towerTop(tower));
      capPoints.push(towerTop(tower));
    }

    view
      .array({ width: towers.length, items: 2, channels: 3, data: barPoints, live: false })
      .vector({ color: fill, width: 9, start: false, end: false, opacity: 0.92, zBias: 1 });

    view
      .array({ width: towers.length, items: 1, channels: 3, data: capPoints, live: false })
      .point({ color: stroke, size: 11, opacity: 0.96, zBias: 2 });
  }
}

export function buildEdges(view: MathBoxView, scene: ExecutionScene, palette: InkPalette): void {
  if (!scene.edges.length) return;
  const points: Point3[] = [];
  for (const edge of scene.edges) {
    const source = scene.towerById.get(edge.source);
    const target = scene.towerById.get(edge.target);
    if (!source || !target) continue;
    points.push(towerTop(source), towerTop(target));
  }
  if (!points.length) return;

  view
    .array({ width: points.length / 2, items: 2, channels: 3, data: points, live: false })
    .vector({
      color: withAlpha(palette.inkMute, 0.55),
      width: 1.5,
      start: false,
      end: true,
      size: 2.5,
      opacity: 0.5,
      zBias: 0,
    });
}
