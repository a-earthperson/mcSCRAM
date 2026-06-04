import type { InkPalette } from "../../../../../components/plotTokens";
import type { SceneTower } from "./executionScene";
import type { MathBoxView, Point3 } from "./types";

const SEL_BAR_ID = "exec-sel-bar";
const SEL_CAP_ID = "exec-sel-cap";

export function towerBase(tower: SceneTower): Point3 {
  return [tower.x, 0, tower.z];
}

export function towerTop(tower: SceneTower): Point3 {
  return [tower.x, tower.height, tower.z];
}

export function buildSelectionLayer(view: MathBoxView, palette: InkPalette) {
  const hidden: Point3 = [0, -1, 0];
  const barData = view.array({
    id: SEL_BAR_ID,
    width: 1,
    items: 2,
    channels: 3,
    data: [hidden, hidden],
    live: true,
  });
  const bar = barData.vector({
    color: palette.ink,
    width: 14,
    start: false,
    end: false,
    opacity: 0.4,
    visible: false,
    zBias: 3,
  });
  const capData = view.array({
    id: SEL_CAP_ID,
    width: 1,
    items: 1,
    channels: 3,
    data: [hidden],
    live: true,
  });
  const cap = capData.point({
    color: palette.indigoDeep,
    size: 22,
    opacity: 0.95,
    visible: false,
    zBias: 4,
  });
  return { barData, bar, capData, cap };
}
