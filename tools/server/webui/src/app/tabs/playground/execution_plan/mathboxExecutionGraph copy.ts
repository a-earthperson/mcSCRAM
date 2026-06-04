import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mathBox } from "mathbox";
import type { InkPalette } from "../../../../components/plotTokens";
import { withAlpha } from "../../../../components/plotTokens";
import {
  kindColor,
  towerFill,
  type ExecutionScene,
  type SceneTower,
} from "./mathbox/executionScene";

/**
 * Imperative MathBox controller for the execution landscape.
 *
 * MathBox bootstraps its own three.js renderer, scene, camera and RAF loop
 * (via threestrap). We keep that entirely behind this module so the React
 * layer only deals with a tiny controller surface and never imports three or
 * mathbox directly. Construction is fully guarded: if WebGL is unavailable or
 * MathBox cannot initialise, we return `null` and the caller renders the 2D
 * fallback instead.
 */

export type ExecutionGraphController = {
  setSelected: (partitionId: number | null) => void;
  /** Sync renderer dimensions after the stage container resizes (e.g. fullscreen). */
  resize: () => void;
  dispose: () => void;
};

export type ExecutionGraphOptions = {
  container: HTMLElement;
  scene: ExecutionScene;
  palette: InkPalette;
};

/** Three coordinates per point, two points per vector segment. */
type Point3 = [number, number, number];

const VIEW_RANGE: [[number, number], [number, number], [number, number]] = [
  [0, 1],
  [0, 1],
  [-0.62, 0.62],
];
const VIEW_SCALE: Point3 = [1.55, 0.95, 1.55];

const SEL_BAR_ID = "exec-sel-bar";
const SEL_CAP_ID = "exec-sel-cap";

function towerBase(tower: SceneTower): Point3 {
  return [tower.x, 0, tower.z];
}

function towerTop(tower: SceneTower): Point3 {
  return [tower.x, tower.height, tower.z];
}

export function createExecutionGraph(
  options: ExecutionGraphOptions,
): ExecutionGraphController | null {
  const { container, scene, palette } = options;

  let mathbox: any = null;
  try {
    // Installing "mathbox" up front means the meta-API is registered before
    // mathBox() runs its own bootstrap, so it never auto-installs the
    // fullscreen "splash" loader plugin (which would cover the panel).
    mathbox = mathBox({
      element: container,
      plugins: ["core", "controls", "cursor", "mathbox"],
      controls: { klass: OrbitControls },
      camera: {},
      mathbox: { inspect: true, warmup: 0, splash: false },
    });
  } catch (error) {
    cleanupContainer(container);
    throw error instanceof Error ? error : new Error(String(error));
  }

  // WebGL unavailable -> threestrap returns the bootstrap with `.fallback` set
  // and never installs the mathbox meta-API (no `.cartesian`).
  if (!mathbox || mathbox.fallback || typeof mathbox.cartesian !== "function") {
    try {
      mathbox?.three?.destroy?.();
    } catch {
      /* ignore */
    }
    cleanupContainer(container);
    return null;
  }

  const three = mathbox.three;

  try {
    three.renderer.setClearColor(new THREE.Color(palette.plate || "#0c0e15"), 1);
  } catch {
    /* renderer flavour without setClearColor — ignore */
  }

  // Camera framing + orbit target. Mirrors the documented network-graph setup.
  try {
    three.camera.position.set(1.9, 1.35, 2.0);
    if (three.controls) {
      three.controls.target?.set?.(0.5, 0.32, 0);
      three.controls.minDistance = 1.1;
      three.controls.maxDistance = 6;
      three.controls.enablePan = true;
      three.controls.rotateSpeed = 1;
      three.controls.zoomSpeed = 0.8;
      three.controls.enableDamping = true;
      three.controls.dampingFactor = 0.2;
      three.controls.update?.();
    }
  } catch {
    /* control flavour differences — non-fatal */
  }

  let selection: ReturnType<typeof buildSelectionLayer>;
  try {
    mathbox.set({ scale: 720, focus: 1.6 });
    const view = mathbox.cartesian({ range: VIEW_RANGE, scale: VIEW_SCALE });
    buildGround(view, scene, palette);
    // buildTowers(view, scene, palette);
    // buildEdges(view, scene, palette);
    selection = buildSelectionLayer(view, palette);
  } catch (error) {
    try {
      three.destroy?.();
    } catch {
      /* ignore */
    }
    cleanupContainer(container);
    throw error instanceof Error ? error : new Error(String(error));
  }

  const resize = () => {
    try {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      three.renderer?.setSize?.(width, height);
      const camera = three.camera;
      if (camera && "aspect" in camera) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix?.();
      }
    } catch {
      /* resize is best-effort */
    }
  };

  resize();

  const setSelected = (partitionId: number | null) => {
    const tower = partitionId === null ? null : scene.towerById.get(partitionId) ?? null;
    try {
      if (!tower) {
        selection.bar.set("visible", false);
        selection.cap.set("visible", false);
        return;
      }
      const base = towerBase(tower);
      const top = towerTop(tower);
      const accent = kindColor(tower.colorKey, palette);
      selection.barData.set("data", [base, top]);
      selection.capData.set("data", [top]);
      selection.bar.set("color", accent);
      selection.cap.set("color", accent);
      selection.bar.set("visible", true);
      selection.cap.set("visible", true);
    } catch {
      /* selection update is best-effort */
    }
  };

  const dispose = () => {
    try {
      three.controls?.dispose?.();
    } catch {
      /* ignore */
    }
    try {
      three.destroy?.();
    } catch {
      /* ignore */
    }
    cleanupContainer(container);
  };

  return { setSelected, resize, dispose };
}

function cleanupContainer(container: HTMLElement): void {
  try {
    while (container.firstChild) container.removeChild(container.firstChild);
  } catch {
    /* ignore */
  }
}

function buildGround(view: any, scene: ExecutionScene, palette: InkPalette): void {
  view.grid({
    axes: [1, 3],
    divideX: Math.min(12, Math.max(2, scene.layerCount)),
    divideY: Math.min(12, Math.max(2, scene.maxLanes + 1)),
    width: 1,
    color: palette.inkFaint,
    opacity: 1,
    zBias: -5,
  });
  // Execution-order axis (layers) and work-volume axis (height).
  // view.axis({ axis: 1, width: 2, color: withAlpha(palette.inkMute, 0.7), end: false, opacity: 0.5 });
  // view.axis({ axis: 2, width: 2, color: withAlpha(palette.inkMute, 0.7), end: false, opacity: 0.4 });
}

function buildTowers(view: any, scene: ExecutionScene, palette: InkPalette): void {
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

function buildEdges(view: any, scene: ExecutionScene, palette: InkPalette): void {
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

function buildSelectionLayer(view: any, palette: InkPalette) {
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
