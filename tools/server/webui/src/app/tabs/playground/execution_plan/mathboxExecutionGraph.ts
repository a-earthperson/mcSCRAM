import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mathBox } from "mathbox";
import type { InkPalette } from "../../../../components/plotTokens";
import { kindColor, type ExecutionScene } from "./mathbox/executionScene";
import {
  BITPACK_SIZE,
  buildBitpackPointField,
  bitpackPointFieldFromBasicEventBitpackBufferShape,
  bitpackPointFieldWorldExtent,
} from "./mathbox/bitpackField";
import type { BasicEventBitpackBufferShape } from "./mathbox/types";
import { buildGlobalAxes } from "./mathbox/globalAxes";
import { buildSelectionLayer, towerBase, towerTop } from "./mathbox/selectionLayer";
import { cleanupContainer, releaseWebGlContexts } from "./mathbox/utils";
import { VIEW_RANGE, VIEW_SCALE } from "./mathbox/viewConfig";

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
      mathbox: { init: true, inspect: true, warmup: 1, splash: false },
    });
  } catch (error) {
    cleanupContainer(container);
    throw error instanceof Error ? error : new Error(String(error));
  }

  // WebGL unavailable -> threestrap returns the bootstrap with `.fallback` set
  // and never installs the mathbox meta-API (no `.cartesian`).
  if (!mathbox || mathbox.fallback || typeof mathbox.cartesian !== "function") {
    try {
      mathbox?.three?.renderer?.dispose?.();
    } catch {
      /* ignore */
    }
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
    three.renderer.setClearColor(new THREE.Color(0x000000), 1);
  } catch {
    /* renderer flavour without setClearColor — ignore */
  }

  const basicEventBitpackBufferShape: BasicEventBitpackBufferShape = {
    batch_size: 1,
    bitpacks_per_batch: 64,
    bits_per_bitpack: BITPACK_SIZE,
    num_events: 64,
  };
  const bitpackPointFieldConfig = bitpackPointFieldFromBasicEventBitpackBufferShape(basicEventBitpackBufferShape);
  const bitpackPointFieldExtent = bitpackPointFieldWorldExtent(bitpackPointFieldConfig);
  const bitpackSceneRadius = Math.max(1, Math.hypot(bitpackPointFieldExtent.width, bitpackPointFieldExtent.height) * 0.5);

  try {
    three.camera.position.set(1.9, 1.35, 2.0);
    if ("far" in three.camera) {
      three.camera.far = Math.max(three.camera.far, bitpackSceneRadius * 16);
      three.camera.updateProjectionMatrix?.();
    }
    if (three.controls) {
      three.controls.target?.set?.(0.5, 0.32, 0);
      three.controls.minDistance = 0.2;
      three.controls.maxDistance = Math.max(6, bitpackSceneRadius * 8);
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

  let selection: ReturnType<typeof buildSelectionLayer> | null = null;
  try {
    mathbox.set({ scale: 720 });
    const view = mathbox.cartesian({
      range: VIEW_RANGE,
      scale: VIEW_SCALE,
    });
    buildGlobalAxes(view, palette);
    buildBitpackPointField(view, bitpackPointFieldConfig);
    // buildGround(view, scene, palette);
    // buildTowers(view, scene, palette);
    // buildEdges(view, scene, palette);
    //selection = buildSelectionLayer(view, palette);
  } catch (error) {
    try {
      three.renderer?.dispose?.();
    } catch {
      /* ignore */
    }
    try {
      three.destroy?.();
    } catch {
      /* ignore */
    }
    releaseWebGlContexts(container);
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
      const activeSelection = selection as ReturnType<typeof buildSelectionLayer> | null;
      if (!activeSelection) return;
      if (!tower) {
        activeSelection.bar.set("visible", false);
        activeSelection.cap.set("visible", false);
        return;
      }
      const base = towerBase(tower);
      const top = towerTop(tower);
      const accent = kindColor(tower.colorKey, palette);
      activeSelection.barData.set("data", [base, top]);
      activeSelection.capData.set("data", [top]);
      activeSelection.bar.set("color", accent);
      activeSelection.cap.set("color", accent);
      activeSelection.bar.set("visible", true);
      activeSelection.cap.set("visible", true);
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
      three.renderer?.dispose?.();
    } catch {
      /* ignore */
    }
    try {
      three.destroy?.();
    } catch {
      /* ignore */
    }
    releaseWebGlContexts(container);
    cleanupContainer(container);
  };

  return { setSelected, resize, dispose };
}
