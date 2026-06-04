import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mathBox } from "mathbox";
import type { InkPalette } from "../../../../components/plotTokens";
import { type ExecutionModel, defaultExecutionModel } from "./sycl";
import type { FieldExtent } from "./mathbox/bitpackField";
import { buildGlobalAxes } from "./mathbox/globalAxes";
import {
  type ExecutionSequencer,
  type KernelSceneController,
  createExecutionSequencer,
  createKernelSceneController,
  stepIndexForKernel,
} from "./mathbox/executionSequencer";
import type { KernelSceneHandles } from "./mathbox/kernelScene";
import { cleanupContainer, releaseWebGlContexts } from "./mathbox/utils";
import { VIEW_RANGE, VIEW_SCALE } from "./mathbox/viewConfig";

/**
 * Imperative MathBox controller for the execution scene.
 *
 * MathBox bootstraps its own three.js renderer, scene, camera and RAF loop
 * (via threestrap). We keep that entirely behind this module so the React layer
 * only deals with a tiny controller surface and never imports three or mathbox
 * directly. Construction is fully guarded: if WebGL is unavailable or MathBox
 * cannot initialise, we return `null` and the caller renders the 2D fallback.
 *
 * The scene itself is assembled from the pure {@link ExecutionModel}: a
 * {@link KernelSceneController} renders one kernel's work-group window at a time
 * (build-once + visibility toggle) and a {@link ExecutionSequencer} walks the
 * topological execution schedule. Selection and stepping just switch which
 * kernel window is visible and reframe the camera to it.
 */

export type ExecutionGraphController = {
  /** Show a kernel partition's window (null -> the model's default kernel). */
  setSelected: (partitionId: number | null) => void;
  /** Walk the execution schedule. */
  step: (direction: "next" | "prev") => void;
  goToStep: (stepIndex: number) => void;
  stepCount: () => number;
  stepIndex: () => number;
  /** Sync renderer dimensions after the stage container resizes. */
  resize: () => void;
  dispose: () => void;
};

export type ExecutionGraphOptions = {
  container: HTMLElement;
  model: ExecutionModel;
  palette: InkPalette;
};

const CAMERA_DIRECTION = new THREE.Vector3(0.32, 0.2, 1).normalize();

function frameCamera(three: any, extent: FieldExtent): void {
  const radius = Math.max(0.05, 0.5 * Math.hypot(extent.width, extent.height));
  try {
    const camera = three.camera;
    const fovDeg = typeof camera?.fov === "number" ? camera.fov : 50;
    const fov = (fovDeg * Math.PI) / 180;
    const distance = (radius / Math.tan(fov / 2)) * 1.25;
    const eye = CAMERA_DIRECTION.clone().multiplyScalar(distance);
    camera.position?.set?.(eye.x, eye.y, eye.z);
    if ("near" in camera) camera.near = Math.max(0.01, distance * 0.01);
    if ("far" in camera) camera.far = distance * 24;
    camera.updateProjectionMatrix?.();

    if (three.controls) {
      three.controls.target?.set?.(0, 0, 0);
      three.controls.minDistance = distance * 0.12;
      three.controls.maxDistance = distance * 8;
      three.controls.update?.();
    }
  } catch {
    /* camera/control flavour differences — non-fatal */
  }
}

export function createExecutionGraph(
  options: ExecutionGraphOptions,
): ExecutionGraphController | null {
  const { container, palette } = options;
  const model = options.model ?? defaultExecutionModel();

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

  let controller: KernelSceneController | null = null;
  let sequencer: ExecutionSequencer | null = null;

  try {
    three.camera.position.set(1.9, 1.35, 2.0);
    if (three.controls) {
      three.controls.enablePan = true;
      three.controls.rotateSpeed = 1;
      three.controls.zoomSpeed = 0.8;
      three.controls.enableDamping = true;
      three.controls.dampingFactor = 0.2;
    }
  } catch {
    /* control flavour differences — non-fatal */
  }

  try {
    mathbox.set({ scale: 720 });
    const view = mathbox.cartesian({ range: VIEW_RANGE, scale: VIEW_SCALE });
    buildGlobalAxes(view, palette);

    controller = createKernelSceneController(view, model, { palette });
    sequencer = createExecutionSequencer(controller, model);

    const initial = controller.show(model.defaultKernelId);
    if (initial) frameCamera(three, initial.extent);
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

  const showHandles = (handles: KernelSceneHandles | null) => {
    if (handles) frameCamera(three, handles.extent);
  };

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
    const targetId = partitionId === null ? model.defaultKernelId : partitionId;
    const handles = controller?.show(targetId) ?? null;
    showHandles(handles);
  };

  const step = (direction: "next" | "prev") => {
    if (!sequencer) return;
    showHandles(direction === "next" ? sequencer.next() : sequencer.prev());
  };

  const goToStep = (stepIndex: number) => {
    showHandles(sequencer?.goTo(stepIndex) ?? null);
  };

  const dispose = () => {
    try {
      controller?.dispose();
    } catch {
      /* ignore */
    }
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

  return {
    setSelected,
    step,
    goToStep,
    stepCount: () => sequencer?.stepCount ?? 0,
    stepIndex: () => sequencer?.index() ?? 0,
    resize,
    dispose,
  };
}

export { stepIndexForKernel };
