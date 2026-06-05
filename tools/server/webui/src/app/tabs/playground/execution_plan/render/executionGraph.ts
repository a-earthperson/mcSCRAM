import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mathBox } from "mathbox";
import type { InkPalette } from "../../../../../components/plotTokens";
import { type ExecutionModel, defaultExecutionModel } from "../sycl";
import { buildGlobalAxes } from "./globalAxes";
import type { FieldExtent } from "./bitpackField";
import {
  type ExecutionSceneController,
  type ExecutionSceneOptions,
  createExecutionSceneController,
} from "./executionSceneController";
import { cleanupContainer, releaseWebGlContexts } from "./utils";
import { VIEW_RANGE, VIEW_SCALE } from "./viewConfig";

/**
 * Imperative MathBox controller for the execution scene.
 *
 * MathBox bootstraps its own three.js renderer, scene, camera and RAF loop
 * (via threestrap). We keep that entirely behind this module so the React layer
 * only deals with a tiny controller surface and never imports three or mathbox
 * directly. Construction is fully guarded: if WebGL is unavailable or MathBox
 * cannot initialise, we return `null` and the caller renders the 2D fallback.
 *
 * Framing: a single world-fixed bitpack field (sized to the largest work-group
 * across the model) is framed-to-fit at a fixed oblique angle. Navigation is:
 *   - wheel  -> discrete layer cursor (Z), swapping the visible layer overlays;
 *   - drag   -> pan/scroll across the field plane (X/Y);
 *   - rotate -> locked (fixed viewing angle).
 * The field and camera never move when changing layers — only overlays swap.
 */

/** Live camera framing readout (for tuning constants in this module). */
export type CameraFrameState = {
  position: [number, number, number];
  /** Orbit pivot / pan position. */
  target: [number, number, number];
  /** Unit vector from target toward the eye (matches `CAMERA_DIRECTION`). */
  direction: [number, number, number];
  /** Eye-to-target distance; primary zoom readout. */
  distance: number;
  /** `distance /` distance right after initial `frameField`. */
  zoomRatio: number;
  fov: number;
};

export type ExecutionGraphController = {
  /** Focus a partition's layer overlays (null -> the model's default kernel). */
  setSelected: (partitionId: number | null) => void;
  /** Step the layer cursor along +Z / -Z. */
  step: (direction: "next" | "prev") => void;
  goToStep: (stepIndex: number) => void;
  stepCount: () => number;
  stepIndex: () => number;
  /** Current camera framing (updated each call from the live three.js camera). */
  getCameraFrame: () => CameraFrameState | null;
  /** Sync renderer dimensions after the stage container resizes. */
  resize: () => void;
  dispose: () => void;
};

export type ExecutionGraphOptions = {
  container: HTMLElement;
  model: ExecutionModel;
  sceneOptions: ExecutionSceneOptions;
};

// Fixed viewing angle (rotation is locked). +X points left-of-screen so the
// field reads with execution depth receding along +Z.
const CAMERA_DIRECTION = new THREE.Vector3(-0.5, 0.25, 1).normalize();
/** Vertical field of view (degrees) on the perspective camera. Orbit zoom changes distance, not FOV. */
const CAMERA_FOV = 50;
const FIT_MARGIN = 1.2;
const FOV_MIN = 12;
const FOV_MAX = 90;
const FOV_KEY_STEP = 2;
const FOV_WHEEL_STEP = 0.15;
/** Wheel delta (deltaY) that advances the layer cursor by one (Z axis). */
const WHEEL_STEP_THRESHOLD = 60;
/** World units of Y field scroll per unit of horizontal wheel delta (deltaX). */
const Y_SCROLL_SPEED = 0.012;
/** World units of Y field scroll per Up/Down arrow press. */
const KEY_PAN_STEP = 0.3;

/**
 * Bitpack-field / scene tuning. Edit these to dial in the rendered span; they
 * are forwarded verbatim into the scene controller. Leave a value `undefined`
 * to fall back to the controller's built-in default.
 *
 * Geometry recap: the field is a 2D lattice of bitpack tiles (rows = events,
 * cols = samples). Each tile is `bitsPerSide x bitsPerSide` points. `*Pitch`
 * are world-unit spacings; padding tiles ring the core and fade to transparent.
 */
const SCENE_CONFIG: Omit<ExecutionSceneOptions, "palette"> = {
  // --- rendered span size (in tiles) ---------------------------------------
  // Set fieldRows/fieldCols explicitly to render exactly this many tiles,
  // decoupled from the per-graph max work-group shape. Set to `undefined` to
  // auto-size to the largest work-group across the model (clamped by maxField*).
  fieldRows: 8,
  fieldCols: 16,
  maxFieldRows: 64,
  maxFieldCols: 64,

  // --- faded padding ring ---------------------------------------------------
  paddingRows: 3,
  paddingCols: 3,
  /** Fade function exponent: 1 = linear, >1 eases harder toward the edge. */
  paddingFalloff: 1.5,

  // --- lattice spacing / point size ----------------------------------------
  /** Point-to-point spacing within one tile (world units). */
  bitPitch: 0.02,
  /** Tile-to-tile spacing (world units). */
  tilePitch: 0.18,
  /** Bit-point screen size (px). */
  pointSize: 7,

  // --- layer-transition animation ------------------------------------------
  transitionMs: 320,
  transitionTravel: 1.5,

  // --- overlay tiers --------------------------------------------------------
  showWorkItems: false,
  showSubGroups: true,
  showWorkGroup: false,
};

const _target = new THREE.Vector3();
const _eye = new THREE.Vector3();
const _viewDir = new THREE.Vector3();

function readCameraFrame(three: any, baselineDistance: number): CameraFrameState | null {
  try {
    const camera = three.camera;
    const controlsTarget = three.controls?.target;
    if (!camera?.position) return null;

    _eye.set(camera.position.x, camera.position.y, camera.position.z);
    if (controlsTarget) {
      _target.set(controlsTarget.x, controlsTarget.y, controlsTarget.z);
    } else {
      _target.set(0, 0, 0);
    }

    const distance = _eye.distanceTo(_target);
    _viewDir.subVectors(_eye, _target);
    if (_viewDir.lengthSq() > 1e-12) {
      _viewDir.normalize();
    } else {
      _viewDir.copy(CAMERA_DIRECTION);
    }

    return {
      position: [_eye.x, _eye.y, _eye.z],
      target: [_target.x, _target.y, _target.z],
      direction: [_viewDir.x, _viewDir.y, _viewDir.z],
      distance,
      zoomRatio: baselineDistance > 1e-9 ? distance / baselineDistance : 1,
      fov: typeof camera.fov === "number" ? camera.fov : CAMERA_FOV,
    };
  } catch {
    return null;
  }
}

/** Frame the camera once to fit the world-fixed field at the locked angle. */
function frameField(three: any, extent: FieldExtent): void {
  const radius = Math.max(0.05, 0.5 * Math.max(extent.width, extent.height));
  try {
    const camera = three.camera;
    const fovDeg = typeof camera?.fov === "number" ? camera.fov : CAMERA_FOV;
    const fov = (fovDeg * Math.PI) / 180;
    const distance = (radius / Math.tan(fov / 2)) * FIT_MARGIN;

    _target.set(0, 0, 0);
    _eye.copy(CAMERA_DIRECTION).multiplyScalar(distance).add(_target);
    camera.position?.set?.(_eye.x, _eye.y, _eye.z);
    if ("near" in camera) camera.near = Math.max(0.01, distance * 0.01);
    if ("far" in camera) camera.far = distance * 24;
    camera.updateProjectionMatrix?.();

    if (three.controls) {
      three.controls.target?.set?.(0, 0, 0);
      three.controls.update?.();
    }
  } catch {
    /* camera/control flavour differences — non-fatal */
  }
}

export function createExecutionGraph(
  options: ExecutionGraphOptions,
): ExecutionGraphController | null {
  const { container, sceneOptions } = options;
  const { palette } = sceneOptions;
  const model = options.model ?? defaultExecutionModel();

  let mathbox: any = null;
  try {
    // Installing "mathbox" up front means the meta-API is registered before
    // mathBox() runs its own bootstrap, so it never auto-installs the
    // fullscreen "splash" loader plugin (which would cover the panel).
    mathbox = mathBox({
      element: container,
      plugins: ["core", "controls", "cursor", "stats", "mathbox"],
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

  // Controls: the camera angle is locked and all OrbitControls interaction is
  // off. Scrolling is managed by hand in the wheel handler below (vertical ->
  // layer cursor, horizontal -> Y field scroll); we only reuse the controls'
  // target so update() keeps the camera pointed as we pan it.
  try {
    if (three.controls) {
      three.controls.enableRotate = true;
      three.controls.enablePan = true;
      three.controls.enableZoom = true;
      three.controls.enableDamping = true;
    }
  } catch {
    /* control flavour differences — non-fatal */
  }

  let controller: ExecutionSceneController | null = null;
  let baselineCameraDistance = 1;

  try {
    mathbox.set({ scale: 720 });
    const view = mathbox.cartesian({ range: VIEW_RANGE, scale: VIEW_SCALE });
    buildGlobalAxes(view, palette);

    // SCENE_CONFIG (defined in this file) is the source of truth for field
    // tuning; the caller's sceneOptions supplies the palette (and may override).
    controller = createExecutionSceneController(view, model, { ...sceneOptions, ...SCENE_CONFIG });
    controller.focusKernel(model.defaultKernelId);
    try {
      if (three.camera && "fov" in three.camera) {
        three.camera.fov = CAMERA_FOV;
        three.camera.updateProjectionMatrix?.();
      }
    } catch {
      /* non-fatal */
    }
    frameField(three, controller.fieldExtent());
    baselineCameraDistance = readCameraFrame(three, 1)?.distance ?? 1;
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

  const adjustFov = (deltaDeg: number) => {
    try {
      const camera = three.camera;
      if (!camera || !("fov" in camera)) return;
      camera.fov = Math.min(FOV_MAX, Math.max(FOV_MIN, camera.fov + deltaDeg));
      camera.updateProjectionMatrix?.();
    } catch {
      /* non-fatal */
    }
  };

  // Pan the (fixed-angle) camera along world Y by translating both eye and
  // target so the offset — hence the viewing angle — is preserved. The target
  // is clamped to the padded field's Y extent so it can't drift into the void.
  const panY = (dy: number) => {
    try {
      const target = three.controls?.target;
      if (!target) return;
      const limit = controller?.scrollLimitY() ?? Infinity;
      const nextY = Math.max(-limit, Math.min(limit, target.y + dy));
      const applied = nextY - target.y;
      if (applied === 0) return;
      if (three.camera?.position) three.camera.position.y += applied;
      target.y = nextY;
      three.controls?.update?.();
    } catch {
      /* non-fatal */
    }
  };

  // Self-managed scroll:
  //   vertical wheel (deltaY)   -> discrete layer cursor (Z), accumulated so
  //     trackpad momentum advances one layer per threshold crossing;
  //   horizontal wheel (deltaX) -> Y-axis field scroll;
  //   X-axis scroll is intentionally ignored.
  let wheelAccum = 0;
  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    // Alt+wheel adjusts FOV (OrbitControls zoom only dollies distance).
    if (event.altKey) {
      adjustFov(event.deltaY > 0 ? FOV_WHEEL_STEP : -FOV_WHEEL_STEP);
      return;
    }
    wheelAccum += event.deltaY;
    while (Math.abs(wheelAccum) >= WHEEL_STEP_THRESHOLD) {
      controller?.step(wheelAccum > 0 ? "next" : "prev");
      wheelAccum -= Math.sign(wheelAccum) * WHEEL_STEP_THRESHOLD;
    }
    if (event.deltaX !== 0) panY(-event.deltaX * Y_SCROLL_SPEED);
  };
  container.addEventListener("wheel", onWheel, { passive: false });

  // Arrow keys mirror the wheel: Up/Down -> Y field scroll, Left/Right -> layer
  // cursor. Gated on pointer-over-scene (like the wheel) so we never hijack the
  // page's arrow-key scrolling unless the user is actually over the scene.
  let pointerInside = false;
  const onPointerEnter = () => {
    pointerInside = true;
  };
  const onPointerLeave = () => {
    pointerInside = false;
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (!pointerInside) return;
    switch (event.key) {
      case "ArrowUp":
        event.preventDefault();
        panY(KEY_PAN_STEP);
        break;
      case "ArrowDown":
        event.preventDefault();
        panY(-KEY_PAN_STEP);
        break;
      case "ArrowRight":
        event.preventDefault();
        controller?.step("next");
        break;
      case "ArrowLeft":
        event.preventDefault();
        controller?.step("prev");
        break;
      case "[":
        event.preventDefault();
        adjustFov(-FOV_KEY_STEP);
        break;
      case "]":
        event.preventDefault();
        adjustFov(FOV_KEY_STEP);
        break;
      default:
        break;
    }
  };
  container.addEventListener("pointerenter", onPointerEnter);
  container.addEventListener("pointerleave", onPointerLeave);
  window.addEventListener("keydown", onKeyDown);

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
    controller?.focusKernel(targetId);
  };

  const step = (direction: "next" | "prev") => {
    controller?.step(direction);
  };

  const goToStep = (stepIndex: number) => {
    controller?.showLayer(stepIndex);
  };

  const dispose = () => {
    container.removeEventListener("wheel", onWheel);
    container.removeEventListener("pointerenter", onPointerEnter);
    container.removeEventListener("pointerleave", onPointerLeave);
    window.removeEventListener("keydown", onKeyDown);
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
    stepCount: () => controller?.stepCount() ?? 0,
    stepIndex: () => controller?.layerIndex() ?? 0,
    getCameraFrame: () => readCameraFrame(three, baselineCameraDistance),
    resize,
    dispose,
  };
}
