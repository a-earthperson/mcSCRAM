/**
 * Execution scene controller — one world-fixed bitpack field + a layer cursor.
 *
 * One static bitpack lattice (the data-layer canvas) is built once, sized to the
 * largest single work-group across the model plus a faded padding margin on all
 * sides. Navigation is overlay swapping: each layer's SYCL launch geometry is a
 * transform-wrapped overlay group registered onto the field's core. Moving the
 * layer cursor (Z) does not move the field — it cross-fades the outgoing
 * overlays *out and away* while the incoming overlays *travel in and fade up*,
 * so the discrete cursor still reads as smooth motion. The field and its camera
 * framing stay put; the controller (above) handles Y scrolling by panning.
 */
import type { InkPalette } from "../../../../components/plot/plotTokens";
import {
  type ExecutionModel,
  bitpackGeometry,
  maxWorkGroupShape,
  representativeKernelId,
  stepIndexOfKernel,
} from "../sycl";
import { type FieldExtent, buildBitpackField } from "./bitpackField";
import { uninitializedBitSource } from "./bitSource";
import { type LayerOverlayHandles, buildLayerOverlays } from "./layerOverlays";
import type { MathBoxView } from "./types";

export type ExecutionSceneOptions = {
  palette: InkPalette;
  bitPitch?: number;
  tilePitch?: number;
  pointSize?: number;
  /**
   * Explicit core field size in tiles (rows = events, cols = samples). When
   * set, this is used verbatim and the per-graph max work-group shape (and the
   * `maxField*` caps) are ignored. Leave undefined to auto-size to the largest
   * work-group across the model, clamped by the caps below.
   */
  fieldRows?: number;
  fieldCols?: number;
  /** Caps applied only when auto-sizing (i.e. `fieldRows`/`fieldCols` unset). */
  maxFieldRows?: number;
  maxFieldCols?: number;
  /** Faded padding tiles per side; tune to taste. */
  paddingRows?: number;
  paddingCols?: number;
  /** Exponential decay constant (per padding tile) along +X/-X. */
  paddingFalloffX?: number;
  /** Exponential decay constant (per padding tile) along +Y/-Y. */
  paddingFalloffY?: number;
  /** Layer-transition animation. */
  transitionMs?: number;
  transitionTravel?: number;
  showWorkItems?: boolean;
  showSubGroups?: boolean;
  showWorkGroup?: boolean;
};

/** The active layer slice on the shared field. */
export type LayerCursor = {
  stepIndex: number;
  layer: number;
  kernelId: number;
  footprint: { rows: number; cols: number };
};

export type ExecutionSceneController = {
  showLayer: (stepIndex: number) => LayerCursor | null;
  focusKernel: (kernelId: number) => LayerCursor | null;
  step: (direction: "next" | "prev") => LayerCursor | null;
  stepCount: () => number;
  layerIndex: () => number;
  /** Core extent of the field (for one-time camera framing). */
  fieldExtent: () => FieldExtent;
  fieldShape: () => { rows: number; cols: number };
  /** World Y half-extent of the full (padded) field, for clamping Y scroll. */
  scrollLimitY: () => number;
  dispose: () => void;
};

// --- Tunables (override via options) ---------------------------------------
const DEFAULT_FIELD_CAP = 64;
const DEFAULT_PADDING_ROWS = 8;
const DEFAULT_PADDING_COLS = 8;
const DEFAULT_PADDING_FALLOFF_X = 2;
const DEFAULT_PADDING_FALLOFF_Y = 2;
const DEFAULT_TRANSITION_MS = 320;
const DEFAULT_TRANSITION_TRAVEL = 1.5;

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

type Transition = {
  outgoing: LayerOverlayHandles | null;
  incoming: LayerOverlayHandles;
  dir: number;
  start: number;
  duration: number;
  travel: number;
};

export function createExecutionSceneController(
  view: MathBoxView,
  model: ExecutionModel,
  options: ExecutionSceneOptions,
): ExecutionSceneController {
  const { palette } = options;
  const bitPitch = options.bitPitch ?? 0.02;
  const tilePitch = options.tilePitch ?? 0.18;
  const pointSize = options.pointSize ?? 7;
  const capRows = options.maxFieldRows ?? DEFAULT_FIELD_CAP;
  const capCols = options.maxFieldCols ?? DEFAULT_FIELD_CAP;
  const padRows = Math.max(0, options.paddingRows ?? DEFAULT_PADDING_ROWS);
  const padCols = Math.max(0, options.paddingCols ?? DEFAULT_PADDING_COLS);
  const paddingFalloffX = options.paddingFalloffX ?? DEFAULT_PADDING_FALLOFF_X;
  const paddingFalloffY = options.paddingFalloffY ?? DEFAULT_PADDING_FALLOFF_Y;
  const transitionMs = Math.max(1, options.transitionMs ?? DEFAULT_TRANSITION_MS);
  const transitionTravel = options.transitionTravel ?? DEFAULT_TRANSITION_TRAVEL;

  // Explicit dimensions win outright; otherwise auto-size to the largest
  // work-group across the model, clamped by the caps.
  const autoWg = maxWorkGroupShape(model);
  const coreRows = Math.max(
    1,
    Math.floor(options.fieldRows ?? Math.min(autoWg.rows, capRows)),
  );
  const coreCols = Math.max(
    1,
    Math.floor(options.fieldCols ?? Math.min(autoWg.cols, capCols)),
  );
  const totalRows = coreRows + 2 * padRows;
  const totalCols = coreCols + 2 * padCols;
  const bitsPerSide = bitpackGeometry(model.bitsPerBitpack).bitsPerSide;

  // The world-fixed blank canvas: a neutral, flat lattice with faded padding.
  const neutral = palette.inkFaint;
  const fieldGroup: MathBoxView = view.group({ id: "bitpack-field" });
  const field = buildBitpackField(fieldGroup, {
    columns: totalCols,
    rows: totalRows,
    bitsPerSide,
    bitPitch,
    tilePitch,
    pointSize,
    setBitZDisplacement: 0,
    unsetBitZDisplacement: 0,
    colors: { base: neutral, set: neutral, unset: neutral },
    bitSource: uninitializedBitSource(),
    showDisplacementVectors: false,
    paddingCols: padCols,
    paddingRows: padRows,
    paddingFalloffX,
    paddingFalloffY,
    idPrefix: "field",
  });
  const frame = field.frame;

  // Overlays anchor on the core (past the padding), centred in the field.
  const coreOrigin = { col: padCols, row: padRows };
  const coreRect = frame.regionRect(padCols, coreCols, padRows, coreRows);
  const coreExtent: FieldExtent = {
    width: coreRect.x1 - coreRect.x0,
    height: coreRect.y1 - coreRect.y0,
    maxSpan: Math.max(coreRect.x1 - coreRect.x0, coreRect.y1 - coreRect.y0),
  };

  const overlayRoot: MathBoxView = view.group({ id: "layer-overlays" });
  const overlayOptions = {
    palette,
    origin: coreOrigin,
    showWorkItems: options.showWorkItems,
    showSubGroups: options.showSubGroups,
    showWorkGroup: options.showWorkGroup,
  };

  const built = new Map<number, LayerOverlayHandles>();
  let active: LayerOverlayHandles | null = null;
  let cursorStep = 0;

  // --- transition tween ------------------------------------------------------
  let transition: Transition | null = null;
  let rafId: number | null = null;

  const finalize = () => {
    if (!transition) return;
    const { outgoing, incoming } = transition;
    incoming.setTravel(0);
    incoming.setOpacity(1);
    if (outgoing) {
      outgoing.setOpacity(0);
      outgoing.setVisible(false);
      outgoing.setTravel(0);
    }
    transition = null;
  };

  const tick = () => {
    rafId = null;
    if (!transition) return;
    const t = Math.min(1, (performance.now() - transition.start) / transition.duration);
    const e = easeInOut(t);
    const { incoming, outgoing, dir, travel } = transition;
    incoming.setTravel((1 - e) * -dir * travel);
    incoming.setOpacity(e);
    if (outgoing) {
      outgoing.setTravel(e * dir * travel);
      outgoing.setOpacity(1 - e);
    }
    if (t >= 1) finalize();
    else rafId = requestAnimationFrame(tick);
  };

  const startTransition = (outgoing: LayerOverlayHandles | null, incoming: LayerOverlayHandles, dir: number) => {
    // Snap any in-flight transition to its end before starting a new one.
    if (rafId != null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    finalize();

    incoming.setVisible(true);
    if (!outgoing) {
      incoming.setTravel(0);
      incoming.setOpacity(1);
      return;
    }
    incoming.setTravel(-dir * transitionTravel);
    incoming.setOpacity(0);
    transition = {
      outgoing,
      incoming,
      dir,
      start: performance.now(),
      duration: transitionMs,
      travel: transitionTravel,
    };
    rafId = requestAnimationFrame(tick);
  };

  const cursorOf = (handles: LayerOverlayHandles, stepIndex: number): LayerCursor => ({
    stepIndex,
    layer: model.steps[stepIndex]?.layer ?? 0,
    kernelId: handles.kernelId,
    footprint: handles.footprint,
  });

  const showKernel = (kernelId: number): LayerCursor | null => {
    const kernel = model.byId.get(kernelId);
    if (!kernel) return active ? cursorOf(active, cursorStep) : null;

    const newStep = stepIndexOfKernel(model, kernelId);

    let handles = built.get(kernelId);
    if (!handles) {
      handles = buildLayerOverlays(overlayRoot, frame, kernel, { ...overlayOptions, visible: false });
      built.set(kernelId, handles);
    }

    if (active && active.kernelId === kernelId) {
      // Re-selecting the active layer: ensure it is home.
      handles.setVisible(true);
      handles.setTravel(0);
      handles.setOpacity(1);
      cursorStep = newStep;
      return cursorOf(handles, newStep);
    }

    const dir = Math.sign(newStep - cursorStep) || 1;
    startTransition(active, handles, dir);
    active = handles;
    cursorStep = newStep;
    return cursorOf(handles, newStep);
  };

  const showLayer = (stepIndex: number): LayerCursor | null => {
    if (model.steps.length === 0) return null;
    const clamped = Math.max(0, Math.min(model.steps.length - 1, Math.floor(stepIndex)));
    return showKernel(representativeKernelId(model, model.steps[clamped]));
  };

  return {
    showLayer,
    focusKernel: showKernel,
    step: (direction) => showLayer(cursorStep + (direction === "next" ? 1 : -1)),
    stepCount: () => model.steps.length,
    layerIndex: () => cursorStep,
    fieldExtent: () => coreExtent,
    fieldShape: () => ({ rows: coreRows, cols: coreCols }),
    scrollLimitY: () => frame.spanY / 2,
    dispose: () => {
      if (rafId != null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      transition = null;
      for (const handles of built.values()) handles.remove();
      built.clear();
      active = null;
      try {
        overlayRoot.remove();
      } catch {
        /* ignore */
      }
      try {
        fieldGroup.remove();
      } catch {
        /* ignore */
      }
    },
  };
}
