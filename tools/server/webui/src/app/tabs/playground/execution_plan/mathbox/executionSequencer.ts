/**
 * Kernel scene controller + execution-order sequencer.
 *
 * The controller renders *one* kernel window at a time using a build-once /
 * visibility-toggle strategy: a kernel's subtree is created lazily the first
 * time it is shown, then only its `visible` flag flips on subsequent switches.
 * This caps GPU cost to "kernels the user actually inspects" and never rebuilds
 * heavy resources on selection — so switching kernels stays allocation-free.
 *
 * The sequencer layers the pure execution schedule (`ExecutionModel.steps`) on
 * top of the controller: stepping advances through layers in topological order,
 * showing each layer's representative kernel. This is how the execution layer
 * "declares ordering" without knowing anything about MathBox.
 */
import type { ExecutionModel, ExecutionStep, KernelModel } from "../sycl";
import { type KernelSceneHandles, type KernelSceneOptions, buildKernelScene } from "./kernelScene";
import type { MathBoxView } from "./types";

export type KernelSceneController = {
  /** Show a kernel (building its subtree on first use). Returns its handles. */
  show: (kernelId: number) => KernelSceneHandles | null;
  /** Currently visible kernel handles, if any. */
  current: () => KernelSceneHandles | null;
  /** Whether a kernel's subtree has been built. */
  has: (kernelId: number) => boolean;
  /** Hide all kernels and drop references. */
  dispose: () => void;
};

export function createKernelSceneController(
  view: MathBoxView,
  model: ExecutionModel,
  options: Omit<KernelSceneOptions, "visible">,
): KernelSceneController {
  const built = new Map<number, KernelSceneHandles>();
  let active: KernelSceneHandles | null = null;

  const show = (kernelId: number): KernelSceneHandles | null => {
    const kernel = model.byId.get(kernelId);
    if (!kernel) return active;

    if (active && active.kernelId !== kernelId) {
      active.setVisible(false);
    }

    let handles = built.get(kernelId);
    if (!handles) {
      handles = buildKernelScene(view, kernel, { ...options, visible: true });
      built.set(kernelId, handles);
    } else {
      handles.setVisible(true);
    }

    active = handles;
    return handles;
  };

  return {
    show,
    current: () => active,
    has: (kernelId: number) => built.has(kernelId),
    dispose: () => {
      for (const handles of built.values()) handles.setVisible(false);
      built.clear();
      active = null;
    },
  };
}

// ---------------------------------------------------------------------------
//  Sequencer over the execution schedule
// ---------------------------------------------------------------------------

/** Representative kernel for a step: first basic-event, else first kernel. */
export function stepKernelId(model: ExecutionModel, step: ExecutionStep): number {
  const basic = step.kernelIds.find((id) => model.byId.get(id)?.kind === "basic_event");
  return basic ?? step.kernelIds[0] ?? model.defaultKernelId;
}

export type ExecutionSequencer = {
  stepCount: number;
  index: () => number;
  step: () => ExecutionStep | null;
  goTo: (stepIndex: number) => KernelSceneHandles | null;
  next: () => KernelSceneHandles | null;
  prev: () => KernelSceneHandles | null;
};

export function createExecutionSequencer(
  controller: KernelSceneController,
  model: ExecutionModel,
): ExecutionSequencer {
  const steps = model.steps;
  let cursor = 0;

  const goTo = (stepIndex: number): KernelSceneHandles | null => {
    if (steps.length === 0) return controller.current();
    cursor = Math.max(0, Math.min(steps.length - 1, stepIndex));
    return controller.show(stepKernelId(model, steps[cursor]));
  };

  return {
    stepCount: steps.length,
    index: () => cursor,
    step: () => steps[cursor] ?? null,
    goTo,
    next: () => goTo(cursor + 1),
    prev: () => goTo(cursor - 1),
  };
}

/** Locate the step that launches a given kernel (for syncing UI selection). */
export function stepIndexForKernel(model: ExecutionModel, kernelId: number): number {
  const kernel: KernelModel | undefined = model.byId.get(kernelId);
  if (!kernel) return 0;
  const index = model.steps.findIndex((step) => step.layer === kernel.layer);
  return index >= 0 ? index : 0;
}
