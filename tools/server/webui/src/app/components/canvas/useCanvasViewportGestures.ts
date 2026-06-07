import {
  useCallback,
  useEffect,
  useRef,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";
import type { CanvasViewportConfig } from "./canvasViewport";
import { firstTwoPointers, pointerDistance, pointerMidpoint } from "./canvasPointerGestures";
import {
  panViewTransform,
  pinchZoomAtCanvasMidpoint,
  wheelZoomAtCanvasPoint,
} from "./canvasViewportTransform";
import type { ViewTransform } from "./types";
import { useNonPassiveWheel } from "./useNonPassiveWheel";

/** Ignore pinch deltas until finger separation exceeds this (avoids jitter). */
const MIN_PINCH_DISTANCE_PX = 24;

export type CanvasViewportGestureHandlers = Pick<
  React.JSX.IntrinsicElements["canvas"],
  "onPointerDown" | "onPointerMove" | "onPointerUp" | "onPointerCancel"
>;

export type UseCanvasViewportGesturesOptions = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  viewport: CanvasViewportConfig;
  getView: () => ViewTransform;
  setView: (update: (prev: ViewTransform) => ViewTransform) => void;
  enabled?: boolean;
};

/**
 * Wheel zoom (trackpad / mouse), single-finger pan, and two-finger pan + pinch zoom.
 * Spread the returned handlers onto `.fault-tree-canvas`.
 */
export function useCanvasViewportGestures({
  canvasRef,
  viewport,
  getView,
  setView,
  enabled = true,
}: UseCanvasViewportGesturesOptions): CanvasViewportGestureHandlers {
  const getViewRef = useRef(getView);
  const setViewRef = useRef(setView);
  getViewRef.current = getView;
  setViewRef.current = setView;

  const pointersRef = useRef(new Map<number, { clientX: number; clientY: number }>());
  const panRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startOffsetX: number;
    startOffsetY: number;
  } | null>(null);
  const twoPointerRef = useRef<{
    lastDistance: number;
    lastMidX: number;
    lastMidY: number;
  } | null>(null);

  const handleWheel = useCallback((event: WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setViewRef.current((prev) => wheelZoomAtCanvasPoint(prev, canvas, event, viewport));
  }, [canvasRef, viewport]);

  useNonPassiveWheel(canvasRef, handleWheel, enabled);

  const beginTwoPointerGesture = useCallback(() => {
    panRef.current = null;
    const pair = firstTwoPointers(pointersRef.current);
    if (!pair) {
      twoPointerRef.current = null;
      return;
    }
    const [p0, p1] = pair;
    const mid = pointerMidpoint(p0, p1);
    twoPointerRef.current = {
      lastDistance: pointerDistance(p0, p1),
      lastMidX: mid.clientX,
      lastMidY: mid.clientY,
    };
  }, []);

  const applyTwoPointerMove = useCallback(() => {
    const canvas = canvasRef.current;
    const gesture = twoPointerRef.current;
    const pair = firstTwoPointers(pointersRef.current);
    if (!canvas || !gesture || !pair) return;

    const [p0, p1] = pair;
    const mid = pointerMidpoint(p0, p1);
    const dx = mid.clientX - gesture.lastMidX;
    const dy = mid.clientY - gesture.lastMidY;
    const distance = pointerDistance(p0, p1);

    let ratio = 1;
    if (
      distance >= MIN_PINCH_DISTANCE_PX &&
      gesture.lastDistance >= MIN_PINCH_DISTANCE_PX
    ) {
      ratio = distance / gesture.lastDistance;
    }

    gesture.lastMidX = mid.clientX;
    gesture.lastMidY = mid.clientY;
    gesture.lastDistance = distance;

    if (dx === 0 && dy === 0 && ratio === 1) return;

    setViewRef.current((prev) => {
      let next = dx !== 0 || dy !== 0 ? panViewTransform(prev, dx, dy) : prev;
      if (ratio !== 1) {
        next = pinchZoomAtCanvasMidpoint(
          next,
          canvas,
          p0.clientX,
          p0.clientY,
          p1.clientX,
          p1.clientY,
          ratio,
          viewport,
        );
      }
      return next;
    });
  }, [canvasRef, viewport]);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!enabled) return;

      event.currentTarget.setPointerCapture(event.pointerId);
      pointersRef.current.set(event.pointerId, {
        clientX: event.clientX,
        clientY: event.clientY,
      });

      if (pointersRef.current.size === 1) {
        const view = getViewRef.current();
        panRef.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          startOffsetX: view.offsetX,
          startOffsetY: view.offsetY,
        };
        return;
      }

      if (pointersRef.current.size === 2) {
        beginTwoPointerGesture();
      }
    },
    [beginTwoPointerGesture, enabled],
  );

  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!enabled) return;

      const tracked = pointersRef.current.get(event.pointerId);
      if (!tracked) return;

      tracked.clientX = event.clientX;
      tracked.clientY = event.clientY;

      if (pointersRef.current.size >= 2) {
        applyTwoPointerMove();
        return;
      }

      const pan = panRef.current;
      if (!pan || pan.pointerId !== event.pointerId) return;

      const offsetX = pan.startOffsetX + event.clientX - pan.startX;
      const offsetY = pan.startOffsetY + event.clientY - pan.startY;
      setViewRef.current((prev) => ({ ...prev, offsetX, offsetY }));
    },
    [applyTwoPointerMove, enabled],
  );

  const onPointerEnd = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    pointersRef.current.delete(event.pointerId);

    if (panRef.current?.pointerId === event.pointerId) {
      panRef.current = null;
    }

    if (pointersRef.current.size < 2) {
      twoPointerRef.current = null;
    } else if (pointersRef.current.size === 2) {
      beginTwoPointerGesture();
    }
  }, [beginTwoPointerGesture]);

  useEffect(() => {
    if (enabled) return;
    pointersRef.current.clear();
    panRef.current = null;
    twoPointerRef.current = null;
  }, [enabled]);

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp: onPointerEnd,
    onPointerCancel: onPointerEnd,
  };
}
