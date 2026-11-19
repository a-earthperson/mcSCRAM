import { useEffect, useRef, type RefObject } from "react";

/**
 * React registers `wheel` as passive, so `preventDefault()` in `onWheel` does not
 * stop ancestor/page scroll. Attach a native non-passive listener instead.
 */
export function useNonPassiveWheel(
  targetRef: RefObject<HTMLElement | null>,
  onWheel: (event: WheelEvent) => void,
  enabled = true,
): void {
  const onWheelRef = useRef(onWheel);
  onWheelRef.current = onWheel;

  useEffect(() => {
    if (!enabled) return undefined;
    const target = targetRef.current;
    if (!target) return undefined;

    const handler = (event: WheelEvent) => {
      event.preventDefault();
      onWheelRef.current(event);
    };

    target.addEventListener("wheel", handler, { passive: false });
    return () => target.removeEventListener("wheel", handler);
  }, [enabled, targetRef]);
}
