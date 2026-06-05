import { useCallback, useEffect, useState, type RefObject } from "react";

type FullscreenState = {
  /** Whether `target` is the active fullscreen element. */
  active: boolean;
  /** Whether the Fullscreen API is available in this browser. */
  supported: boolean;
  enter: () => Promise<void>;
  exit: () => Promise<void>;
  toggle: () => Promise<void>;
};

function readActive(target: HTMLElement | null): boolean {
  return Boolean(target && document.fullscreenElement === target);
}

/**
 * Imperative Fullscreen API wrapper for a single DOM subtree.
 *
 * Keeps React state in sync via `fullscreenchange` so callers can style or
 * resize without remounting heavy canvases (WebGL, MathBox, etc.).
 */
export function useElementFullscreen(targetRef: RefObject<HTMLElement | null>): FullscreenState {
  const [active, setActive] = useState(false);

  useEffect(() => {
    const target = targetRef.current;
    if (!target) return undefined;

    const sync = () => setActive(readActive(target));
    sync();
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, [targetRef]);

  const supported = typeof document !== "undefined" && document.fullscreenEnabled;

  const enter = useCallback(async () => {
    const target = targetRef.current;
    if (!target || readActive(target)) return;
    await target.requestFullscreen();
  }, [targetRef]);

  const exit = useCallback(async () => {
    if (!document.fullscreenElement) return;
    await document.exitFullscreen();
  }, []);

  const toggle = useCallback(async () => {
    if (readActive(targetRef.current)) await exit();
    else await enter();
  }, [enter, exit, targetRef]);

  return { active, supported, enter, exit, toggle };
}
