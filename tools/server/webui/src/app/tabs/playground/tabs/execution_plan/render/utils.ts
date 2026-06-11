/** Release browser WebGL contexts held by canvases under `container`. */
export function releaseWebGlContexts(container: HTMLElement): void {
  for (const canvas of container.querySelectorAll("canvas")) {
    loseWebGlContext(canvas);
  }
}

function loseWebGlContext(canvas: HTMLCanvasElement): void {
  for (const contextId of ["webgl2", "webgl", "experimental-webgl"] as const) {
    const gl = canvas.getContext(contextId);
    if (!gl || !("getExtension" in gl)) continue;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

export function cleanupContainer(container: HTMLElement): void {
  try {
    releaseWebGlContexts(container);
    while (container.firstChild) container.removeChild(container.firstChild);
  } catch {
    /* ignore */
  }
}
