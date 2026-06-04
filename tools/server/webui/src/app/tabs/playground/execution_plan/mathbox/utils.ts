export function cleanupContainer(container: HTMLElement): void {
  try {
    while (container.firstChild) container.removeChild(container.firstChild);
  } catch {
    /* ignore */
  }
}
