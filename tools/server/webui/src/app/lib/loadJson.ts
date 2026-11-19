export async function loadJson<T>(name: string): Promise<T> {
  const basePath = `${import.meta.env.BASE_URL || "/"}`.trim();
  const normalizedBase = basePath.endsWith("/") ? basePath : `${basePath}/`;
  const url = new URL(`data/${name}`, new URL(normalizedBase, window.location.origin));

  try {
    const response = await fetch(url.toString());
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return (await response.json()) as T;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to load ${name} from ${url.toString()}: ${message}`);
  }
}
