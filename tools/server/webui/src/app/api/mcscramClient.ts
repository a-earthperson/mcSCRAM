/** HTTP/WebSocket base URLs for the optional live mcSCRAM server. */

export function mcscramWebSocketUrl({ replay = false }: { replay?: boolean } = {}): string {
  const configured = import.meta.env.VITE_MCSCRAM_WS_URL as string | undefined;
  const base = configured || (() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname || "127.0.0.1";
    const port = import.meta.env.VITE_MCSCRAM_WS_PORT as string | undefined;
    const path = (import.meta.env.VITE_MCSCRAM_WS_PATH as string | undefined) || "/ws";
    const authority = port ? `${host}:${port}` : window.location.host;
    return `${protocol}//${authority}${path}`;
  })();

  if (!replay) return base;
  const url = new URL(base);
  url.searchParams.set("replay", "1");
  return url.toString();
}

export function mcscramHttpUrl(path: string, searchParams?: Record<string, string>): string {
  const url = new URL(mcscramWebSocketUrl());
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = path;
  url.search = "";
  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      url.searchParams.set(key, value);
    }
  }
  return url.toString();
}
