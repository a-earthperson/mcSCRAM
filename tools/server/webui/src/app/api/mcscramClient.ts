/** HTTP/WebSocket base URLs for the optional live mcSCRAM server. */

const API_BASE_PATH = "/api/v1";

function apiPath(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (normalized === API_BASE_PATH || normalized.startsWith(`${API_BASE_PATH}/`)) {
    return normalized;
  }
  return `${API_BASE_PATH}${normalized}`;
}

export function mcscramWebSocketUrl({ replay = false }: { replay?: boolean } = {}): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const base = `${protocol}//${window.location.host}${apiPath("/api/v1/ws")}`;

  if (!replay) return base;
  const url = new URL(base);
  url.searchParams.set("replay", "1");
  return url.toString();
}

export function mcscramHttpUrl(path: string, searchParams?: Record<string, string>): string {
  const url = new URL(mcscramWebSocketUrl());
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = apiPath(path);
  url.search = "";
  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      url.searchParams.set(key, value);
    }
  }
  return url.toString();
}
