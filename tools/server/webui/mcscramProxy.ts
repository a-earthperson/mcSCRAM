import type { Express } from "express";
import { createProxyMiddleware } from "http-proxy-middleware";
import type { IncomingMessage } from "node:http";
import type { Server } from "node:http";
import type { Socket } from "node:net";

export const MCSCRAM_API_PREFIX = "/api/v1";

export function mcscramApiTarget(): string {
  const configured = process.env.MCSCRAM_API_URL?.trim();
  if (configured) return configured;
  const port = process.env.VITE_MCSCRAM_DEV_API_PORT ?? process.env.MCSCRAM_SERVER_PUBLISHED_PORT ?? "8001";
  return `http://127.0.0.1:${port}`;
}

export function mcscramViteProxy() {
  return {
    [MCSCRAM_API_PREFIX]: {
      target: mcscramApiTarget(),
      ws: true,
      changeOrigin: true,
    },
  };
}

function matchesProxyPath(pathname: string): boolean {
  return pathname === MCSCRAM_API_PREFIX || pathname.startsWith(`${MCSCRAM_API_PREFIX}/`);
}

export function attachMcscramProxy(app: Express, httpServer: Server): void {
  const target = mcscramApiTarget();
  const proxy = createProxyMiddleware({
    target,
    changeOrigin: true,
    ws: true,
    logger: console,
  });

  app.use((req, res, next) => {
    if (!matchesProxyPath(req.path)) {
      next();
      return;
    }
    proxy(req, res, next);
  });

  httpServer.on("upgrade", (req: IncomingMessage, socket: Socket, head: Buffer) => {
    const pathname = req.url?.split("?")[0] ?? "";
    if (!matchesProxyPath(pathname)) return;
    proxy.upgrade(req, socket, head);
  });
}
