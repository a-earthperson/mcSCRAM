import express from "express";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { attachMcscramProxy, mcscramApiTarget } from "./mcscramProxy.js";

type RenderMode = "csr" | "ssr";
type SsrEntry = {
  render: (url: string) => Promise<{ html: string; siteData: unknown }>;
};

const repoRoot = process.env.REPO_ROOT ? path.resolve(process.env.REPO_ROOT) : process.cwd();
const siteRoot = path.resolve(repoRoot, "site");
const clientRoot = path.resolve(siteRoot, ".dist/client");
const docsRoot = path.resolve(repoRoot, "docs");
const serverEntry = path.resolve(siteRoot, ".dist/server/entry-server.js");
const port = Number(process.env.PORT ?? "5173");
const renderMode = renderModeFromEnv();
const basePath = (process.env.VITE_BASE_PATH ?? "/").trim();
const mountPath = basePath === "/" ? "/" : basePath.endsWith("/") ? basePath.slice(0, -1) : basePath;

function renderModeFromEnv(): RenderMode {
  const raw = (process.env.FRONTEND_RENDER_MODE ?? "ssr").toLowerCase();
  if (raw === "csr" || raw === "ssr") return raw;
  throw new Error(`Invalid FRONTEND_RENDER_MODE=${raw}; expected "csr" or "ssr".`);
}

function mountUnderBase(pathname: string): string {
  const suffix = pathname.startsWith("/") ? pathname : `/${pathname}`;
  return mountPath === "/" ? suffix : `${mountPath}${suffix}`;
}

function injectSsrPayload(template: string, html: string, siteData: unknown): string {
  const payload = JSON.stringify(siteData).replace(/</g, "\\u003c");
  return template
    .replace("<!--ssr-outlet-->", html)
    .replace(
      '<script type="module"',
      `<script>window.__SITE_DATA__=${payload}</script>\n    <script type="module"`,
    );
}

async function main() {
  const app = express();
  const template = fs.readFileSync(path.resolve(clientRoot, "index.html"), "utf-8");
  const ssr: SsrEntry | null = renderMode === "ssr" ? await import(serverEntry) as SsrEntry : null;
  const httpServer = http.createServer(app);

  attachMcscramProxy(app, httpServer);
  app.use(mountPath, express.static(clientRoot, { index: false }));
  app.use(mountUnderBase("/data"), express.static(path.resolve(docsRoot, "data"), { index: false }));
  app.use(mountUnderBase("/artifacts"), express.static(path.resolve(docsRoot, "artifacts"), { index: false }));

  app.use(async (req, res, next) => {
    if (req.method !== "GET" && req.method !== "HEAD") {
      next();
      return;
    }

    const accept = req.headers.accept ?? "";
    if (!accept.includes("text/html") && !accept.includes("*/*")) {
      next();
      return;
    }

    try {
      if (!ssr) {
        const document = template.replace("<!--ssr-outlet-->", "");
        res.status(200).set({ "Content-Type": "text/html" }).send(document);
        return;
      }
      const { html, siteData } = await ssr.render(req.originalUrl);
      res.status(200).set({ "Content-Type": "text/html" }).send(injectSsrPayload(template, html, siteData));
      return;
    } catch (error) {
      console.error("SSR render failed:", error);
      next(error);
    }
  });

  httpServer.listen(port, "0.0.0.0", () => {
    console.log(`${renderMode.toUpperCase()} server listening on http://0.0.0.0:${port}${mountPath === "/" ? "" : mountPath}`);
    console.log(`Proxying mcSCRAM API routes to ${mcscramApiTarget()}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
