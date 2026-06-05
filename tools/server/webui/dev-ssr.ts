import express from "express";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer as createViteServer } from "vite";
import { attachMcscramProxy, mcscramApiTarget } from "./mcscramProxy.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const siteRoot = path.resolve(repoRoot, "site");
const port = Number(process.env.PORT ?? "5173");

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
  const httpServer = http.createServer(app);
  const vite = await createViteServer({
    configFile: path.resolve(repoRoot, "vite.config.ts"),
    server: { middlewareMode: true },
    appType: "custom",
  });

  attachMcscramProxy(app, httpServer);

  app.use(async (req, res, next) => {
    if (req.method !== "GET" && req.method !== "HEAD") {
      next();
      return;
    }

    const accept = req.headers.accept ?? "";
    if (!accept.includes("text/html")) {
      next();
      return;
    }

    try {
      const url = req.originalUrl;
      const template = await fs.readFile(path.resolve(siteRoot, "index.html"), "utf-8");
      const transformed = await vite.transformIndexHtml(url, template);
      const { render } = await vite.ssrLoadModule("/src/entry-server.tsx");
      const { html, siteData } = await render(url);
      const document = injectSsrPayload(transformed, html, siteData);
      res.status(200).set({ "Content-Type": "text/html" }).send(document);
    } catch (error) {
      if (error instanceof Error) vite.ssrFixStacktrace(error);
      next(error);
    }
  });

  app.use(vite.middlewares);

  httpServer.listen(port, "0.0.0.0", () => {
    console.log(`SSR dev server on http://0.0.0.0:${port}`);
    console.log(`Proxying mcSCRAM API routes to ${mcscramApiTarget()}`);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
