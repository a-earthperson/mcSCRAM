import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repoRoot, "VITE_");
  const isDev = mode === "development";
  const apiTarget = `http://127.0.0.1:${env.VITE_MCSCRAM_DEV_API_PORT || "8001"}`;
  const mcscramProxy = isDev
    ? {
        "/ws": { target: apiTarget, ws: true, changeOrigin: true },
        "/model": { target: apiTarget, changeOrigin: true },
        "/healthz": { target: apiTarget, changeOrigin: true },
        "/status": { target: apiTarget, changeOrigin: true },
        "/reset": { target: apiTarget, changeOrigin: true },
        "/start": { target: apiTarget, changeOrigin: true },
        "/stop": { target: apiTarget, changeOrigin: true },
      }
    : undefined;

  return {
    base: env.VITE_BASE_PATH || "/",
    root: "site",
    // Serve generated benchmark JSON/plots from docs/ during `vite` dev only.
    publicDir: isDev ? path.join(repoRoot, "docs") : false,
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      fs: {
        allow: [repoRoot],
      },
      proxy: mcscramProxy,
    },
    build: {
      outDir: "../docs",
      emptyOutDir: true,
      sourcemap: true,
      chunkSizeWarningLimit: 5000,
    },
    preview: {
      host: true,
      port: 5173,
      strictPort: true,
      allowedHosts: true,
    },
  };
});
