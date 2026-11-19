import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const appRoot = path.dirname(fileURLToPath(import.meta.url));

function devProxyTarget(): string {
  return (
    process.env.FUZZYBITS_SERVER_URL?.trim() ||
    process.env.MCSCRAM_API_URL?.trim() ||
    "http://127.0.0.1:8000"
  );
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, appRoot, "VITE_");
  const isDev = mode === "development";
  const target = devProxyTarget();

  return {
    base: env.VITE_BASE_PATH || "/",
    root: ".",
    publicDir: false,
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      fs: { allow: [appRoot] },
      proxy: {
        "/api/v1": { target, ws: true, changeOrigin: true },
      },
    },
    build: {
      outDir: "dist",
      emptyOutDir: true,
      manifest: true,
      minify: !isDev,
      sourcemap: true,
      chunkSizeWarningLimit: 5000,
    },
    preview: {
      host: true,
      port: 5173,
      strictPort: true,
      allowedHosts: true,
      proxy: {
        "/api/v1": { target, ws: true, changeOrigin: true },
      },
    },
  };
});
