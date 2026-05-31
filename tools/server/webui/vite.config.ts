import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const isDev = mode === "development";

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
      allowedHosts: ["mc.earthperson.org", "www.mc.earthperson.org"],
    },
  };
});
