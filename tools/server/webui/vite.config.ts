import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { mcscramViteProxy } from "./site/mcscramProxy";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));
const lazyTabs = path.resolve(repoRoot, "site/src/app/tabs/lazyTabs.tsx");
const lazyTabsSsr = path.resolve(repoRoot, "site/src/app/tabs/lazyTabs.ssr.tsx");

export default defineConfig(({ mode, isSsrBuild }) => {
  const env = loadEnv(mode, repoRoot, "VITE_");
  const isDev = mode === "development";

  return {
    base: env.VITE_BASE_PATH || "/",
    root: "site",
    // Serve generated benchmark JSON/plots from docs/ during `vite` dev only.
    publicDir: isDev ? path.join(repoRoot, "docs") : false,
    plugins: [react()],
    resolve: {
      alias: isSsrBuild ? [{ find: lazyTabs, replacement: lazyTabsSsr }] : undefined,
    },
    ssr: {
      noExternal: ["react-router"],
    },
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      fs: {
        allow: [repoRoot],
      },
      proxy: mcscramViteProxy(),
    },
    build: {
      outDir: isSsrBuild ? ".dist/server" : ".dist/client",
      emptyOutDir: !isSsrBuild,
      manifest: !isSsrBuild,
      minify: isDev ? false : "esbuild",
      sourcemap: true,
      chunkSizeWarningLimit: 5000,
      rollupOptions: isSsrBuild
        ? {
            input: "src/entry-server.tsx",
          }
        : undefined,
    },
    preview: {
      host: true,
      port: 5173,
      strictPort: true,
      allowedHosts: true,
    },
  };
});
