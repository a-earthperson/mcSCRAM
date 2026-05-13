import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/mcSCRAM-benchmarks/",
  root: "site",
  publicDir: false,
  plugins: [react()],
  build: {
    outDir: "../docs",
    emptyOutDir: true,
    sourcemap: true,
    chunkSizeWarningLimit: 5000,
  },
});
