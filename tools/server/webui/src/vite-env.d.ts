/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BASE_PATH?: string;
  readonly VITE_BENCH_DATA_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface Window {
  __FUZZYBITS_BENCH_DATA_URL__?: string;
}

declare module "plotly.js-dist-min" {
  const Plotly: {
    react: (element: HTMLElement, data: unknown[], layout?: Record<string, unknown>, config?: Record<string, unknown>) => Promise<unknown>;
    purge: (element: HTMLElement) => void;
  };
  export default Plotly;
}
