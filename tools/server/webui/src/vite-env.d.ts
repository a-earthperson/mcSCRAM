/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BASE_PATH?: string;
  readonly VITE_MCSCRAM_WS_URL?: string;
  readonly VITE_MCSCRAM_WS_PORT?: string;
  readonly VITE_MCSCRAM_WS_PATH?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "plotly.js-dist-min" {
  const Plotly: {
    react: (element: HTMLElement, data: unknown[], layout?: Record<string, unknown>, config?: Record<string, unknown>) => Promise<unknown>;
    purge: (element: HTMLElement) => void;
  };
  export default Plotly;
}
