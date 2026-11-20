import type { BenchIndex } from "./types";

export const DEFAULT_BENCH_INDEX_URL =
  "https://fuzzybit-org.github.io/fuzzybits/bench/v1/index.json";

const SAME_ORIGIN_INDEX = "/api/v1/bench/v1/index.json";

export function resolveBenchIndexUrl(): string | null {
  const runtime = typeof window !== "undefined" ? window.__FUZZYBITS_BENCH_DATA_URL__ : undefined;
  if (typeof runtime === "string" && runtime.trim()) return runtime.trim();
  const env = import.meta.env.VITE_BENCH_DATA_URL;
  if (typeof env === "string" && env.trim()) return env.trim();
  return null;
}

function candidateUrls(): string[] {
  const override = resolveBenchIndexUrl();
  if (override) return [override];
  return [SAME_ORIGIN_INDEX, DEFAULT_BENCH_INDEX_URL];
}

export class BenchDataError extends Error {
  constructor(message: string, readonly urls: string[]) {
    super(message);
    this.name = "BenchDataError";
  }
}

export async function loadBenchIndex(signal?: AbortSignal): Promise<BenchIndex> {
  const urls = candidateUrls();
  const errors: string[] = [];
  for (const url of urls) {
    try {
      const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
      if (!response.ok) {
        errors.push(`${url} → HTTP ${response.status}`);
        continue;
      }
      const payload = (await response.json()) as BenchIndex;
      if (!payload || !Array.isArray(payload.points)) {
        errors.push(`${url} → not a history index`);
        continue;
      }
      return payload;
    } catch (err) {
      if (signal?.aborted) throw err;
      errors.push(`${url} → ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  throw new BenchDataError(
    "Benchmark history is offline. The rest of the UI is unaffected.",
    urls,
  );
}
