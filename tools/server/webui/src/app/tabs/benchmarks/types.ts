export const BENCH_TARGETS = ["v100s", "rtx3070", "cpu"] as const;
export type BenchTarget = (typeof BENCH_TARGETS)[number];

export type BenchChannel = "release" | "develop";

export type BenchFlags = {
  recipe_changed: boolean;
  driver_changed: boolean;
  runner_changed: boolean;
  image_changed: boolean;
};

export type AccuracyMetrics = {
  z: number | null;
  verdict: string | null;
  status: string | null;
  estimate: number | null;
  n: number | null;
  hits: number | null;
  expected_hits: number | null;
  p_ref: number | null;
};

export type PerfMetrics = {
  gpu_total_ms: number | null;
  gpu_kernel_ms: number | null;
  cpu_sample_s: number | null;
  wall_s: number | null;
  verdict: string | null;
  status: string | null;
};

export type BenchModel = {
  id: string;
  accuracy?: AccuracyMetrics;
  perf?: PerfMetrics;
  compile?: { verdict: string | null; status: string | null; compile_wall_s: number | null };
};

export type BenchPoint = {
  version: string;
  channel: BenchChannel;
  target: BenchTarget | string;
  dataset: string;
  created_utc: string | null;
  path: string;
  runner: string | null;
  runner_name?: string | null;
  device: string | null;
  driver: string | null;
  image: string | null;
  image_tag: string | null;
  image_digest: string | null;
  image_pin_source: string | null;
  recipe: string | null;
  commit: string | null;
  backend: string | null;
  verdict: string | null;
  flags: BenchFlags;
  models: BenchModel[];
};

export type BenchIndex = {
  schema_version: number;
  dataset: string;
  generated_utc: string;
  source_url?: string;
  n_points: number;
  points: BenchPoint[];
};

export type RatioTone = "pass" | "warn" | "block" | "neutral";

export type CompareRow = {
  modelId: string;
  metric: string;
  release: number | null;
  develop: number | null;
  ratio: number | null;
  tone: RatioTone;
  note: string | null;
};
