import type { BenchIndex, BenchModel, BenchPoint, BenchTarget, CompareRow, RatioTone } from "./types";

/** Same thresholds as ``benchmarks.harness.verdict`` (GPU 3%, CPU 5% report-only). */
export const GPU_BLOCK = 0.03;
export const CPU_BLOCK = 0.05;
export const Z_BLOCK = 4;

export const TARGET_LABEL: Record<string, string> = {
  v100s: "V100S",
  rtx3070: "RTX 3070",
  cpu: "CPU (gateway)",
};

export function isGpuTarget(target: string): boolean {
  return target === "v100s" || target === "rtx3070";
}

export function pointsForTarget(index: BenchIndex, target: string): BenchPoint[] {
  return index.points
    .filter((point) => point.target === target)
    .slice()
    .sort((a, b) => String(a.created_utc ?? "").localeCompare(String(b.created_utc ?? "")));
}

export function latestByChannel(
  points: BenchPoint[],
): { release: BenchPoint | null; develop: BenchPoint | null } {
  let release: BenchPoint | null = null;
  let develop: BenchPoint | null = null;
  for (const point of points) {
    if (point.channel === "release") release = point;
    else if (point.channel === "develop") develop = point;
  }
  return { release, develop };
}

export function modelIds(points: BenchPoint[]): string[] {
  const ids = new Set<string>();
  for (const point of points) {
    for (const model of point.models) ids.add(model.id);
  }
  return [...ids].sort();
}

function modelOn(point: BenchPoint | null, id: string): BenchModel | null {
  if (!point) return null;
  return point.models.find((model) => model.id === id) ?? null;
}

function toneForRatio(value: number | null, gpu: boolean): RatioTone {
  if (value === null || !Number.isFinite(value)) return "neutral";
  const limit = gpu ? 1 + GPU_BLOCK : 1 + CPU_BLOCK;
  if (gpu) {
    if (value > limit) return "block";
    if (value > 1) return "warn";
    return "pass";
  }
  if (value > limit) return "warn";
  if (value > 1) return "warn";
  return "pass";
}

function ratio(head: number | null | undefined, base: number | null | undefined): number | null {
  if (head == null || base == null || !Number.isFinite(head) || !Number.isFinite(base) || base === 0) {
    return null;
  }
  return head / base;
}

function pushRow(
  rows: CompareRow[],
  modelId: string,
  metric: string,
  release: number | null | undefined,
  develop: number | null | undefined,
  tone: RatioTone,
  note: string | null = null,
): void {
  const rel = release ?? null;
  const dev = develop ?? null;
  rows.push({
    modelId,
    metric,
    release: rel,
    develop: dev,
    ratio: ratio(dev, rel),
    tone,
    note,
  });
}

export function compareReleaseDevelop(release: BenchPoint | null, develop: BenchPoint | null): CompareRow[] {
  if (!release || !develop) return [];
  const gpu = isGpuTarget(String(release.target));
  const ids = modelIds([release, develop]);
  const rows: CompareRow[] = [];
  for (const id of ids) {
    const rel = modelOn(release, id);
    const dev = modelOn(develop, id);
    if (gpu) {
      const rTotal = rel?.perf?.gpu_total_ms ?? null;
      const dTotal = dev?.perf?.gpu_total_ms ?? null;
      pushRow(rows, id, "gpu_total_ms", rTotal, dTotal, toneForRatio(ratio(dTotal, rTotal), true));
      const rKer = rel?.perf?.gpu_kernel_ms ?? null;
      const dKer = dev?.perf?.gpu_kernel_ms ?? null;
      pushRow(rows, id, "gpu_kernel_ms", rKer, dKer, toneForRatio(ratio(dKer, rKer), true));
    } else {
      const rCpu = rel?.perf?.cpu_sample_s ?? null;
      const dCpu = dev?.perf?.cpu_sample_s ?? null;
      pushRow(
        rows,
        id,
        "cpu_sample_s",
        rCpu,
        dCpu,
        toneForRatio(ratio(dCpu, rCpu), false),
        "CPU is report-only (#124); ±3–7% noise on gateway.",
      );
    }
    const rZ = rel?.accuracy?.z ?? null;
    const dZ = dev?.accuracy?.z ?? null;
    const zTone: RatioTone =
      (dZ != null && Math.abs(dZ) > Z_BLOCK) || (rZ != null && Math.abs(rZ) > Z_BLOCK) ? "block" : "pass";
    const accNote = [rel?.accuracy?.verdict, dev?.accuracy?.verdict].filter(Boolean).join(" / ") || null;
    pushRow(rows, id, "accuracy_z", rZ, dZ, zTone, accNote);
  }
  return rows;
}

export function seriesForModel(
  points: BenchPoint[],
  modelId: string,
  metric: "gpu_total_ms" | "gpu_kernel_ms" | "cpu_sample_s" | "accuracy_z",
): { version: string; created_utc: string | null; value: number | null; flags: BenchPoint["flags"]; channel: string }[] {
  return points.map((point) => {
    const model = modelOn(point, modelId);
    let value: number | null = null;
    if (metric === "accuracy_z") value = model?.accuracy?.z ?? null;
    else if (metric === "gpu_total_ms") value = model?.perf?.gpu_total_ms ?? null;
    else if (metric === "gpu_kernel_ms") value = model?.perf?.gpu_kernel_ms ?? null;
    else value = model?.perf?.cpu_sample_s ?? null;
    return {
      version: point.version,
      created_utc: point.created_utc,
      value,
      flags: point.flags,
      channel: point.channel,
    };
  });
}

export function flagSummary(flags: BenchPoint["flags"] | undefined): string[] {
  if (!flags) return [];
  const labels: string[] = [];
  if (flags.image_changed) labels.push("image");
  if (flags.driver_changed) labels.push("driver");
  if (flags.recipe_changed) labels.push("recipe");
  if (flags.runner_changed) labels.push("runner");
  return labels;
}

export function targetOptions(index: BenchIndex): BenchTarget[] {
  const present = new Set(index.points.map((p) => p.target));
  return (["v100s", "rtx3070", "cpu"] as BenchTarget[]).filter((t) => present.has(t));
}
