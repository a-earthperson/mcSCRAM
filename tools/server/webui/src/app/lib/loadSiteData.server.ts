import { readFile } from "node:fs/promises";
import path from "node:path";
import type {
  Artifacts,
  CompilerData,
  ConvergenceView,
  DeviceSummary,
  ExactResult,
  HardwareData,
  ModelCard,
  NarrativeCard,
  PerformanceRun,
  SetIndexEntry,
  SiteData,
  SpeedupData,
  Summary,
  Tallies,
  TraceIndexData,
} from "../types";

export async function loadSiteDataFromDisk(): Promise<SiteData> {
  const dataDir = process.env.SITE_DATA_DIR ?? path.resolve(process.cwd(), "docs/data");

  async function readJson<T>(name: string): Promise<T> {
    const raw = await readFile(path.join(dataDir, name), "utf-8");
    return JSON.parse(raw) as T;
  }

  const [
    summary,
    narrative,
    models,
    performance,
    devices,
    speedups,
    compiler,
    convergence,
    traces,
    setIndex,
    artifacts,
    tallies,
    exactResults,
    hardware,
  ] = await Promise.all([
    readJson<Summary>("summary.json"),
    readJson<NarrativeCard[]>("narrative.json"),
    readJson<ModelCard[]>("models.json"),
    readJson<PerformanceRun[]>("performance.json"),
    readJson<DeviceSummary[]>("device-summaries.json"),
    readJson<SpeedupData>("speedups.json"),
    readJson<CompilerData>("compiler.json"),
    readJson<ConvergenceView>("convergence.json"),
    readJson<TraceIndexData>("traces-index.json"),
    readJson<Record<string, SetIndexEntry>>("set-index.json"),
    readJson<Artifacts>("artifacts.json"),
    readJson<Tallies>("tallies.json"),
    readJson<ExactResult[]>("exact-results.json"),
    readJson<HardwareData>("hardware.json"),
  ]);

  return {
    summary,
    narrative,
    models,
    performance,
    devices,
    speedups,
    compiler,
    convergence,
    traces,
    setIndex,
    artifacts,
    tallies,
    exactResults,
    hardware,
  };
}
