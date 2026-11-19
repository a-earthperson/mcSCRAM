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
import { loadJson } from "./loadJson";

export async function loadSiteData(): Promise<SiteData> {
  if (window.__SITE_DATA__) return window.__SITE_DATA__;

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
    loadJson<Summary>("summary.json"),
    loadJson<NarrativeCard[]>("narrative.json"),
    loadJson<ModelCard[]>("models.json"),
    loadJson<PerformanceRun[]>("performance.json"),
    loadJson<DeviceSummary[]>("device-summaries.json"),
    loadJson<SpeedupData>("speedups.json"),
    loadJson<CompilerData>("compiler.json"),
    loadJson<ConvergenceView>("convergence.json"),
    loadJson<TraceIndexData>("traces-index.json"),
    loadJson<Record<string, SetIndexEntry>>("set-index.json"),
    loadJson<Artifacts>("artifacts.json"),
    loadJson<Tallies>("tallies.json"),
    loadJson<ExactResult[]>("exact-results.json"),
    loadJson<HardwareData>("hardware.json"),
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
