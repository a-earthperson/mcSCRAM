import { useMemo } from "react";
import type { PerformanceRun, SiteData } from "../types";
import { formatInteger, formatNumber } from "../lib/format";
import { Metric } from "../components/ui";

export function Hero({ data }: { data: SiteData }) {
  const { summary } = data;
  const peakRow = useMemo(() => {
    return data.performance.reduce<PerformanceRun | null>((best, run) => {
      if (run.bitsPerSecond === null) return best;
      if (!best || (best.bitsPerSecond ?? 0) < (run.bitsPerSecond ?? 0)) return run;
      return best;
    }, null);
  }, [data.performance]);

  const corpus = summary.corpus;
  const corpusLabel = corpus.datasets.length === 1
    ? corpus.datasets[0].label
    : `${corpus.datasets.length} fault-tree datasets`;
  const gtx = data.devices.find((d) => d.deviceShort === "GTX 1660 S");

  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">{corpusLabel} · mcSCRAM</p>
        <h1>Benchmarks for a GPU Monte-Carlo PRA engine</h1>
        <p className="lede">
          mcSCRAM is a Monte-Carlo successor to the SCRAM probabilistic risk-assessment toolkit that compiles fault-tree models into a propagation DAG (<em>PDAG</em>) and samples them on GPUs via SYCL.
          This page distills <strong>{formatInteger(summary.counts.performanceRuns)}</strong> cross-device runs over <strong>{summary.counts.devices}</strong> devices,{" "}
          <strong>{formatInteger(summary.counts.compilerRows)}</strong> compiler ablation rows, and <strong>{formatInteger(summary.counts.traceFiles)}</strong> sampled convergence traces into one
          interactive archive — scoped to the {corpus.totalModelCount}-model{" "}
          {corpus.datasets.map((d) => d.label).join(" + ")} corpus.
        </p>
        <div className="hero-actions">
          <a href={summary.mcScramRepo}>mcSCRAM</a>
          <a href={summary.sourceRepo}>Archive</a>
          <a href={summary.doi}>Zenodo DOI</a>
        </div>
      </div>
      <div className="hero-card">
        <Metric
          label="Peak throughput"
          value={peakRow ? `${formatNumber(peakRow.bitsPerSecond, 3)} bits/s` : "n/a"}
          detail={peakRow ? `${peakRow.modelName} · ${peakRow.deviceShort}` : undefined}
        />
        <Metric
          label="Median GPU bits/s"
          value={formatNumber(gtx?.bitsPerSecond.median ?? null, 3)}
          detail={gtx ? `across ${formatInteger(gtx.count)} GTX 1660 SUPER runs` : undefined}
        />
        <Metric
          label="GPU vs CPU host"
          value={`${formatNumber(data.speedups.perDevice["GTX 1660 S"]?.median ?? null, 2)}x`}
          detail="matched-configuration median speedup"
        />
        <Metric
          label="Convergence exponent"
          value={formatNumber(data.convergence.exponentSummary.median, 2)}
          detail="median slope of log rel-err vs log trials"
        />
      </div>
    </header>
  );
}
