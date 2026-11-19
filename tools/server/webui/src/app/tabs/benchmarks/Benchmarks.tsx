import { useEffect, useMemo, useState } from "react";
import { EmptyState, Metric, SectionTitle, Select } from "../../components";
import { formatNumber } from "../../lib/format";
import { CompareTable } from "./CompareTable";
import { HistoryChart } from "./HistoryChart";
import {
  TARGET_LABEL,
  compareReleaseDevelop,
  flagSummary,
  isGpuTarget,
  latestByChannel,
  modelIds,
  pointsForTarget,
  seriesForModel,
  targetOptions,
} from "./compare";
import { BenchDataError, DEFAULT_BENCH_INDEX_URL, loadBenchIndex } from "./load";
import type { BenchIndex, BenchTarget } from "./types";
import "./benchmarks.css";

export function Benchmarks() {
  const [index, setIndex] = useState<BenchIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<string | null>(null);
  const [target, setTarget] = useState<string>("v100s");
  const [modelId, setModelId] = useState<string>("");

  useEffect(() => {
    const ac = new AbortController();
    loadBenchIndex(ac.signal)
      .then((payload) => {
        setIndex(payload);
        const targets = targetOptions(payload);
        setTarget((current) =>
          targets.includes(current as BenchTarget) ? current : (targets[0] ?? current),
        );
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setError(err instanceof BenchDataError ? err.message : "Benchmark history is offline.");
        setDetail(err instanceof BenchDataError ? err.urls.join(" · ") : String(err));
      });
    return () => ac.abort();
  }, []);

  const points = useMemo(() => (index ? pointsForTarget(index, target) : []), [index, target]);
  const models = useMemo(() => modelIds(points), [points]);
  const selectedModel = models.includes(modelId) ? modelId : (models[0] ?? "");
  const latest = useMemo(() => latestByChannel(points), [points]);
  const rows = useMemo(
    () => compareReleaseDevelop(latest.release, latest.develop),
    [latest.release, latest.develop],
  );
  const gpu = isGpuTarget(target);

  useEffect(() => {
    if (selectedModel && selectedModel !== modelId) setModelId(selectedModel);
  }, [selectedModel, modelId]);

  if (error) {
    return (
      <section className="section-grid">
        <div className="panel span-3 bench-offline" role="status">
          <SectionTitle
            eyebrow="History"
            title="Benchmark history is offline"
            text="The playground and the rest of the UI still work. History is published as static JSON on GitHub Pages; this tab fetches it from the browser or from a cached copy on fuzzybits-server."
          />
          <p>{error}</p>
          {detail ? <p className="bench-offline__detail">{detail}</p> : null}
          <p>
            Default URL: <code>{DEFAULT_BENCH_INDEX_URL}</code>
          </p>
        </div>
      </section>
    );
  }

  if (!index) {
    return (
      <section className="section-grid">
        <div className="panel span-3">
          <EmptyState text="Loading published history…" />
        </div>
      </section>
    );
  }

  if (!index.points.length) {
    return (
      <section className="section-grid">
        <div className="panel span-3">
          <SectionTitle
            eyebrow="History"
            title="No published runs yet"
            text="The bench-history workflow writes one JSON per version × target × dataset to the bench-results branch. Dispatch it for v0.17.0, v0.17.1, and develop to seed this tab."
          />
        </div>
      </section>
    );
  }

  const releaseFlags = flagSummary(latest.develop?.flags);
  const digestShort = (digest: string | null | undefined) =>
    digest ? digest.replace(/^sha256:/, "").slice(0, 12) : "—";

  return (
    <section className="section-grid bench">
      <div className="panel span-3">
        <SectionTitle
          eyebrow={`Dataset ${index.dataset}`}
          title="Latest release vs develop"
          text="Ratios are develop / release on the same target and recipe. GPU colouring matches the PR gate: block when the point estimate exceeds 3%, warn when it is slower but under 3%. CPU is report-only (warn above 5%, never block; gateway noise is about ±3–7%, #124). Accuracy |z| > 4 is a block. A starred point changed image digest, driver, recipe, or runner."
        />
        <div className="bench-metrics">
          <Metric
            label="Points"
            value={String(index.n_points)}
            detail={`updated ${index.generated_utc}`}
          />
          <Metric
            label="Release"
            value={latest.release?.version ?? "—"}
            detail={latest.release?.created_utc ?? "no tagged run on this target"}
          />
          <Metric
            label="Develop"
            value={latest.develop ? "develop" : "—"}
            detail={latest.develop?.created_utc ?? "no nightly on this target"}
          />
          <Metric
            label="Image"
            value={digestShort(latest.develop?.image_digest ?? latest.release?.image_digest)}
            detail={releaseFlags.length ? `changed ${releaseFlags.join(", ")}` : "same family"}
          />
        </div>
      </div>

      <div className="panel span-3 bench-controls">
        <Select
          label="Target"
          value={target}
          onChange={setTarget}
          options={targetOptions(index).length ? targetOptions(index) : [target]}
        />
        <Select
          label="Model"
          value={selectedModel}
          onChange={setModelId}
          options={models.length ? models : ["—"]}
        />
        <p className="bench-provenance">
          {TARGET_LABEL[target] ?? target}
          {latest.release?.runner ? ` · ${latest.release.runner}` : ""}
          {latest.release?.driver ? ` · driver ${latest.release.driver}` : ""}
          {latest.release?.recipe ? ` · ${latest.release.recipe}` : ""}
        </p>
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow={TARGET_LABEL[target] ?? target}
          title="Per-model ratios"
        />
        <CompareTable
          rows={rows}
          releaseLabel={latest.release?.version ?? "release"}
          developLabel="develop"
        />
      </div>

      <div className="panel span-3">
        <SectionTitle
          eyebrow={selectedModel || "model"}
          title="Over versions"
          text={gpu
            ? "GPU total is all compute kernels (memcpy excluded). GPU kernel is basic_event."
            : "CPU sample is in-process fuzzybits.run after AdaptiveCpp init and a 2^20 JIT warmup."}
        />
        <div className="bench-charts">
          {gpu ? (
            <>
              <HistoryChart
                title="GPU total"
                unit="ms"
                points={seriesForModel(points, selectedModel, "gpu_total_ms")}
              />
              <HistoryChart
                title="GPU kernel (basic_event)"
                unit="ms"
                points={seriesForModel(points, selectedModel, "gpu_kernel_ms")}
              />
            </>
          ) : (
            <HistoryChart
              title="CPU sample"
              unit="s"
              points={seriesForModel(points, selectedModel, "cpu_sample_s")}
            />
          )}
          <HistoryChart
            title="Accuracy z"
            unit=""
            points={seriesForModel(points, selectedModel, "accuracy_z")}
          />
        </div>
      </div>

      <div className="panel span-3">
        <SectionTitle eyebrow="Provenance" title="Every point" />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Version</th>
                <th>When</th>
                <th>Runner</th>
                <th>Driver</th>
                <th>Digest</th>
                <th>Flags</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {points.map((point) => (
                <tr key={`${point.version}-${point.created_utc}-${point.path}`}>
                  <td>{point.version}</td>
                  <td>{point.created_utc ?? "—"}</td>
                  <td>{point.runner ?? "—"}</td>
                  <td>{point.driver ?? "—"}</td>
                  <td><code>{digestShort(point.image_digest)}</code></td>
                  <td>{flagSummary(point.flags).join(", ") || "—"}</td>
                  <td>{point.verdict ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="bench-footnote">
          Absolute numbers compare only on the same runner and recipe.
          CPU history is indicative. Index: <code>{index.source_url ?? DEFAULT_BENCH_INDEX_URL}</code>
          {" · "}
          {formatNumber(index.n_points, 2)} published files.
        </p>
      </div>
    </section>
  );
}
