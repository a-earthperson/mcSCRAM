import { formatBytes, formatInteger } from "../../../../../lib/format";
import type { ExecutionSampleShaperDescription } from "../../../../../api/faultTreeApi";
import { compactCount } from "../scene/executionScene";
import { ExecutionStatGrid, type ExecutionStat } from "./ExecutionStatGrid";

const STRATEGY_LABELS: Record<string, string> = {
  auto_memory_budget: "auto / memory budget",
  auto_memory_budget_cacheline_aligned: "auto / memory budget, cache-aligned",
  requested_trials_per_pass: "requested trials/pass",
  requested_trials_per_pass_cacheline_aligned: "requested trials/pass, cache-aligned",
  uninitialized: "uninitialized",
};

export function SampleShaperSummary({
  sample,
}: {
  sample: ExecutionSampleShaperDescription;
}) {
  const shape = sample.sample_shape;
  const strategy = STRATEGY_LABELS[sample.shape_strategy] ?? sample.shape_strategy;

  const stats: ExecutionStat[] = [
    { label: "requested trials", value: compactCount(sample.requested_trials), title: `${sample.requested_trials}` },
    { label: "rounded trials", value: compactCount(sample.rounded_trials), title: `${sample.rounded_trials}` },
    {
      label: "iterations",
      value: sample.unlimited_iterations ? "unbounded" : formatInteger(sample.total_iterations),
    },
    {
      label: "sample shape",
      value: `${formatInteger(shape.batch_size)} x ${formatInteger(shape.bitpacks_per_batch)}`,
      title: `${shape.num_bitpacks} bitpacks/pass`,
    },
    { label: "trials / pass", value: compactCount(sample.bits_per_iteration), title: `${sample.bits_per_iteration} bits` },
    { label: "bits / bitpack", value: formatInteger(sample.bits_per_bitpack) },
    {
      label: "requested trials/pass",
      value: sample.requested_trials_per_pass ? compactCount(sample.requested_trials_per_pass) : "auto",
      title: `${sample.requested_trials_per_pass} (0 = auto)`,
    },
    {
      label: "realized trials/pass",
      value: compactCount(sample.realized_trials_per_pass),
      title: `${sample.realized_trials_per_pass} trials`,
    },
    {
      label: "resident",
      value: formatBytes(sample.resident_memory?.bytes_total),
      title: `${sample.resident_memory?.bytes_total ?? 0} bytes`,
    },
    {
      label: "device budget",
      value: formatBytes(sample.max_device_bytes),
      title: `${sample.max_device_bytes} bytes`,
    },
  ];

  return (
    <section className="exec-card" aria-label="Sample shaping">
      <header className="exec-card__head">
        <p className="exec-card__eyebrow">Sample shaping</p>
        <h3 className="exec-card__title">{strategy}</h3>
        <div className="exec-card__tags">
          <span className="exec-pill exec-pill--shape">{sample.bitpack_type}</span>
          <span className="exec-pill exec-pill--quiet">
            {compactCount(shape.num_bitpacks)} bitpacks/pass
          </span>
        </div>
      </header>
      <ExecutionStatGrid stats={stats} columns={2} />
    </section>
  );
}
