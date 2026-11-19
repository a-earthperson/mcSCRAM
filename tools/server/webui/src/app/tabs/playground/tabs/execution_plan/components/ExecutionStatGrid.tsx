import type { CSSProperties, ReactNode } from "react";
import { classNames } from "../../../../../lib/format";

export type ExecutionStat = {
  label: string;
  value: ReactNode;
  /** Optional hover detail / native units. */
  title?: string;
};

/**
 * Compact ruled key/value grid used by the device and sample-shaper summaries.
 * Mirrors the archive `.metric-grid` rhythm (hairline cells, tracked caps
 * labels, tabular mono values) but tuned denser for engineering readouts.
 */
export function ExecutionStatGrid({
  stats,
  columns,
  className,
}: {
  stats: ExecutionStat[];
  columns?: number;
  className?: string;
}) {
  const style = columns
    ? ({ "--exec-stat-columns": String(columns) } as CSSProperties)
    : undefined;
  return (
    <dl className={classNames("exec-stat-grid", className)} style={style}>
      {stats.map((stat) => (
        <div className="exec-stat" key={stat.label} title={stat.title}>
          <dt className="exec-stat__label">{stat.label}</dt>
          <dd className="exec-stat__value">{stat.value}</dd>
        </div>
      ))}
    </dl>
  );
}
