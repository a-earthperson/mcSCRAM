import { classNames, formatNumber } from "../../lib/format";
import type { CompareRow } from "./types";

type CompareTableProps = {
  rows: CompareRow[];
  releaseLabel: string;
  developLabel: string;
};

function fmt(value: number | null): string {
  if (value === null) return "—";
  return formatNumber(value, 4);
}

function fmtRatio(value: number | null): string {
  if (value === null) return "—";
  return `${formatNumber(value, 4)}×`;
}

export function CompareTable({ rows, releaseLabel, developLabel }: CompareTableProps) {
  if (!rows.length) {
    return <p className="bench-empty">Need both a tagged release and a develop point on this target.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Metric</th>
            <th>{releaseLabel}</th>
            <th>{developLabel}</th>
            <th>Ratio</th>
            <th>Gate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.modelId}-${row.metric}`}>
              <td>{row.modelId}</td>
              <td><code>{row.metric}</code></td>
              <td>{fmt(row.release)}</td>
              <td>{fmt(row.develop)}</td>
              <td className={classNames("bench-cell", `bench-cell--${row.tone}`)}>{fmtRatio(row.ratio)}</td>
              <td className={classNames("bench-cell", `bench-cell--${row.tone}`)}>
                {row.tone}
                {row.note ? <span className="bench-note">{row.note}</span> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
