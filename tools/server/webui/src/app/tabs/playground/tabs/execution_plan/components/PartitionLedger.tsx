import type { InkPalette } from "../../../../../components/plot/plotTokens";
import { formatBytes, formatInteger } from "../../../../../lib/format";
import type { ExecutionPartitionDescription } from "../../../../../api/faultTreeApi";
import {
  compactCount,
  formatRange3,
  kindColor,
  kindLabel,
  overprovisionRatio,
  partitionKind,
} from "../scene/executionScene";

export function PartitionLedger({
  partitions,
  selectedId,
  onSelect,
  palette,
}: {
  partitions: ExecutionPartitionDescription[];
  selectedId: number | null;
  onSelect: (partitionId: number | null) => void;
  palette: InkPalette;
}) {
  const ordered = [...partitions].sort((a, b) => {
    if (a.layer !== b.layer) return a.layer - b.layer;
    return a.id - b.id;
  });

  return (
    <div className="exec-ledger">
      <table className="exec-ledger__table">
        <thead>
          <tr>
            <th scope="col">partition</th>
            <th scope="col" className="exec-ledger__num">layer</th>
            <th scope="col" className="exec-ledger__num">nodes</th>
            <th scope="col">global range</th>
            <th scope="col" className="exec-ledger__num">work-items</th>
            <th scope="col" className="exec-ledger__num">over</th>
            <th scope="col" className="exec-ledger__num">resident</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((partition) => {
            const key = partitionKind(partition);
            const selected = partition.id === selectedId;
            const over = overprovisionRatio(partition);
            return (
              <tr
                key={partition.id}
                className={selected ? "exec-ledger__row exec-ledger__row--selected" : "exec-ledger__row"}
                aria-selected={selected}
                onClick={() => onSelect(selected ? null : partition.id)}
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(selected ? null : partition.id);
                  }
                }}
              >
                <td>
                  <span className="exec-ledger__kind">
                    <span
                      className="exec-ledger__dot"
                      style={{ backgroundColor: kindColor(key, palette) }}
                      aria-hidden="true"
                    />
                    {kindLabel(key)}
                  </span>
                </td>
                <td className="exec-ledger__num">{partition.layer}</td>
                <td className="exec-ledger__num">{formatInteger(partition.node_count)}</td>
                <td className="exec-ledger__range">{formatRange3(partition.global_range)}</td>
                <td className="exec-ledger__num">{compactCount(partition.work_item_count)}</td>
                <td className="exec-ledger__num">{formatPercent(over)}</td>
                <td className="exec-ledger__num">{formatBytes(partition.resident_bytes)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatPercent(ratio: number): string {
  if (!Number.isFinite(ratio) || ratio <= 0) return "0%";
  if (ratio < 0.001) return "<0.1%";
  return `${(ratio * 100).toFixed(ratio < 0.1 ? 1 : 0)}%`;
}
