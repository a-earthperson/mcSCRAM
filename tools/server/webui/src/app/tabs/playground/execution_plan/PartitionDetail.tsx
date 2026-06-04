import type { InkPalette } from "../../../../components/plotTokens";
import { formatBytes, formatInteger } from "../../../../lib/format";
import type { ExecutionPartitionDescription } from "../../../faultTreeApi";
import {
  compactCount,
  formatRange3,
  kindColor,
  kindLabel,
  overprovisionRatio,
  partitionKind,
} from "./mathbox/executionScene";
import { ExecutionStatGrid, type ExecutionStat } from "./ExecutionStatGrid";
import { WorkGroupTiling } from "./WorkGroupTiling";

export function PartitionDetail({
  partition,
  palette,
  onSelect,
}: {
  partition: ExecutionPartitionDescription | null;
  palette: InkPalette;
  onSelect: (partitionId: number | null) => void;
}) {
  if (!partition) {
    return (
      <div className="exec-detail exec-detail--empty">
        <p>Select a kernel partition to inspect its launch geometry, memory footprint, and dependencies.</p>
      </div>
    );
  }

  const key = partitionKind(partition);
  const color = kindColor(key, palette);
  const over = overprovisionRatio(partition);

  const launchStats: ExecutionStat[] = [
    { label: "logical range", value: formatRange3(partition.logical_range) },
    { label: "local range", value: formatRange3(partition.local_range) },
    { label: "global range", value: formatRange3(partition.global_range) },
    { label: "work-groups", value: compactCount(partition.work_group_count) },
    { label: "work-items", value: compactCount(partition.work_item_count), title: `${partition.work_item_count}` },
    { label: "logical work-items", value: compactCount(partition.logical_work_item_count) },
    {
      label: "overprovisioned",
      value: `${compactCount(partition.overprovisioned_work_items)} (${(over * 100).toFixed(over < 0.1 ? 1 : 0)}%)`,
    },
    { label: "resident", value: formatBytes(partition.resident_bytes), title: `${partition.resident_bytes} bytes` },
  ];

  const shapeStats: ExecutionStat[] = [
    { label: "bitpacks / node", value: compactCount(partition.bitpacks_per_node) },
    { label: "bits / node", value: compactCount(partition.bits_per_node) },
    { label: "bytes / node", value: formatBytes(partition.bytes_per_node) },
    { label: "fan-in (min/max)", value: `${formatInteger(partition.min_fan_in)} / ${formatInteger(partition.max_fan_in)}` },
    { label: "total fan-in", value: compactCount(partition.total_fan_in) },
    {
      label: "inputs +/~",
      value: `${formatInteger(partition.positive_input_count)} / ${formatInteger(partition.negated_input_count)}`,
      title: "positive / negated inputs",
    },
  ];

  return (
    <div className="exec-detail">
      <header className="exec-detail__head">
        <span className="exec-detail__dot" style={{ backgroundColor: color }} aria-hidden="true" />
        <div className="exec-detail__heading">
          <h4 className="exec-detail__title">{kindLabel(key)} partition</h4>
          <p className="exec-detail__sub">
            layer {partition.layer} &middot; #{partition.id} &middot; {formatInteger(partition.node_count)} nodes
          </p>
        </div>
      </header>

      <ExecutionStatGrid stats={launchStats} columns={2} className="exec-detail__grid" />

      <WorkGroupTiling partition={partition} colorKey={key} palette={palette} />

      <ExecutionStatGrid stats={shapeStats} columns={2} className="exec-detail__grid" />

      {partition.dependency_partition_ids?.length ? (
        <div className="exec-detail__deps">
          <span className="exec-detail__deps-label">depends on</span>
          <div className="exec-detail__deps-chips">
            {partition.dependency_partition_ids.map((depId) => (
              <button
                key={depId}
                type="button"
                className="exec-detail__dep"
                onClick={() => onSelect(depId)}
              >
                #{depId}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <p className="exec-detail__deps exec-detail__deps--none">no upstream partitions (source layer)</p>
      )}
    </div>
  );
}
