import { formatBytes, formatInteger } from "../../../../lib/format";
import type { ExecutionDeviceDescription } from "../../../../api/faultTreeApi";
import { ExecutionStatGrid, type ExecutionStat } from "./ExecutionStatGrid";

export function DeviceSummary({
  device,
  bitpackType,
}: {
  device: ExecutionDeviceDescription;
  bitpackType: string;
}) {
  const subGroups = device.sub_group_sizes?.length
    ? device.sub_group_sizes.join(", ")
    : "n/a";
  const workItems3d = `${formatInteger(device.max_work_item_sizes_3d?.x)} x ${formatInteger(
    device.max_work_item_sizes_3d?.y,
  )} x ${formatInteger(device.max_work_item_sizes_3d?.z)}`;

  const stats: ExecutionStat[] = [
    { label: "compute units", value: formatInteger(device.max_compute_units) },
    { label: "clock", value: `${formatInteger(device.max_clock_frequency)} MHz` },
    { label: "max work-group", value: formatInteger(device.max_work_group_size) },
    { label: "sub-group sizes", value: subGroups },
    { label: "work-item 3D", value: workItems3d, title: "max_work_item_sizes<3>" },
    { label: "max alloc", value: formatBytes(device.max_mem_alloc_size), title: `${device.max_mem_alloc_size} bytes` },
    { label: "global memory", value: formatBytes(device.global_mem_size), title: `${device.global_mem_size} bytes` },
    { label: "cache line", value: formatBytes(device.global_mem_cache_line_size) },
    { label: "local memory", value: formatBytes(device.local_mem_size), title: `${device.local_mem_size} bytes` },
    { label: "bitpack", value: bitpackType },
  ];

  return (
    <section className="exec-card" aria-label="Execution device">
      <header className="exec-card__head">
        <p className="exec-card__eyebrow">Device</p>
        <h3 className="exec-card__title" title={device.name}>{device.name || "Unknown device"}</h3>
        <div className="exec-card__tags">
          <span className="exec-pill exec-pill--device">{device.type || "device"}</span>
          {device.vendor ? <span className="exec-pill">{device.vendor}</span> : null}
          {device.driver_version ? (
            <span className="exec-pill exec-pill--quiet">driver {device.driver_version}</span>
          ) : null}
        </div>
      </header>
      <ExecutionStatGrid stats={stats} columns={2} />
    </section>
  );
}
