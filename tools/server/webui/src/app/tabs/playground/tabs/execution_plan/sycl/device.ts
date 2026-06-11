/**
 * Distilled device model for the SYCL execution layer.
 *
 * The server hands us a verbose `ExecutionDeviceDescription`; the renderer only
 * needs the handful of capabilities that drive launch geometry and the
 * hardware-overlay vocabulary (sub-group size, work-group budget, cache line,
 * compute units). This keeps the pure layer decoupled from the wire format and
 * gives a single place to encode the backend-specific naming correction noted
 * in `sycl_range_mapping.md` (NVIDIA sub-group == warp, work-group == CTA, etc.).
 */
import type { ExecutionDeviceDescription } from "../../../../../api/faultTreeApi";
import { type Dim3, fromRange3, positiveInt } from "./geometry";

/** Human-facing names for SYCL units under a particular backend. */
export type VendorMapping = {
  workItem: string;
  subGroup: string;
  workGroup: string;
  computeUnit: string;
};

export type DeviceModel = {
  name: string;
  vendor: string;
  version: string;
  /** Lockstep SIMT width. NVIDIA reports `[32]` (one warp). */
  subGroupSize: number;
  maxNumSubGroups: number;
  maxWorkGroupSize: number;
  maxWorkItemSizes3d: Dim3;
  /** Global memory cache line in bytes (coalescing target). */
  cacheLineBytes: number;
  /** Physical executors that consume work-groups (NVIDIA SMs). */
  computeUnits: number;
  localMemBytes: number;
  mapping: VendorMapping;
};

const SYCL_MAPPING: VendorMapping = {
  workItem: "work-item",
  subGroup: "sub-group",
  workGroup: "work-group",
  computeUnit: "compute unit",
};

const NVIDIA_MAPPING: VendorMapping = {
  workItem: "thread",
  subGroup: "warp",
  workGroup: "CTA",
  computeUnit: "SM",
};

const AMD_MAPPING: VendorMapping = {
  workItem: "work-item",
  subGroup: "wavefront",
  workGroup: "workgroup",
  computeUnit: "CU",
};

/** Resolve the hardware vocabulary for a device's vendor string. */
export function vendorMapping(vendor: string | undefined): VendorMapping {
  const v = (vendor ?? "").toLowerCase();
  if (v.includes("nvidia")) return NVIDIA_MAPPING;
  if (v.includes("amd") || v.includes("advanced micro")) return AMD_MAPPING;
  return SYCL_MAPPING;
}

function firstSubGroupSize(sizes: number[] | undefined): number {
  if (Array.isArray(sizes) && sizes.length > 0 && Number.isFinite(sizes[0]) && sizes[0] > 0) {
    return Math.floor(sizes[0]);
  }
  return 32;
}

export function deviceFromDescription(desc: ExecutionDeviceDescription): DeviceModel {
  return {
    name: desc.name ?? "device",
    vendor: desc.vendor ?? "",
    version: desc.version ?? "",
    subGroupSize: firstSubGroupSize(desc.sub_group_sizes),
    maxNumSubGroups: positiveInt(desc.max_num_sub_groups, 32),
    maxWorkGroupSize: positiveInt(desc.max_work_group_size, 1024),
    maxWorkItemSizes3d: fromRange3(desc.max_work_item_sizes_3d, { x: 64, y: 1024, z: 1024 }),
    cacheLineBytes: positiveInt(desc.global_mem_cache_line_size, 128),
    computeUnits: positiveInt(desc.max_compute_units, 1),
    localMemBytes: positiveInt(desc.local_mem_size, 49152),
    mapping: vendorMapping(desc.vendor),
  };
}

/**
 * The `nus9601` / RTX 5060 Ti (`sm_120`) device from `sycl_range_mapping.md`.
 * Used when no execution plan has been hydrated yet so the scene still renders
 * a representative, geometrically-correct example.
 */
export const DEFAULT_DEVICE: DeviceModel = {
  name: "NVIDIA GeForce RTX 5060 Ti",
  vendor: "NVIDIA",
  version: "sm_120",
  subGroupSize: 32,
  maxNumSubGroups: 32,
  maxWorkGroupSize: 1024,
  maxWorkItemSizes3d: { x: 64, y: 1024, z: 1024 },
  cacheLineBytes: 128,
  computeUnits: 36,
  localMemBytes: 49152,
  mapping: NVIDIA_MAPPING,
};
