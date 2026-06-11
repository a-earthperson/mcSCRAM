/**
 * Pure 3D integer geometry + SYCL nd-range linearization.
 *
 * This module is the foundation of the SYCL execution model. It has **no**
 * rendering dependencies (no three.js, no MathBox) so the device/kernel
 * reasoning above it stays trivially testable and portable. Everything here is
 * exact integer arithmetic over launch grids and the bitpack data layer; the
 * mapping to world/screen space lives entirely in the rendering adapter.
 */

/** A 3D launch coordinate or extent. SYCL dimension order is x, y, z. */
export type Dim3 = { x: number; y: number; z: number };

/** Server-side `Range3Description` shape (x/y/z with an optional product). */
export type Range3Like = { x: number; y: number; z: number; product?: number };

/** A half-open 1D interval `[start, start + count)` measured in tiles. */
export type Span = { start: number; count: number };

/**
 * A rectangular crop of the bitpack data layer in *tile* units. Rows index
 * events/nodes (data-layer Y); columns index samples/bitpacks (data-layer X).
 */
export type TileWindow = { row: Span; col: Span };

export function dim3(x: number, y: number, z: number): Dim3 {
  return { x, y, z };
}

/** Total work-items / cells in a 3D extent. */
export function volumeOf(d: Dim3): number {
  return d.x * d.y * d.z;
}

/** Project a server `Range3Description` into a clean `Dim3` (drops `product`). */
export function fromRange3(range: Range3Like | undefined, fallback: Dim3 = { x: 1, y: 1, z: 1 }): Dim3 {
  if (!range) return { ...fallback };
  return {
    x: positiveInt(range.x, fallback.x),
    y: positiveInt(range.y, fallback.y),
    z: positiveInt(range.z, fallback.z),
  };
}

export function positiveInt(value: number, fallback = 1): number {
  return Number.isFinite(value) && value >= 1 ? Math.floor(value) : fallback;
}

/** Ceiling division for non-negative integers; `ceilDiv(n, 0)` is treated as `n`. */
export function ceilDiv(numerator: number, denominator: number): number {
  if (denominator <= 0) return numerator > 0 ? 1 : 0;
  return Math.ceil(numerator / denominator);
}

/** Round `value` up to the nearest multiple of `multiple` (SYCL global-range padding). */
export function roundUpToMultiple(value: number, multiple: number): number {
  if (multiple <= 0) return value;
  return ceilDiv(value, multiple) * multiple;
}

/** Per-axis work-group grid: `ceilDiv(global, local)` in each dimension. */
export function workGroupGrid(global: Dim3, local: Dim3): Dim3 {
  return {
    x: ceilDiv(global.x, local.x),
    y: ceilDiv(global.y, local.y),
    z: ceilDiv(global.z, local.z),
  };
}

/**
 * SYCL row-major local linear id. Dimension 0 (x) is slowest-changing and
 * dimension 2 (z) is fastest-changing, matching `sycl::nd_item::get_local_linear_id`
 * and the warp linearization described in `sycl_range_mapping.md`.
 */
export function localLinearId(local: Dim3, lid: Dim3): number {
  return (lid.x * local.y + lid.y) * local.z + lid.z;
}

/** Inverse of {@link localLinearId}: recover the 3D local id from a linear index. */
export function delinearizeLocal(local: Dim3, linear: number): Dim3 {
  const z = linear % local.z;
  const yx = (linear - z) / local.z;
  const y = yx % local.y;
  const x = (yx - y) / local.y;
  return { x, y, z };
}

export function span(start: number, count: number): Span {
  return { start: Math.max(0, Math.floor(start)), count: Math.max(0, Math.floor(count)) };
}

/** Exclusive end of a span. */
export function spanEnd(s: Span): number {
  return s.start + s.count;
}

/**
 * Intersect a span with the half-open data range `[0, limit)`. Returns the
 * in-bounds portion; the caller can compare counts to detect padding.
 */
export function clampSpanToLimit(s: Span, limit: number): Span {
  const start = Math.min(Math.max(0, s.start), limit);
  const end = Math.min(spanEnd(s), limit);
  return { start, count: Math.max(0, end - start) };
}

export function tileWindow(row: Span, col: Span): TileWindow {
  return { row, col };
}

/** Number of tiles covered by a window. */
export function windowTileCount(w: TileWindow): number {
  return w.row.count * w.col.count;
}
