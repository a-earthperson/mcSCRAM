/**
 * Pure SYCL execution model — no rendering dependencies.
 *
 * Layer boundary: everything re-exported here reasons about device capabilities,
 * launch geometry (work-items / sub-groups / work-groups / nd-ranges) and the
 * bitpack data layer in exact integer terms. The MathBox rendering adapter
 * (`../mathbox`) consumes these types and is the *only* place that knows about
 * shaders, world coordinates, or three.js.
 */
export * from "./geometry";
export * from "./device";
export * from "./kernel";
export * from "./model";
