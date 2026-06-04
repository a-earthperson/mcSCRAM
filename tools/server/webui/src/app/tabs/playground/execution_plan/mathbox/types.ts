/** Three coordinates per point, two points per vector segment. */
export type Point3 = [number, number, number];

/** Closed-open-ish numeric interval as consumed by MathBox cartesian `range`. */
export type MathBoxSpan = [min: number, max: number];

/** MathBox's 3D cartesian range shape: [x, y, z], each as [min, max]. */
export type MathBoxRange3 = [x: MathBoxSpan, y: MathBoxSpan, z: MathBoxSpan];

/** MathBox cartesian view node (untyped meta-API surface). */
export type MathBoxView = any;

export type BitpackBufferShape = {
  bits_per_bitpack: number; // 64.
  batch_size: number;
  bitpacks_per_batch: number;
};

export type BasicEventBitpackBufferShape = BitpackBufferShape & {
  num_events: number;
};
