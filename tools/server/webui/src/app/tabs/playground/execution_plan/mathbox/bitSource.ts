/**
 * Pluggable bit provider for the bitpack field.
 *
 * The field's shader chain expands every bitpack tile into a `bitsPerSide^2`
 * grid of points on the GPU (via `repeat`), then asks a *bit source* for the
 * 0/1 trial value at each point. Isolating that one slot behind an interface is
 * what lets the execution layer stay free of GLSL: it hands the renderer a
 * `KernelModel` + numbers, and the bit source turns those into the shader (and,
 * for hydrated data, the texture) that colours the field.
 *
 * Two implementations:
 *
 *  - {@link proceduralBitSource} — a cheap deterministic coordinate hash. This
 *    is the default and, at `seed == 0`, is byte-for-byte the original field
 *    shader, so swapping the field over to this abstraction is a zero-cost,
 *    zero-visual-change refactor. A non-zero seed gives each kernel a stable but
 *    distinct "representative" pattern.
 *
 *  - {@link sampledBitSource} — samples real per-bit values uploaded as a small
 *    data texture (one texel per trial bit of the rendered window). This is the
 *    forward seam for true server-hydrated Monte Carlo output; it is exercised
 *    only when a caller explicitly constructs it.
 */

export type BitSourceKind = "procedural" | "sampled";

/** A small per-bit data buffer to upload alongside a sampled bit source. */
export type BitDataBuffer = {
  id: string;
  /** Texel grid width = window columns * bitsPerSide. */
  width: number;
  /** Texel grid height = window rows * bitsPerSide. */
  height: number;
  /** Row-major (width-fastest) flat array of 0/1 values. */
  data: number[];
};

type BitSourceBase = {
  /**
   * GLSL for the bit-assignment resample stage. It receives the repeated
   * carrier via `getSample(vec3)` (whose `.xy` is the local bit cell) and the
   * global repeated coordinate via `xyz.xy`, and must return
   * `vec4(localCell.xy, bit, alpha)`.
   */
  glsl: string;
  /** Uniforms the field builder forwards onto the `<shader>` node. */
  uniforms?: Record<string, number>;
};

export type ProceduralBitSource = BitSourceBase & {
  kind: "procedural";
};

export type SampledBitSource = BitSourceBase & {
  kind: "sampled";
  /** Extra shader source bound as `sources: [buffer.id]`. */
  buffer: BitDataBuffer;
};

export type BitSource = ProceduralBitSource | SampledBitSource;

/**
 * Deterministic coordinate hash. With `seed == 0` the emitted GLSL + uniform
 * matches the historical field shader exactly (the `+ bitSeed * 89.0` term
 * vanishes), guaranteeing no visual or performance regression for the default
 * scene. A stable non-zero seed gives a kernel its own representative pattern.
 */
export function proceduralBitSource(seed = 0): ProceduralBitSource {
  return {
    kind: "procedural",
    uniforms: { bitSeed: Number.isFinite(seed) ? seed : 0 },
    glsl: `
uniform float bitSeed;

vec4 getSample(vec3 xyz);

float randomBit(vec2 globalCell, vec2 localCell) {
  float state = globalCell.x * 17.0 + globalCell.y * 131.0;
  state += localCell.x * 29.0 + localCell.y * 43.0 + 23.0;
  state += bitSeed * 89.0;
  state = mod(state, 251.0);
  state = mod(state * 73.0 + 41.0, 251.0);
  state = mod(state * state + state * 37.0 + 17.0, 251.0);
  return step(125.0, state);
}

vec4 getFramesSample(vec3 xyz) {
  vec4 c = getSample(xyz);
  vec2 globalCell = floor(xyz.xy + vec2(0.5));
  vec2 localCell = floor(c.xy + vec2(0.5));
  float bit = randomBit(globalCell, localCell);
  return vec4(c.xy, bit, c.a);
}
`,
  };
}

/**
 * Sample real per-bit values from an uploaded buffer. The buffer is addressed in
 * the same absolute integer coordinate space as the repeated field grid, so the
 * texel at `(globalCol, globalRow)` is the trial bit drawn at that point.
 *
 * Experimental: wire only when the server actually hydrates per-bit output for a
 * window. Until then the field uses {@link proceduralBitSource}.
 */
export function sampledBitSource(buffer: BitDataBuffer): SampledBitSource {
  return {
    kind: "sampled",
    buffer,
    glsl: `
vec4 getBitDataSample(vec3 xyz);
vec4 getSample(vec3 xyz);

vec4 getFramesSample(vec3 xyz) {
  vec4 c = getSample(xyz);
  float bit = step(0.5, getBitDataSample(xyz).x);
  return vec4(c.xy, bit, c.a);
}
`,
  };
}
