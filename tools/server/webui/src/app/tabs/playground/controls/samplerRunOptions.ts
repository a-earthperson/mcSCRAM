export type SamplerRunOptions = {
  num_trials: number;
  early_stop: boolean;
  /** Per-node allocator slack for sample_shaper (`settings.overhead_ratio`). */
  overhead_ratio: number;
  confidence?: number;
  delta?: number;
  /** Known true probability µ (mcSCRAM `oracle`); omitted when diagnostics are off. */
  oracle?: number;
};
