export type SamplerRunOptions = {
  model_id?: string;
  compilation_level?: number;
  expand_atleast_gates?: boolean;
  expand_xor_gates?: boolean;
  recognize_atleast_groups?: boolean;
  recognize_xor_groups?: boolean;
  ccf?: boolean;
  keep_null_gates?: boolean;
  num_trials: number;
  early_stop: boolean;
  /** Per-node allocator slack for sample_shaper (`settings.overhead_ratio`). */
  overhead_ratio: number;
  confidence?: number;
  delta?: number;
  /** Known true probability µ (mcSCRAM `oracle`); omitted when diagnostics are off. */
  oracle?: number;
};
