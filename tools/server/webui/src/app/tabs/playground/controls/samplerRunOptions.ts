export type SamplerRunOptions = {
  model_id?: string;
  compilation_level?: number;
  expand_atleast_gates?: boolean;
  expand_xor_gates?: boolean;
  recognize_atleast_groups?: boolean;
  recognize_xor_groups?: boolean;
  ccf?: boolean;
  keep_null_gates?: boolean;
  /** Total trial budget (0 = auto: unbounded / backend default). */
  num_trials: number;
  early_stop: boolean;
  /** Optimistic per-node trials-per-pass target (`settings.requested_trials_per_pass`; 0 = auto). */
  requested_trials_per_pass: number;
  confidence?: number;
  delta?: number;
  /** Host-side convergence/stopping policy ("wald" | "gbas"). */
  policy?: "wald" | "gbas";
  /** Known true probability µ (mcSCRAM `oracle`); omitted when diagnostics are off. */
  oracle?: number;
};
