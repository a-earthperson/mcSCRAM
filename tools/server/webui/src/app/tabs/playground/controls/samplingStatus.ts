/** mcSCRAM live-server run state returned by /reset, /start, /stop. */

export type SamplingStatus = {
  status: string;
  pid: number | null;
  alive: boolean | null;
  started_at: number | null;
  finished_at: number | null;
  update_count: number;
  last_update: unknown | null;
  last_result: unknown | null;
  last_error: string | null;
};

export type SamplingControlResponse = {
  accepted: boolean;
  reason?: string;
  status?: SamplingStatus;
};
