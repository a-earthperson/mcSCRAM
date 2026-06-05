/** mcSCRAM live-server run state returned by /api/v1/reset, /api/v1/start, /api/v1/stop. */

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
