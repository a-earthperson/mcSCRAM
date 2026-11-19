const DELTA_EPSILON = 1.0e-12;
const LN2 = Math.LN2;
const LOG10 = Math.LN10;

export const DEFAULT_WALD_ALPHA = 0.01;
export const DEFAULT_WALD_DELTA = 0.001;

export type ConvergenceStatsPayload = {
  iterations?: number;
  trials?: number;
  projected_iterations?: number;
  projected_trials?: number;
  remaining_iterations?: number;
  remaining_trials?: number;
  epsilon?: number;
  epsilon_log10?: number;
  target_epsilon?: number;
  target_epsilon_log10?: number;
  epsilon_converged?: boolean;
  epsilon_log10_converged?: boolean;
  std_err?: number;
  p01?: number;
  p05?: number;
  mean?: number;
  p95?: number;
  p99?: number;
  info_bits_last?: number;
  info_bits_total?: number;
  oracle_p?: number;
  phase?: string;

  // --- host-side (Python) engine extensions -------------------------------
  // The convergence/statistics loop now runs on the host (see
  // server `fuzzybits_server.convergence`).  In addition to the canonical Wald
  // fields above, every attached policy/metric is streamed under its own
  // namespace.  All optional and additive; existing consumers are unaffected.
  pass_index?: number;
  trials_per_pass?: number;
  converged?: boolean;
  stop_reason?: string;
  stopping_policy?: string;
  primary_policy?: string;

  // Wald policy (namespaced duplicate of the canonical Wald fields).
  wald_epsilon?: number;
  wald_target_epsilon?: number;
  wald_converged?: boolean;
  wald_projected_trials?: number;

  // Bayesian (Jeffreys Beta posterior) policy stream.
  bayes_mean?: number;
  bayes_std_err?: number;
  bayes_epsilon?: number;
  bayes_target_epsilon?: number;
  bayes_converged?: boolean;
  bayes_ci_low?: number;
  bayes_ci_high?: number;
  bayes_alpha?: number;
  bayes_beta?: number;
  bayes_projected_trials?: number;

  // GBAS (Huber 2017) exact relative-error stream.  Stopping is on a fixed
  // success count `gbas_k`; `gbas_epsilon`/`gbas_target_epsilon` are the current
  // and target *relative* standard deviations (1/sqrt(successes-2) vs.
  // 1/sqrt(k-2)), so converged iff successes >= k.  The point estimate
  // (`gbas_mean = (k-1)/R`) and exact CI are populated only once converged.
  gbas_k?: number;
  gbas_successes?: number;
  gbas_target_delta?: number;
  gbas_achieved_delta?: number;
  gbas_relative_tolerance?: number;
  gbas_rel_sd?: number;
  gbas_rel_sd_target?: number;
  gbas_epsilon?: number;
  gbas_target_epsilon?: number;
  gbas_converged?: boolean;
  gbas_projected_trials?: number;
  // Running exact relative-error bands (every pass, once successes >= 3): the
  // running mean scaled by the Gamma(S, S-1) relative interval at the current
  // success count S. `run_ci_*` is the client-requested level; `run_ci95_*` and
  // `run_ci99_*` are fixed 95%/99% bands (parity with the p05-p95 / p01-p99
  // display). All shrink toward the exact intervals at the stopping time.
  gbas_run_ci_low?: number;
  gbas_run_ci_high?: number;
  gbas_run_ci95_low?: number;
  gbas_run_ci95_high?: number;
  gbas_run_ci99_low?: number;
  gbas_run_ci99_high?: number;
  // Exact estimate + intervals, populated only once converged (successes >= k).
  // `ci_*` is the client level; `ci95_*` / `ci99_*` are the fixed 95%/99% pivots.
  gbas_mean?: number;
  gbas_std_err?: number;
  gbas_ci_low?: number;
  gbas_ci_high?: number;
  gbas_ci95_low?: number;
  gbas_ci95_high?: number;
  gbas_ci99_low?: number;
  gbas_ci99_high?: number;
  gbas_rel_low?: number;
  gbas_rel_high?: number;
  gbas_n_star?: number;
  gbas_gamma_r?: number;

  // Optional per-node summaries when multiple nodes are tracked.
  nodes?: Record<string, { mean?: number; trials?: number }>;
};

export type WaldConvergenceParams = {
  alpha: number;
  delta: number;
};

type RawConvergenceInput = Pick<ConvergenceStatsPayload, "iterations" | "trials" | "mean" | "oracle_p" | "phase"> | null | undefined;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function finitePositive(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && Number.isFinite(value) && value > 0;
}

function ceilFinite(value: number): number | undefined {
  if (!Number.isFinite(value) || value < 0) return undefined;
  return Math.ceil(value);
}

function trialsPerIteration(update: RawConvergenceInput): number | undefined {
  if (!finitePositive(update?.trials) || !finitePositive(update?.iterations)) return undefined;
  return update.trials / update.iterations;
}

function iterationsFromTrials(trials: number | undefined, trialsPerIter: number | undefined): number | undefined {
  if (!finitePositive(trials) || !finitePositive(trialsPerIter)) return undefined;
  return Math.ceil(trials / trialsPerIter);
}

/** GBAS projected trial budget expressed as pass count (ceil trials / trials-per-pass). */
export function gbasProjectedPasses(
  update: Pick<ConvergenceStatsPayload, "gbas_projected_trials" | "trials_per_pass" | "trials" | "iterations">,
): number | undefined {
  const trialsPerPass = finitePositive(update.trials_per_pass)
    ? update.trials_per_pass
    : trialsPerIteration(update);
  return iterationsFromTrials(update.gbas_projected_trials, trialsPerPass);
}

export function normalQuantileTwoSided(confidence: number): number {
  const p = clamp(confidence, DELTA_EPSILON, 1.0 - DELTA_EPSILON);
  const alpha = 1.0 - p;
  const q = 1.0 - alpha / 2.0;

  const a = [
    -3.969683028665376e+1,
    2.209460984245205e+2,
    -2.759285104469687e+2,
    1.38357751867269e+2,
    -3.066479806614716e+1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e+1,
    1.615858368580409e+2,
    -1.556989798598866e+2,
    6.680131188771972e+1,
    -1.328068155288572e+1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];

  const plow = 0.02425;
  const phigh = 1 - plow;

  if (q < plow) {
    const u = Math.sqrt(-2.0 * Math.log(q));
    return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) /
      ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0);
  }

  if (q > phigh) {
    const u = Math.sqrt(-2.0 * Math.log(1.0 - q));
    return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) /
      ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0);
  }

  const u = q - 0.5;
  const t = u * u;
  return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * u /
    (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1.0);
}

export function confidenceFromAlpha(alpha: number): number {
  const clampedAlpha = Number.isFinite(alpha) ? clamp(alpha, DELTA_EPSILON, 1.0 - DELTA_EPSILON) : DEFAULT_WALD_ALPHA;
  return 1.0 - clampedAlpha;
}

function requiredTrials(p: number, epsilon: number, z: number): number | undefined {
  if (!finitePositive(epsilon)) return undefined;
  return ceilFinite((z * z * p * (1.0 - p)) / (epsilon * epsilon));
}

function requiredTrialsLog10(p: number, epsilonLog10: number, z: number): number | undefined {
  if (!finitePositive(p) || !finitePositive(epsilonLog10)) return undefined;
  return ceilFinite((z * z * (1.0 - p)) / (p * epsilonLog10 * epsilonLog10 * LOG10 * LOG10));
}

function logGamma(z: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];

  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }

  let x = 0.9999999999998099;
  const shifted = z - 1;
  for (let i = 0; i < coefficients.length; i += 1) {
    x += coefficients[i] / (shifted + i + 1);
  }
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}

function digamma(x: number): number {
  let value = x;
  let result = 0;
  while (value < 8) {
    result -= 1 / value;
    value += 1;
  }

  const inv = 1 / value;
  const inv2 = inv * inv;
  return result + Math.log(value) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 / 240)));
}

function betaEntropyBits(alpha: number, beta: number): number {
  const lnBeta = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  const nats = lnBeta -
    (alpha - 1.0) * digamma(alpha) -
    (beta - 1.0) * digamma(beta) +
    (alpha + beta - 2.0) * digamma(alpha + beta);
  return nats / LN2;
}

function informationGain(update: RawConvergenceInput, previous: RawConvergenceInput): Pick<ConvergenceStatsPayload, "info_bits_last" | "info_bits_total"> {
  if (!finitePositive(update?.trials) || update.mean === undefined || !Number.isFinite(update.mean)) return {};

  const p = clamp(update.mean, 0, 1);
  const successes = Math.round(p * update.trials);
  const failures = Math.max(0, Math.round(update.trials - successes));
  const currentEntropy = betaEntropyBits(0.5 + successes, 0.5 + failures);
  const priorEntropy = betaEntropyBits(0.5, 0.5);
  const infoBitsTotal = priorEntropy - currentEntropy;

  if (!finitePositive(previous?.trials) || previous.mean === undefined || previous.trials > update.trials) {
    return { info_bits_total: Number.isFinite(infoBitsTotal) ? infoBitsTotal : undefined };
  }

  const prevP = clamp(previous.mean, 0, 1);
  const prevSuccesses = Math.round(prevP * previous.trials);
  const prevFailures = Math.max(0, Math.round(previous.trials - prevSuccesses));
  const previousEntropy = betaEntropyBits(0.5 + prevSuccesses, 0.5 + prevFailures);
  const infoBitsLast = previousEntropy - currentEntropy;

  return {
    info_bits_last: Number.isFinite(infoBitsLast) ? infoBitsLast : undefined,
    info_bits_total: Number.isFinite(infoBitsTotal) ? infoBitsTotal : undefined,
  };
}

export function computeWaldConvergenceStats(
  update: RawConvergenceInput,
  params: WaldConvergenceParams,
  previous?: RawConvergenceInput,
): ConvergenceStatsPayload | null {
  if (!finitePositive(update?.trials) || update.mean === undefined || !Number.isFinite(update.mean)) return null;

  const confidence = confidenceFromAlpha(params.alpha);
  const delta = params.delta > 0 && Number.isFinite(params.delta) ? params.delta : DEFAULT_WALD_DELTA;
  const trials = update.trials;
  const mean = clamp(update.mean, 0, 1);
  const pForTargets = Math.max(mean, DELTA_EPSILON);
  const z = normalQuantileTwoSided(confidence);
  const stdErr = Math.sqrt((mean * (1.0 - mean)) / trials);
  const epsilon = z * stdErr;
  const epsilonLog10 = z * stdErr / (pForTargets * LOG10);
  const targetEpsilon = delta * pForTargets;
  const targetEpsilonLog10 = delta;

  const projectedEpsilonTrials = requiredTrials(mean, targetEpsilon, z) ?? 0;
  const projectedLogTrials = requiredTrialsLog10(pForTargets, targetEpsilonLog10, z) ?? 0;
  const projectedTrials = Math.max(projectedEpsilonTrials, projectedLogTrials);
  const remainingTrials = Math.max(0, projectedTrials - trials);
  const tpi = trialsPerIteration(update);

  const z95 = 1.959963984540054;
  const z99 = 2.5758293035489004;
  const halfWidth95 = z95 * stdErr;
  const halfWidth99 = z99 * stdErr;

  return {
    iterations: update.iterations,
    trials,
    projected_iterations: iterationsFromTrials(projectedTrials, tpi),
    projected_trials: projectedTrials,
    remaining_iterations: iterationsFromTrials(remainingTrials, tpi),
    remaining_trials: remainingTrials,
    epsilon,
    epsilon_log10: epsilonLog10,
    target_epsilon: targetEpsilon,
    target_epsilon_log10: targetEpsilonLog10,
    epsilon_converged: epsilon > 0 && epsilon <= targetEpsilon,
    epsilon_log10_converged: epsilonLog10 > 0 && epsilonLog10 <= targetEpsilonLog10,
    std_err: stdErr,
    p01: clamp(mean - halfWidth99, 0, 1),
    p05: clamp(mean - halfWidth95, 0, 1),
    mean,
    p95: clamp(mean + halfWidth95, 0, 1),
    p99: clamp(mean + halfWidth99, 0, 1),
    ...informationGain(update, previous),
    oracle_p: update.oracle_p,
    phase: update.phase,
  };
}
