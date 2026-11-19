export type NumberSummary = {
  count: number;
  min: number | null;
  p05: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  p95: number | null;
  max: number | null;
  mean: number | null;
};

export type LogFit = {
  slope: number | null;
  intercept: number | null;
  r2: number | null;
  n: number;
};

export type CorpusDatasetDescriptor = {
  name: string;
  label: string;
  description: string;
  directory: string | null;
  modelCount: number;
  modelIds: string[];
};

export type CorpusSummary = {
  datasets: CorpusDatasetDescriptor[];
  totalModelCount: number;
  modelIds: string[];
};

export type Summary = {
  generatedAt: string;
  sourceRepo: string;
  mcScramRepo: string;
  doi: string;
  corpus: CorpusSummary;
  canonicalSources: Record<string, string>;
  counts: Record<string, number>;
  facets: {
    devices: string[];
    deviceShort: string[];
    families: string[];
    compileFlags: string[];
  };
  metrics: Record<string, NumberSummary>;
  flagDocs: Record<string, string>;
  deviceCategory: Record<string, string>;
  deviceShort: Record<string, string>;
};

export type NarrativeCard = {
  title: string;
  metric: string;
  detail: string;
  body: string;
};

export type ModelCard = {
  modelId: string;
  modelName: string;
  family: string;
  reference?: {
    oracleP: number | null;
    rareP: number | null;
    mcubP: number | null;
    mcs: number | null;
  };
  performanceRunCount: number;
  compilerRunCount: number;
  bestBitsPerSecond: number | null;
  bestDevice: string | null;
  bestConvergenceTimeMs: number | null;
  bestRelError: number | null;
  maxNodes: number | null;
  maxLayers: number | null;
  maxPdagGates: number | null;
  modelGates: number | null;
  basicEvents: number | null;
  bestCompressionAllNodes: number | null;
  bestCompileFlag: string | null;
  bestNoKn: boolean | null;
  bestNoXor: boolean | null;
  perDeviceBestBps: Record<string, number>;
  oracleP: number | null;
  magnitude: number | null;
};

export type PerformanceRun = {
  runId: string;
  setId: string;
  modelId: string;
  modelName: string;
  family: string;
  inputFile: string;
  inputUrl: string | null;
  deviceName: string;
  deviceShort: string;
  deviceCategory: string;
  deviceType: string;
  vendor: string;
  driverVersion?: string;
  maxComputeUnits?: number | null;
  maxClockMHz?: number | null;
  globalMemSize?: number | null;
  compileFlag: string;
  noKn: boolean | null;
  noXor: boolean | null;
  ccf: boolean | null;
  confidenceLevel: number | null;
  delta: number | null;
  stopOnConvergence: boolean | null;
  convergencePolicy: string;
  requestedTrials: number | null;
  burninTrials: number | null;
  numNodes: number | null;
  numLayers: number | null;
  numKernels: number | null;
  avgKernelsPerLayer: number | null;
  avgBlocksPerKernel: number | null;
  numVariables: number | null;
  numGates: number | null;
  pdagGates: number | null;
  pdagVariables: number | null;
  pdagMaxFanIn: number | null;
  pdagAvgFanIn: number | null;
  pdagAnd: number | null;
  pdagOr: number | null;
  pdagAtleast: number | null;
  pdagXor: number | null;
  mean: number | null;
  p01: number | null;
  p05: number | null;
  p95: number | null;
  p99: number | null;
  stdErr: number | null;
  eps: number | null;
  targetEps: number | null;
  oracleP: number | null;
  rareP: number | null;
  mcubP: number | null;
  mcs: number | null;
  absError: number | null;
  relError: number | null;
  convergenceTimeMs: number | null;
  bitsPerSecond: number | null;
  iterationsPerSecond: number | null;
  bitsPerIteration: number | null;
  shaperBitsPerIteration: number | null;
  shaperNumIterations: number | null;
  totalIterations: number | null;
  oneBits: number | null;
  totalBits: number | null;
  rawSource: string;
  rawUrl: string;
};

export type DeviceSummary = {
  deviceName: string;
  deviceShort: string;
  deviceCategory: string;
  count: number;
  bitsPerSecond: NumberSummary;
  iterationsPerSecond: NumberSummary;
  convergenceTimeMs: NumberSummary;
  relativeError: NumberSummary;
  bitsPerIteration: NumberSummary;
  throughputVsNodesFit: LogFit;
  timeVsNodesFit: LogFit;
};

export type SpeedupRow = {
  modelId: string;
  modelName: string;
  deviceName: string;
  deviceShort: string;
  compileFlag: string;
  noKn: boolean | null;
  noXor: boolean | null;
  delta: number | null;
  ccf: boolean | null;
  baselineBps: number;
  deviceBps: number;
  speedupVsHost: number;
  numNodes: number | null;
  pdagGates: number | null;
};

export type SpeedupData = {
  baseline: string;
  rows: SpeedupRow[];
  perDevice: Record<string, NumberSummary>;
};

export type CompilerRow = {
  rowId: number;
  modelId: string;
  modelName: string;
  family: string;
  compileFlag: string;
  noKn: boolean | null;
  noXor: boolean | null;
  ccf: boolean | null;
  buildType: string;
  version: string;
  basicEvents: number | null;
  houseEvents: number | null;
  modelGates: number | null;
  modelAnd: number | null;
  modelOr: number | null;
  modelAtleast: number | null;
  modelXor: number | null;
  pdagGates: number | null;
  pdagModules: number | null;
  pdagVariables: number | null;
  pdagAnd: number | null;
  pdagOr: number | null;
  pdagAtleast: number | null;
  pdagXor: number | null;
  pdagMaxFanIn: number | null;
  pdagAvgFanIn: number | null;
  compressionBasicEvents: number | null;
  compressionGates: number | null;
  compressionAllNodes: number | null;
  preprocessingTimeUs: number | null;
};

export type CompilerSummary = {
  compileFlag: string;
  noKn: boolean | null;
  noXor: boolean | null;
  count: number;
  compressionAllNodes: NumberSummary;
  compressionGates: NumberSummary;
  compressionBasicEvents: NumberSummary;
  preprocessingTimeUs: NumberSummary;
  pdagGates: NumberSummary;
  pdagAvgFanIn: NumberSummary;
  pdagMaxFanIn: NumberSummary;
  andShare: NumberSummary;
  orShare: NumberSummary;
  atleastShare: NumberSummary;
  xorShare: NumberSummary;
  andSum: number;
  orSum: number;
  atleastSum: number;
  xorSum: number;
  modelsWithAtleast: number;
  modelsWithXor: number;
  fanInVsCompressionFit: LogFit;
  timeVsGatesFit: LogFit;
};

export type CompileLevelPass = {
  name: string;
  what: string;
};

export type CompileLevelRewrite = {
  from: string;
  to: string;
  note: string;
};

export type CompileLevel = {
  id: string;
  level: number;
  name: string;
  stage: string;
  summary: string;
  passes: CompileLevelPass[];
  mcImpact: string;
  rewrites?: CompileLevelRewrite[];
};

export type CompileLevelStats = {
  compileFlag: string;
  level: number | null;
  name: string | null;
  stage: string | null;
  count: number;
  pdagGates: NumberSummary;
  preprocessingTimeUs: NumberSummary;
  compressionAllNodes: NumberSummary;
  compositionShare: {
    and: number;
    or: number;
    atleast: number;
    xor: number;
  };
  modelsWithAtleast: number;
  modelsWithXor: number;
};

export type ConnectiveSwitch = {
  id: string;
  cliFlag: string;
  expandsWhen: string;
  title: string;
  subtitle: string;
  formula: string;
  bases: string;
  growth: string;
  mcImpact: string;
};

export type ConnectiveLevelEffect = {
  compileFlag: string;
  expandOn: {
    count: number;
    compressionAllNodes: NumberSummary;
    pdagGates: NumberSummary;
  };
  expandOff: {
    count: number;
    compressionAllNodes: NumberSummary;
    pdagGates: NumberSummary;
  };
  pdagRatioOnVsOff: number | null;
};

export type ConnectiveEffectGroup = {
  modelCount: number;
  models: string[];
  rowCount: number;
  levels: ConnectiveLevelEffect[];
};

export type CompilerData = {
  rows: CompilerRow[];
  summaries: CompilerSummary[];
  levelStats: CompileLevelStats[];
  connectiveEffect: {
    kn: ConnectiveEffectGroup;
    xor: ConnectiveEffectGroup;
  };
  taxonomy: CompileLevel[];
  connectiveSwitches: ConnectiveSwitch[];
};

export type TracePoint = {
  iterations?: number | null;
  trials?: number | null;
  projected_trials?: number | null;
  remaining_trials?: number | null;
  epsilon?: number | null;
  target_epsilon?: number | null;
  std_err?: number | null;
  p01?: number | null;
  p05?: number | null;
  mean?: number | null;
  p95?: number | null;
  p99?: number | null;
  info_bits_total?: number | null;
  oracle_p?: number | null;
  abs_error?: number | null;
  rel_error?: number | null;
  n_required?: number | null;
  n_ratio?: number | null;
};

export type TraceSummary = {
  traceId: string;
  setId: string;
  setLabel: string;
  modelId: string;
  modelName: string;
  family: string;
  timestamp: string;
  points: number;
  sampledPoints: number;
  finalTrials: number | null;
  finalIterations: number | null;
  finalMean: number | null;
  oracleP: number | null;
  finalRelError: number | null;
  finalEpsilon: number | null;
  targetEpsilon: number | null;
  infoBitsTotal: number | null;
  convergenceExponent: LogFit;
  trialsAtThreshold: Record<string, number | null>;
  source: string;
  sourceUrl: string;
};

export type TraceSeries = TraceSummary & { pointsData: TracePoint[] };

export type TraceIndexData = {
  index: TraceSummary[];
  maxPointsPerTrace: number;
};

export type TraceModelData = {
  modelId: string;
  series: TraceSeries[];
  maxPointsPerTrace: number;
};

export type ConvergenceView = {
  exponentSummary: NumberSummary;
  infoBitsSummary: NumberSummary;
  trialsToTarget: Record<string, NumberSummary>;
  trialsToTargetSamples: Record<string, number[]>;
  exponentSamples: number[];
  deepTraceModels: Array<{
    modelId: string;
    family: string;
    points: number;
    infoBitsTotal: number | null;
  }>;
};

export type SetIndexEntry = {
  setId: string;
  label: string;
  friendlyName: string;
  deviceTag: string;
  delta: number | null;
  confidence: number | null;
  timeoutSec: number | null;
  requestedTrials: number | null;
};

export type PlotAsset = {
  id: string;
  title: string;
  category: string;
  modelId: string;
  source: string;
  sourceUrl: string;
  assetPath: string;
  extension: string;
  sizeBytes: number;
};

export type LinkedArtifact = {
  title: string;
  kind?: string;
  path: string;
  url: string;
  extension: string;
  sizeBytes?: number;
};

export type Artifacts = {
  plots: PlotAsset[];
  scripts: LinkedArtifact[];
  dataFiles: LinkedArtifact[];
  vectorFigures: LinkedArtifact[];
};

export type Tallies = {
  source: string;
  rowCount: number;
  nonZeroCount: number;
  topEvents: Array<{
    eventId: string;
    mean: number | null;
    p05: number | null;
    p95: number | null;
    stdErr: number | null;
  }>;
  meanSummary: NumberSummary;
  stdErrSummary: NumberSummary;
  logMeanHistogram: Array<{ bucket: number; count: number }>;
};

export type ExactResult = {
  file: string;
  modelId: string;
  method: string;
  probability: number | null;
  calcTime: number | null;
  totalTime: number | null;
  sopProducts: number | null;
  sopProbability: number | null;
};

export type HardwareDevice = {
  deviceShort: string;
  deviceLong: string;
  vendor: string;
  category: string;
  architecture: string;
  process: string;
  year: number;
  cudaCompute: string | null;
  lanes: number;
  lanesLabel: string;
  lanesNote: string;
  hwThreads: number | null;
  subgroupSize: number | null;
  subgroupNote: string;
  baseClockMHz: number;
  boostClockMHz: number;
  computeUnits: number;
  computeUnitsLabel: string;
  smCount: number | null;
  residentWarps: number | null;
  bandwidthGBs: number;
  memoryType: string;
  memoryRateGbps: number;
  memoryBusBits: number;
  l1KiB: number | null;
  l1Note: string;
  l2KiB: number | null;
  l2Note: string;
  l3KiB: number | null;
  l3Note: string | null;
  globalMemMiB: number;
  globalMemLabel: string;
  fp32TFlops: number;
  fp32Note: string;
  sourceUrl: string;
  measuredPeakBps: number | null;
  measuredMedianBps: number | null;
  measuredP95Bps: number | null;
  measuredRuns: number;
  matchedSpeedupMedian: number | null;
  matchedSpeedupP95: number | null;
  matchedSpeedupCount: number | null;
  throughputVsNodesFit: LogFit | null;
  bandwidthUtilization: number | null;
  bpsPerLane: number | null;
  bpsPerSm: number | null;
  bpsPerResidentWarp: number | null;
  bpsPerGBs: number | null;
  flopsPerBit: number | null;
  observed: {
    deviceName?: string;
    vendor?: string;
    driverVersion?: string;
    maxComputeUnits?: number | null;
    maxClockMHz?: number | null;
    globalMemBytes?: number | null;
  };
};

export type HardwareData = {
  devices: HardwareDevice[];
  bandwidthPeakFit: LogFit;
  bandwidthMedianFit: LogFit;
  lanePeakFit: LogFit;
  laneMedianFit: LogFit;
  flopsPeakFit: LogFit;
  flopsMedianFit: LogFit;
  glossary: Array<{ term: string; summary: string; mapping: string }>;
  source: string;
  sourceUrl: string;
  normalizationNotes: string[];
};

export type SiteData = {
  summary: Summary;
  narrative: NarrativeCard[];
  models: ModelCard[];
  performance: PerformanceRun[];
  devices: DeviceSummary[];
  speedups: SpeedupData;
  compiler: CompilerData;
  convergence: ConvergenceView;
  traces: TraceIndexData;
  setIndex: Record<string, SetIndexEntry>;
  artifacts: Artifacts;
  tallies: Tallies;
  exactResults: ExactResult[];
  hardware: HardwareData;
};
