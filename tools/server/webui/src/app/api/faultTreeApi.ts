import type { DagPayload, FaultTreePayload } from "../components/canvas";
import { mcscramHttpUrl } from "./mcscramClient";

async function fetchFaultTreeFromServer(path: string, searchParams?: Record<string, string>): Promise<FaultTreePayload> {
  const response = await fetch(mcscramHttpUrl(path, searchParams));
  if (!response.ok) {
    throw new Error(`Fault-tree fetch failed: HTTP ${response.status}`);
  }
  return (await response.json()) as FaultTreePayload;
}

async function fetchDagFromServer(path: string, searchParams?: Record<string, string>): Promise<DagPayload> {
  const response = await fetch(mcscramHttpUrl(path, searchParams));
  if (!response.ok) {
    throw new Error(`DAG fetch failed: HTTP ${response.status}`);
  }
  return (await response.json()) as DagPayload;
}

/** Active model configured on the mcSCRAM server (`MCSCRAM_INPUT_FILES`). */
export function fetchActiveFaultTreeModel(): Promise<FaultTreePayload> {
  return fetchFaultTreeFromServer("/api/v1/model/fault-tree");
}

export type XmlInputDescriptor = {
  modelId: string;
  relativePath: string;
  root: string;
  /** `openpsa-xml`, `pcfb`, or `nombrilo`; `.pcfb` and `.safetensors` ids keep their suffix. */
  format?: "openpsa-xml" | "pcfb" | "nombrilo";
  oracle_p?: number;
};

export type XmlInputCatalog = {
  roots: string[];
  models: XmlInputDescriptor[];
  modelIds: string[];
};

export type CompileDagOptions = {
  compilationLevel: number;
  expandAtleastGates: boolean;
  expandXorGates: boolean;
  recognizeAtleastGroups: boolean;
  recognizeXorGroups: boolean;
  ccf: boolean;
  keepNullGates: boolean;
};

export type SampleShape = {
  batch_size: number;
  bitpacks_per_batch: number;
  bits_per_bitpack: number;
  trials_per_pass: number;
  total_iterations: number;
};

export type SampleShapeOptions = {
  /** Total trial budget (0 = auto: unbounded / backend default). */
  numTrials: number;
  /** Optimistic per-node trials-per-pass target (0 = auto: fill device memory). */
  requestedTrialsPerPass: number;
};

export type SampleShapePayload = {
  sample_shape: SampleShape | null;
  error?: string;
};

export type Range3Description = {
  x: number;
  y: number;
  z: number;
  product: number;
};

export type ExecutionDeviceDescription = {
  type: string;
  name: string;
  vendor_id: number;
  vendor: string;
  driver_version: string;
  profile: string;
  version: string;
  opencl_c_version: string;
  aspects: number[];
  extensions: string[];
  max_compute_units: number;
  max_clock_frequency: number;
  max_work_item_dimensions: number;
  max_work_item_sizes_1d: Range3Description;
  max_work_item_sizes_2d: Range3Description;
  max_work_item_sizes_3d: Range3Description;
  work_item_independent_forward_progress: boolean;
  max_work_group_size: number;
  max_num_sub_groups: number;
  sub_group_independent_forward_progress: boolean;
  sub_group_sizes: number[];
  preferred_vector_width_char: number;
  max_mem_alloc_size: number;
  global_mem_size: number;
  global_mem_cache_size: number;
  global_mem_cache_line_size: number;
  global_mem_cache_type: string;
  local_mem_type: string;
  local_mem_size: number;
};

export type ExecutionSampleShaperDescription = {
  bitpack_type: string;
  bits_per_bitpack: number;
  requested_trials: number;
  rounded_trials: number;
  total_bits_to_sample: number;
  target_bits_per_iteration: number;
  bits_per_iteration: number;
  total_iterations: number;
  requested_trials_per_pass: number;
  realized_trials_per_pass: number;
  memory_budget_bitpacks_per_node: number;
  max_device_bytes: number;
  max_device_bits: number;
  cache_line_bitpacks: number;
  unlimited_iterations: boolean;
  shape_strategy: string;
  sample_shape: SampleShape & { num_bitpacks: number };
  resident_memory: {
    node_count: number;
    bytes_per_node: number;
    bytes_total: number;
    bits_per_node: number;
  };
};

export type ExecutionPartitionDescription = {
  id: number;
  layer: number;
  kind: string;
  connective: string;
  operation: string;
  node_indices: number[];
  dependency_partition_ids: number[];
  node_count: number;
  logical_range: Range3Description;
  local_range: Range3Description;
  global_range: Range3Description;
  work_group_count: number;
  work_item_count: number;
  logical_work_item_count: number;
  overprovisioned_work_items: number;
  bitpacks_per_node: number;
  bits_per_node: number;
  bytes_per_node: number;
  resident_bytes: number;
  min_fan_in: number;
  max_fan_in: number;
  total_fan_in: number;
  positive_input_count: number;
  negated_input_count: number;
};

export type ExecutionDependencyEdge = {
  source_partition: number;
  target_partition: number;
  reason: string;
};

export type ExecutionNodeDescription = {
  index: number;
  layer: number;
  partition: number;
  kind: string;
  connective: string;
  fan_in: number;
  positive_input_count: number;
  negated_input_count: number;
  atleast_min: number;
  module: boolean;
};

export type ExecutionPlanPayload = {
  schema: "mcscram.execution_plan.v1" | string;
  model?: string | null;
  source?: string | null;
  input_files?: string[];
  target?: string;
  target_name?: string;
  bitpack_type: string;
  bits_per_bitpack: number;
  node_count: number;
  device: ExecutionDeviceDescription;
  sample_shaper: ExecutionSampleShaperDescription;
  partitions: ExecutionPartitionDescription[];
  partition_count: number;
  dependency_edges: ExecutionDependencyEdge[];
  nodes: ExecutionNodeDescription[];
  compilation?: Record<string, unknown>;
  semantic_graph?: Record<string, unknown>;
  error?: string;
};

export type DagBundlePayload = {
  source: DagPayload;
  compiled: DagPayload;
  kernel: DagPayload;
};

async function fetchJsonFromServer<T>(path: string, searchParams?: Record<string, string>): Promise<T> {
  const response = await fetch(mcscramHttpUrl(path, searchParams));
  if (!response.ok) {
    throw new Error(`mcSCRAM request failed: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

/** Recursive ``*.xml`` / ``*.pcfb`` / nombrilo ``*.safetensors`` catalog from the server's search roots. */
export function fetchXmlInputCatalog(): Promise<XmlInputCatalog> {
  return fetchJsonFromServer<XmlInputCatalog>("/api/v1/model/xml-inputs");
}

/** Any XML input exposed by `GET /api/v1/model/fault-tree?model_id=…` on the server. */
export function fetchFaultTreeForModel(modelId: string): Promise<FaultTreePayload> {
  return fetchFaultTreeFromServer("/api/v1/model/fault-tree", { model_id: modelId });
}

/** Render-ready DAG payload with server-provided coordinates for the WebGL2 viewer. */
export function fetchDagForModel(modelId: string, options?: CompileDagOptions): Promise<DagPayload> {
  return fetchDagFromServer("/api/v1/model/dag", {
    model_id: modelId,
    ...(options
      ? {
          compilation_level: String(options.compilationLevel),
          expand_atleast_gates: String(options.expandAtleastGates),
          expand_xor_gates: String(options.expandXorGates),
          recognize_atleast_groups: String(options.recognizeAtleastGroups),
          recognize_xor_groups: String(options.recognizeXorGroups),
          ccf: String(options.ccf),
          keep_null_gates: String(options.keepNullGates),
        }
      : {}),
  });
}

/** Render-ready DAG payload for the source MEF structure before native compilation. */
export function fetchSourceDagForModel(modelId: string): Promise<DagPayload> {
  return fetchDagFromServer("/api/v1/model/source-dag", { model_id: modelId });
}

/** Render-ready DAG payload for the DirectEval kernel execution DAG. */
export function fetchKernelDagForModel(modelId: string, options?: CompileDagOptions): Promise<DagPayload> {
  return fetchDagFromServer("/api/v1/model/kernel-dag", {
    model_id: modelId,
    ...(options
      ? {
          compilation_level: String(options.compilationLevel),
          expand_atleast_gates: String(options.expandAtleastGates),
          expand_xor_gates: String(options.expandXorGates),
          recognize_atleast_groups: String(options.recognizeAtleastGroups),
          recognize_xor_groups: String(options.recognizeXorGroups),
          ccf: String(options.ccf),
          keep_null_gates: String(options.keepNullGates),
        }
      : {}),
  });
}

/** Native DirectEval sample-shape metadata for the current sampler form state. */
export function fetchSampleShapeForModel(
  modelId: string,
  compileOptions: CompileDagOptions,
  sampleOptions: SampleShapeOptions,
): Promise<SampleShapePayload> {
  return fetchJsonFromServer<SampleShapePayload>("/api/v1/model/sample-shape", {
    model_id: modelId,
    compilation_level: String(compileOptions.compilationLevel),
    expand_atleast_gates: String(compileOptions.expandAtleastGates),
    expand_xor_gates: String(compileOptions.expandXorGates),
    recognize_atleast_groups: String(compileOptions.recognizeAtleastGroups),
    recognize_xor_groups: String(compileOptions.recognizeXorGroups),
    ccf: String(compileOptions.ccf),
    keep_null_gates: String(compileOptions.keepNullGates),
    num_trials: String(sampleOptions.numTrials),
    requested_trials_per_pass: String(sampleOptions.requestedTrialsPerPass),
  });
}

/** Native DirectEval execution plan, including device, sample shaping, launch ranges, and partition dependencies. */
export function fetchExecutionPlanForModel(
  modelId: string,
  compileOptions: CompileDagOptions,
  sampleOptions: SampleShapeOptions,
): Promise<ExecutionPlanPayload> {
  return fetchJsonFromServer<ExecutionPlanPayload>("/api/v1/model/execution-plan", {
    model_id: modelId,
    compilation_level: String(compileOptions.compilationLevel),
    expand_atleast_gates: String(compileOptions.expandAtleastGates),
    expand_xor_gates: String(compileOptions.expandXorGates),
    recognize_atleast_groups: String(compileOptions.recognizeAtleastGroups),
    recognize_xor_groups: String(compileOptions.recognizeXorGroups),
    ccf: String(compileOptions.ccf),
    keep_null_gates: String(compileOptions.keepNullGates),
    num_trials: String(sampleOptions.numTrials),
    requested_trials_per_pass: String(sampleOptions.requestedTrialsPerPass),
  });
}

/** Source MEF, compiled PDAG, and kernel DAG in one server round-trip. */
export function fetchDagBundleForModel(modelId: string, options?: CompileDagOptions): Promise<DagBundlePayload> {
  return fetchJsonFromServer<DagBundlePayload>("/api/v1/model/dag-bundle", {
    model_id: modelId,
    ...(options
      ? {
          compilation_level: String(options.compilationLevel),
          expand_atleast_gates: String(options.expandAtleastGates),
          expand_xor_gates: String(options.expandXorGates),
          recognize_atleast_groups: String(options.recognizeAtleastGroups),
          recognize_xor_groups: String(options.recognizeXorGroups),
          ccf: String(options.ccf),
          keep_null_gates: String(options.keepNullGates),
        }
      : {}),
  });
}
