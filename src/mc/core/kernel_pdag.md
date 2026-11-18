# Kernel PDAG / Execution DAG Metadata

## Purpose

The "kernel DAG" is currently intended to explain how a compiled PDAG is organized for
DirectEval execution. That is useful, but incomplete unless the description also includes
the device capabilities and the sample-shaping / launch-geometry decisions that determine
how much work each kernel partition actually performs.

The important distinction:

- The compiled PDAG is a semantic Boolean graph.
- The current "kernel DAG" is mostly the semantic PDAG annotated with layer and partition IDs.
- The actual execution graph is a queue of kernel partitions with dependencies, launch ranges,
  buffers, sample shape, and device-dependent local-range choices.

If the frontend is meant to show how work is chunked, it should consume an execution-plan
description, not only a semantic graph with execution annotations.

## What Is Returned Today

Native `CompiledPdag::to_kernel_graph()` builds a temporary `layer_manager` and calls
`layer_manager::describe_kernel_dag()`.

The current native description contains:

- Per semantic PDAG node:
  - `index`
  - `layer`
  - `partition`
  - `kernel_kind`
  - `kernel_connective`
- Per partition:
  - `id`
  - `layer`
  - `kind`
  - `connective`
  - `node_indices`
  - `node_count`
- Per graph:
  - `batch_size`
  - `bitpacks_per_batch`
  - `total_iterations`

The backend converts this into the existing `DagPayload` by preserving the semantic
nodes and edges, then attaching the `kernel` object.

This is directionally correct for a "where did each PDAG node land?" view. It is not yet
a complete "what work will run on this device?" view.

## Immediate Mismatch To Fix

`CompiledPdag::to_kernel_graph()` currently instantiates:

```cpp
using bitpack_t = std::uint32_t;
```

The Python `LayerManager` wrapper and the probability-analysis path use `std::uint64_t`.
That means the graph-description path can report a different sample shape and different
bits-per-pass than the execution path the Python sampler will use.

This should be normalized before adding more metadata. Either:

- Use the same `bitpack_t` everywhere for DirectEval Python execution and description, or
- Explicitly include `bitpack_bits` / `bitpack_type` in every execution-plan payload and make
  it obvious which execution family was described.

For the current Python-facing work, prefer one canonical native type alias shared by
`to_kernel_graph()`, `create_layer_manager()`, and any new execution-plan serializer.

## What Shapes The Work

The execution shape is determined by four interacting layers.

### 1. Semantic Graph Shape

Source:

- `layer_manager::layered_toposort()`
- `layer_manager::gather_layer_nodes()`
- `layer_manager::map_nodes_by_layer()`

Effects:

- PDAG nodes are grouped into topological layers.
- Within each layer:
  - variables become one basic-event kernel partition, if present;
  - gates are grouped by connective, one kernel partition per connective per layer.
- Tally kernels are added after graph kernels, based on the tally node map.

Current payload coverage:

- Mostly covered for variables and gates.
- Tally partition is not represented in `describe_kernel_dag()`, even though it is a real
  queueable and appears in kernel logs.
- Execution dependency edges are not represented.

### 2. Device Capabilities

Source:

- `working_set`
- `sycl::device` queried inside `sample_shaper`

Relevant fields already available in C++:

- device type, name, vendor, vendor ID, driver, profile, version, OpenCL C version
- aspects and extensions
- max compute units, max clock frequency
- max work-item dimensions and max work-item sizes for 1D/2D/3D
- max work-group size
- max sub-groups, sub-group sizes, preferred vector width
- work-item and sub-group forward-progress flags
- max allocation size, global memory size, cache size, cache line size, cache type
- local memory type and size

Current payload coverage:

- Not exposed structurally.
- Only available as log output through `operator<<(working_set)`.

### 3. Sample Shaping

Source:

- `sample_shaper`

Inputs:

- requested `num_trials`
- number of graph nodes
- `requested_trials_per_pass` (optimistic per-node target; `0` = auto)
- device `max_mem_alloc_size`
- device cache line size
- `bitpack_t` width

Important derived values:

- `requested_num_trials`
- rounded `num_trials`
- `total_bits_to_sample`
- `target_bits_per_iteration` (per-node memory budget, in bits)
- `memory_budget_bitpacks_per_node`
- selected `sample_shape.batch_size` (always `1`)
- selected `sample_shape.bitpacks_per_batch` (the solved `B`)
- `num_bitpacks = batch_size * bitpacks_per_batch`
- `bits_per_iteration = num_bitpacks * bits_per_bitpack` (realized trials/pass/node)
- `total_iterations`
- no-limit mode when requested trials is zero

Shaping model (single backend-agnostic path):

The shaper solves for one integer `B = bitpacks_per_batch` (with `batch_size = 1`,
so each work-item owns one bit-pack). With `P = bits_per_bitpack`:

- (C1) Memory budget (hard): `num_nodes * B * sizeof(bitpack) <= max_alloc_bytes`
  ⇒ `B <= B_mem = floor(max_alloc_bytes / (num_nodes * sizeof(bitpack)))`.
- (C2) Total-work cap (bounded runs only): `B <= B_total = ceil(total_bits_to_sample / P)`.
- (C3) Optimistic target: `B_req = requested_trials_per_pass > 0 ? ceil(requested_trials_per_pass / P) : B_mem`
  (i.e. `0` fills the memory budget).
- (C4) Cache-line alignment: when `B >= cache-line`, round `B` down to a whole cache line.
- (C5) Positivity: `B >= 1`.

Result: `B = align_down(min(B_req, B_mem, B_total))`. Because of (C4) the realized
`bits_per_iteration = B * P` may be smaller than `requested_trials_per_pass` — the
request is optimistic. The same path is used for CPU and GPU backends.

Special cases:

- If no fixed trial limit is supplied (`num_trials == 0`), shaping uses a bounded
  placeholder (`DEFAULT_UNBOUNDED_TRIALS`) to size the resident shape while
  `TOTAL_ITERATIONS` stays zero (run forever).
- The shape is intrinsically 1-D (`batch_size = 1`); all per-pass work lives in
  `bitpacks_per_batch`.

Current payload coverage:

- Partially covered.
- We expose `batch_size`, `bitpacks_per_batch`, `bits_per_bitpack`,
  `trials_per_pass`, and `total_iterations` through the recent Python sample-shape wrapper.
- `to_kernel_graph()` still only includes `batch_size`, `bitpacks_per_batch`, and
  `total_iterations`.
- The reason why the shape was chosen is not exposed.
- The memory budget, rounded trial count, and NVIDIA greedy path are not exposed.

### 4. Kernel Launch Geometry

Source:

- `kernel_builder.h`
- `kernel::basic_event::get_range()`
- `kernel::op<...>::get_range()`
- `kernel::tally::get_range()`
- `working_set::compute_optimal_local_range_3d()`

For each kernel partition, C++ computes:

- local range `(x, y, z)`
- global range `(x, y, z)`, rounded up to local multiples
- logical work dimensions:
  - basic event: `(num_events, batch_size, bitpacks_per_batch)`
  - gate: `(num_gates, batch_size, bitpacks_per_batch)`
  - tally: `(num_tallies, batch_size, bitpacks_per_batch)`
- work-group count
- over-provisioned work-items
- operation kind:
  - basic-event sampling
  - gate operation by connective
  - tally/popcount

Current payload coverage:

- Not exposed structurally.
- Only printed in logs by kernel `get_range()` methods.
- The `queueable` stores `nd_range_`, but only in a typed/protected base and without an
  introspection API on `queueable_base`.

## What Is Missing For A Frontend Execution Graph

Minimum missing pieces:

1. Device summary:
   - enough to explain why local ranges and sample shape are what they are.
2. Sample-shaper summary:
   - requested versus rounded trials;
   - bits per bitpack;
   - bitpacks per pass;
   - trials/bits per pass;
   - memory budget and selected resident bytes;
   - total iterations.
3. Launch descriptor per kernel partition:
   - partition ID;
   - layer;
   - stage/kind/connective;
   - node count and node indices;
   - logical range;
   - local range;
   - global range;
   - work-group count;
   - over-provisioning;
   - sample-shape snapshot.
4. Execution dependencies:
   - partition-to-partition edges, not just semantic PDAG edges.
5. Tally partition metadata:
   - currently omitted from `describe_kernel_dag()`.
6. Consistent bitpack type:
   - current description and Python execution paths can diverge.

Valuable:

- per-node fan-in and negated-input count;
- at-least gate threshold `k`;
- memory allocation blocks by partition type;
- resident bytes per event and total resident bytes;
- whether the selected shape came from closest-fit or NVIDIA greedy allocation.

## Recommended Payload Shape

Prefer a new execution-plan payload instead of stretching the current render DAG schema.

Sketch:

```json
{
  "schema": "mcscram.execution_plan.v1",
  "model": "...",
  "target": "...",
  "compilation": { "...": "..." },
  "device": {
    "type": "gpu",
    "name": "...",
    "vendor": "...",
    "driver_version": "...",
    "max_compute_units": 36,
    "max_work_group_size": 1024,
    "max_work_item_sizes_3d": [64, 1024, 1024],
    "sub_group_sizes": [32],
    "max_mem_alloc_size": 8071348224,
    "global_mem_size": 8071348224,
    "global_mem_cache_line_size": 128,
    "local_mem_size": 49152
  },
  "sample_shaper": {
    "bitpack_type": "uint64",
    "bits_per_bitpack": 64,
    "requested_trials": 1000000000,
    "rounded_trials": 1000000000,
    "target_bits_per_iteration": 22492160,
    "bits_per_iteration": 22492160,
    "total_iterations": 45,
    "requested_trials_per_pass": 0,
    "realized_trials_per_pass": 22492160,
    "memory_budget_bitpacks_per_node": 351440,
    "shape_strategy": "auto_memory_budget_cacheline_aligned",
    "sample_shape": {
      "batch_size": 1,
      "bitpacks_per_batch": 351440,
      "num_bitpacks": 351440,
      "trials_per_pass": 22492160
    },
    "resident_memory": {
      "node_count": 2734,
      "bytes_per_node": 2811520,
      "bytes_total": 7686695680
    }
  },
  "partitions": [
    {
      "id": 0,
      "layer": 0,
      "kind": "basic_event",
      "operation": "sample",
      "node_indices": [2, 3, 4],
      "node_count": 3,
      "logical_range": [3, 1, 351440],
      "local_range": [64, 1, 16],
      "global_range": [64, 1, 351440],
      "work_group_count": 21965,
      "work_item_count": 22492160,
      "overprovisioned_work_items": 0,
      "sample_shape": {
        "batch_size": 1,
        "bitpacks_per_batch": 351440
      }
    }
  ],
  "dependency_edges": [
    { "source_partition": 0, "target_partition": 1, "reason": "buffer_dependency" }
  ],
  "semantic_graph": {
    "nodes": [],
    "edges": []
  }
}
```

The frontend can still derive the old visualization from this, but it should not have to
infer execution topology from the semantic PDAG.

## Incremental Change Path

Use this path if we want to keep most current code structure.

### Step 1: Normalize bitpack type

Create one Python DirectEval binding alias and use it for:

- `CompiledPdag::to_kernel_graph()`
- `PyLayerManager`
- sample-shape endpoint

This removes the current `uint32_t` versus `uint64_t` description mismatch.

### Step 2: Add structured summaries

Add public summary structs/getters:

- `sample_shaper_summary`
- `working_set_summary`

Do not parse logs. The data already exists as fields on `sample_shaper` and `working_set`.
Make it structured at the source.

### Step 3: Capture launch descriptors when kernels are built

The launch ranges are known in `kernel_builder.h` exactly when each queueable is created.
Store them immediately instead of trying to recover them later from `queueable_base`.

Minimal native structs:

```cpp
struct range3_description {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
};

struct kernel_launch_description {
    std::size_t id = 0;
    std::size_t layer = 0;
    std::string kind;
    std::string connective;
    std::vector<index_t_> node_indices;
    range3_description logical_range;
    range3_description local_range;
    range3_description global_range;
};
```

Then:

- pass `layer` and an output metadata vector into kernel builders;
- compute local/global range as today;
- append the descriptor next to `queueables_`.

This keeps `queueable_base` simple and avoids type-erased `nd_range` introspection.

### Step 4: Represent tally kernels

Add a partition descriptor for the tally queueable. It should have:

- `kind = "tally"`
- `connective = "popcount"`
- tally node indices
- launch geometry
- dependencies on the producing partitions

### Step 5: Add partition dependency edges

The builders already compute dependency sets through `queueables_by_index_` and
`layer_dependencies`. To expose execution edges, each queueable needs a stable partition ID.

Incremental approach:

- maintain a map from `queueable_base*` to partition ID in `layer_manager`;
- when building a new partition, convert `layer_dependencies` into partition-edge records;
- keep semantic PDAG edges separately.

### Step 6: Expose through Python and backend

Add:

- `LayerManager.execution_plan` or `CompiledPdag.to_execution_plan()`
- backend route `/model/execution-plan`
- frontend type `ExecutionPlanPayload`

The current `/model/kernel-dag` can either remain as a compatibility view or be rewritten
to consume the new execution-plan payload.

## Rewrite Path

Breaking changes are allowed, and the clean design is:

1. Rename the concept:
   - `to_kernel_graph()` -> `to_execution_plan()`
   - `/model/kernel-dag` -> `/model/execution-plan`
2. Define the primary graph as kernel partitions, not PDAG nodes.
3. Treat the semantic PDAG as an overlay:
   - semantic nodes can be inspected;
   - each semantic node points to its producing partition;
   - semantic edges do not masquerade as execution edges.
4. Make sample shape and device summary first-class, not nested decorations.
5. Make frontend visualization choose among:
   - semantic graph view;
   - partition execution DAG view;
   - memory/work-shape summary view.

This would make the frontend honest: a partition node is what is submitted to SYCL,
and its geometry tells the user how much work is actually launched.

## Frontend Implications

Current frontend can show:

- semantic PDAG layout;
- node layer and partition;
- partition count;
- partial sample shape.

It cannot accurately show:

- device-driven work-group choices;
- launched work-item counts;
- over-provisioning from rounded global ranges;
- tally kernels;
- kernel dependencies;
- memory pressure / allocation budget;
- why a CUDA device selects a huge resident sample shape.

Once `ExecutionPlanPayload` exists, the frontend should add a dedicated execution-plan
panel or inspector:

- device summary card;
- sample-shaper summary;
- partition DAG;
- selected partition details:
  - node count;
  - operation;
  - logical/local/global ranges;
  - work-groups;
  - work-items;
  - bytes touched/resident.

The existing render DAG can continue to answer "where did this PDAG node land?".
The execution-plan view should answer "what kernel partition runs, on what device shape,
and with what launch geometry?".

## Main Recommendation

Do not keep extending `DagPayload` as the source of truth for execution. It is a render
format for semantic graph visualization. Add a native `execution_plan` description at the
`layer_manager` level, expose it directly through Python, and let the backend pass it
through with minimal transformation.

The smallest high-value native changes are:

1. Normalize the bitpack type used by description and execution.
2. Expose structured `sample_shaper` and `working_set` summaries.
3. Store launch descriptors at kernel-build time.
4. Add tally partitions and dependency edges.

After those changes, the frontend will have enough information to explain both the shape
of the work graph and the device-dependent choices that make the same PDAG execute very
differently on CUDA, Metal, OpenMP, or another SYCL backend.
