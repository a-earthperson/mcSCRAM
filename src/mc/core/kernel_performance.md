## Performance critique for `sm_120`

This section critiques the current gate, `k/n` / `atleast`, and tally kernels for the captured `sm_120` NVIDIA target. The relevant device facts from the execution plan are:

```text
SMs / compute units       = 36
reported CUDA cores / SM  = 128
sub-group / warp size     = 32
max work-group size       = 1024 work-items
max 3D local X            = 64
global cache line         = 128 bytes
local memory              = 49152 bytes
bitpack_t                 = uint64
bits per bitpack          = 64
sample shape              = batch_size 1, bitpacks_per_batch 163664
```

The analysis below deliberately does not treat CUDA cores as fixed owners of data. The useful unit is the warp scheduled on an SM; a CTA/work-group is only a scheduling and synchronization container. On this device, a 1024-work-item work-group is 32 warps, and a 64-work-item work-group is 2 warps.

### Ordinary gates (`and`, `or`, `xor`, `not`, `nand`, `nor`, `null`)

The captured ordinary `or` partition is the best case for the current ordinary-gate design:

```text
node_count      = 1
fan_in          = 3
local_range     = (1, 1, 1024)
global_range    = (1, 1, 163840)
work-groups     = 160
work-items      = 163840
real outputs    = 163664 uint64 bitpacks
```

One work-item computes one output `uint64` bitpack by loading one `uint64` from each input buffer and storing one `uint64` output. Ignoring descriptor and input-pointer traffic, a fan-in-3 OR costs:

```text
input bytes / output tile  = 3 * 8 = 24 B
store bytes / output tile  = 1 * 8 = 8 B
minimum memory traffic     = 32 B per output bitpack
```

For one warp:

```text
32 output bitpacks
3 input streams * 32 * 8 B = 768 B of input words
1 output stream * 32 * 8 B = 256 B of output words
```

Since 16 adjacent `uint64` words exactly fill one 128-byte cache line, each warp's output stores are naturally two cache-line segments. Each input stream is also contiguous for that gate row. The basic data access pattern is therefore good: coalesced, sequential, and aligned with the reported cache line when the buffers are aligned as allocated.

The main weaknesses are not coalescing, but occupancy shape, launch granularity, and metadata overhead:

- The partition has only 160 CTAs. Spread across 36 SMs, that is only about 4.4 CTAs per SM over the entire launch stream. Each CTA is large, so the total warp count is still high (`160 * 32 = 5120` warps), but the scheduler has little block-level slack and a visible tail.
- A 1024-thread CTA is the maximum work-group size. It supplies 32 warps from one CTA, but it can reduce CTA residency per SM depending on registers, backend allocation, and architectural block limits. This kernel has no obvious local-memory pressure, so register count and block-size limits dominate.
- The kernel rereads uniform gate metadata and `g.inputs[i]` pointers across many work-items. Those loads are likely cached, and the addresses are warp-uniform, but the source code does not make the broadcast/cache behavior explicit. If the compiler does not hoist or uniformize them well, pointer traffic becomes nontrivial for small fan-in.
- Arithmetic intensity is very low. For OR/AND/XOR with fan-in 3, the kernel is fundamentally memory bandwidth and memory latency bound. The bitwise ALU work is too small to hide a poor scheduling shape by itself.

#### Same input shape recommendations

Keep the current input shape:

```text
uint64 bitpack buffers
batch_size = 1
node-major contiguous buffers
one output buffer materialized per gate
input list as positive pointers followed by negated pointers
```

Under those assumptions, the first improvements to try are geometry and specialization changes:

```text
1. Benchmark smaller local_range.z values for one-row gate partitions.
```

For a single gate row, `local_range = (1, 1, 1024)` gives 160 CTAs. Using 512 or 256 would give:

```text
local.z = 512 -> 320 CTAs
local.z = 256 -> 640 CTAs
```

The memory access pattern remains contiguous and cache-line aligned. The trade-off is more CTAs and slightly more scheduling overhead versus better SM load balancing, more block-level latency hiding, and less tail risk. Because there is no inter-work-item communication in ordinary gates, a 1024-thread CTA is not intrinsically valuable here. A 256-thread or 512-thread CTA is a more plausible default for one-row, bandwidth-bound transforms on `sm_120`.

```text
2. Bucket ordinary gates by small fan-in and negation pattern.
```

The connective is already compile-time specialized, but fan-in is not. For common cases such as fan-in 1, 2, 3, and 4, separate kernels can unroll the input loop and reduce branch/control overhead:

```text
or_fanin_2, or_fanin_3, and_fanin_2, and_fanin_3, xor_fanin_2, ...
```

This preserves the input buffer shape. It changes only the partitioning policy: instead of one partition per `(layer, connective)`, use one partition per `(layer, connective, fan_in, negated_count)` for the common small buckets and keep a generic fallback for the tail.

```text
3. Cache gate input pointers at CTA scope for small one-row partitions.
```

For `local.x = 1`, every work-item in the CTA uses the same gate descriptor and same input pointer list. A small local-memory array populated by a few lanes can remove repeated pointer loads:

```text
lane i < fan_in: local_inputs[i] = g.inputs[i]
barrier
all lanes use local_inputs[i][sample_idx]
```

This should be benchmarked. For fan-in 3, the barrier may cost more than cached pointer loads. For larger fan-in ordinary gates, it becomes more attractive. A lower-risk variant is to copy `g.inputs` and scalar fields into local variables once per work-item and rely on the compiler's uniform-load handling.

```text
4. Use a geometry policy that depends on node_count and fan_in.
```

The current `working_set` heuristic maximizes the work-group budget without knowing the kernel body. That is too generic. Ordinary gates need no group-wide synchronization, so the policy should prefer enough CTAs to keep all 36 SMs busy and avoid maximum-sized blocks unless measured performance justifies them. A reasonable first rule is:

```text
if node_count <= 4 and fan_in <= 8:
  local_range = (1, 1, 256 or 512)
else:
  keep x/y/z power-of-two tiling, but cap product at 256 or 512 before trying 1024
```

#### Relaxed-assumption recommendations

Relax progressively:

```text
A. Relax partition boundaries, keep buffers.
```

Fuse chains of ordinary gates when a producer partition has a single consumer and the intermediate buffer is not needed for tally or another downstream gate. A fused kernel can compute:

```text
intermediate = op_a(inputs...)
output       = op_b(intermediate, other_inputs...)
```

inside one work-item for each bitpack. This removes one global store, one later global load, and one kernel launch. The trade-off is higher register pressure and a more complex compiler/scheduler contract.

```text
B. Relax materialization of intermediate gate buffers.
```

If a gate output is only an internal value and never tallied or shared, do not allocate a resident buffer for it. Inline it into the consumer expression tree. This moves the design toward expression-tree kernels over bitpacks. It is often the largest possible bandwidth win because current DirectEval materializes every semantic gate as a full `node_count * bitpacks_per_node` buffer.

```text
C. Relax homogeneous-connective partitioning.
```

Compile larger fused kernels for mixed connectives over a local subgraph. This gives up the simple one-connective kernel model but can reduce launch count and DRAM traffic substantially for deep, narrow PDAG layers such as the late `nus9601` single-row gates.

### `k/n` / `atleast` gates

The current `k/n` specialization is the most questionable design for the captured `nus9601` shape.

Captured shape:

```text
node_count          = 17
fan_in range        = 3..6
total fan_in        = 54
local_range         = (1, 1, 64)
global_range        = (17, 1, 10474496)
work-groups         = 2782288
work-items          = 178066432
semantic outputs    = 2782288 uint64 bitpacks
```

The kernel assigns one CTA to one output bitpack and one lane to one bit position. For `uint64`, that is 64 work-items, or two warps, per output word. Each lane loops over all inputs and tests one bit with a mask. The group then performs a bitwise-OR reduction and one lane stores the final word.

For fan-in 3..6, this is severe over-parallelization:

```text
current useful work per output word:
  64 lanes * fan_in masked tests
  64 lanes participate in group reduction
  1 final uint64 store

bitpack/SWAR alternative:
  1 lane/thread * fan_in uint64 loads
  a small number of bitwise counter operations
  1 uint64 store
```

The current lane-per-bit decomposition has some redeeming properties:

- It exposes enormous block-level parallelism: 2,782,288 CTAs for this partition, so SM saturation is not a problem.
- Each CTA is only 2 warps, so occupancy can be high if block-count and scheduler limits allow enough resident CTAs.
- Per-lane state is tiny.

But those strengths are outweighed by the shape mismatch for small fan-in:

- A warp's lanes load the same input word address for a given input and bitpack. NVIDIA hardware and caches may reduce the damage for uniform addresses, but the kernel still issues lane work for 32 independent bit tests where one bitwise word operation would represent all 64 trial bits.
- The CTA performs a group reduction to reconstruct a word that already existed naturally as the machine register width.
- A 64-thread CTA per output bitpack creates millions of very small CTAs. That gives saturation, but also high block scheduling overhead and poor work per CTA.
- The execution-plan "overprovisioned" count is not padding; it is a symptom that the launch space is bit-lane expanded by 64x.
- For the captured fan-in range 3..6, the memory and instruction overhead are both excessive relative to a word-parallel implementation.

#### Same input shape recommendations

Keep the current input shape and output materialization:

```text
uint64 input buffers
uint64 output buffers
batch_size = 1
one output bitpack per sample_idx
positive and negated input pointer lists
```

The highest-confidence recommendation is:

```text
1. Replace the lane-per-bit `atleast` kernel with a word-parallel bit-sliced counter for small and moderate fan-in.
```

For each output `uint64`, one work-item can compute the whole k-of-n result. A general bit-sliced counter maintains binary counter planes across all 64 trial lanes simultaneously. For example, for up to 7 inputs, three `uint64` planes can represent per-bit counts:

```text
count bit 0 plane: c0
count bit 1 plane: c1
count bit 2 plane: c2
```

Each input word increments this bit-sliced counter with Boolean full-adder logic. At the end, compare the counter planes to the runtime or compile-time threshold `k` to produce a `uint64` mask. This turns the captured fan-in-3..6 case into:

```text
fan_in global loads
O(fan_in * log2(fan_in)) word bitwise ops
one global store
no group reduction
no 64x lane expansion
```

For `k = 1`, dispatch to OR. For `k = n`, dispatch to AND. For `k = 0`, write all ones. For `k > n`, write zero. These special cases should not use a cardinality counter at all.

```text
2. If the lane-per-bit kernel is retained, use sub-groups rather than full work-groups as the output unit.
```

A 32-lane warp can cover two bit positions per lane for `uint64`:

```text
lane handles bit lane and lane + 32
sub-group reduction reconstructs a 64-bit output
one warp/sub-group per output bitpack
```

This halves the work-items and can avoid a full work-group reduction. A CTA can then contain multiple independent sub-groups, each producing one output bitpack:

```text
local_range product = 128 or 256
4 or 8 sub-groups per CTA
4 or 8 output bitpacks per CTA
```

That improves CTA work granularity while preserving the bit-lane algorithm.

```text
3. Route by fan-in and threshold.
```

Use a simple policy:

```text
fan_in <= 32:
  word-parallel SWAR/cardinality kernel
fan_in > 32:
  benchmark word-parallel versus distributed input-parallel or lane-parallel kernels
k == 1:
  OR kernel
k == fan_in:
  AND kernel
```

The exact threshold should be measured on `sm_120`, but the captured 3..6 fan-in is well inside the word-parallel regime.

```text
4. Fix execution-plan accounting for `atleast`.
```

Even if the kernel stays as-is, expose both concepts:

```text
logical_output_bitpacks = node_count * batch_size * bitpacks_per_batch
launched_work_items     = global_range.product()
lane_expansion_factor   = bits_per_bitpack
masked_padding_items    = true out-of-bounds work-items only
```

This is not a runtime optimization, but it prevents the frontend and performance analysis from mistaking useful lane work for padding.

#### Relaxed-assumption recommendations

Relax progressively:

```text
A. Relax "one generic atleast kernel".
```

Generate specialized kernels for common `(fan_in, k, negated_count)` buckets. A 2-of-3 gate, 2-of-4 gate, and 3-of-5 gate can each have compact Boolean formulas that beat a generic counter. This still uses the same bitpack buffers but changes compile-time specialization and partitioning.

```text
B. Relax runtime input-pointer lists for hot buckets.
```

Store fixed-arity input pointers in a compact struct or separate arrays. For fan-in 3..6, avoiding pointer-list indirection and loop bounds can help the compiler unroll and schedule loads aggressively.

```text
C. Relax intermediate materialization and fuse cardinality subgraphs.
```

If `k/n` gates feed only one downstream gate, fuse them with consumers and tally where possible. The current implementation pays a full output-buffer write for every `k/n` result, then another kernel reads it. Fusion can eliminate that traffic.

```text
D. Relax the bitpack type or memory layout only for cardinality-heavy workloads.
```

For very high fan-in cardinality gates, a different representation can become attractive: store transposed chunks or per-input tiles arranged so a CTA cooperatively processes many inputs and many bitpacks with shared-memory staging. This is not justified by the captured 3..6 fan-in case, but it may be useful for models dominated by large voting gates.

### Tally / popcount

The captured tally partition is:

```text
node_count       = 1
local_range      = (1, 1, 1024)
global_range     = (1, 1, 163840)
real bitpacks    = 163664
work-groups      = 160
work-items       = 163840
atomics/pass     = 160
source bytes     = 163664 * 8 = 1309312 B
source bits      = 10474496
```

The kernel is a streaming read plus popcount plus group reduction:

```text
one work-item -> popcount one uint64
one CTA       -> reduce 1024 popcounts
one atomic    -> add CTA sum to tally.num_one_bits
```

This is a reasonable first implementation. Popcount on `uint64` is cheap; the source scan is memory bandwidth dominated. The reduction and 160 atomics are not alarming for one tallied node.

The weaknesses are still important:

- The same 1024-thread CTA issue appears again. With only 160 CTAs, block-level scheduling slack is low for 36 SMs.
- A 1024-way group reduction is more synchronization/reduction work than needed if each thread only contributes one 0..64 popcount.
- The atomic add count is small, but all atomics target the same tally counter. For one node this is fine; for many nodes the contention is distributed by node, but for each node the groups still converge on one address.
- The kernel writes `total_bits` once per group leader. This is benign but redundant.
- The tally pass rereads a full output buffer that may have just been produced by the preceding gate. For final outputs, this store-then-read sequence is pure bandwidth overhead if the bitpack buffer is not needed for any other purpose.

#### Same input shape recommendations

Keep the current source buffers and tally struct:

```text
source buffer = uint64[num_bitpacks]
tally owns no source buffer
one final num_one_bits counter per tallied node
host computes mean/std_err/CI
```

Recommended changes:

```text
1. Benchmark local_range.z = 256 or 512 for one-row tally partitions.
```

This changes:

```text
1024 -> 160 CTAs, 160 atomics
512  -> 320 CTAs, 320 atomics
256  -> 640 CTAs, 640 atomics
```

The extra atomics are still small compared with the memory scan, while the smaller CTA can improve SM balancing, reduce tail effects, and reduce reduction depth. On `sm_120`, this should be measured rather than inferred; the current 1024-thread choice is a heuristic, not an occupancy model.

```text
2. Give each work-item multiple consecutive words.
```

Use fewer threads and more per-thread instruction-level parallelism:

```text
local_range = (1, 1, 256)
each work-item popcounts 2 or 4 consecutive uint64 words
CTA covers 512 or 1024 source words
reduce 256 partial sums instead of 1024
```

This preserves contiguous memory access, reduces reduction fan-in, and gives each lane enough arithmetic to hide some load latency. It also makes the CTA size less extreme without increasing atomics if the per-CTA word coverage remains 1024.

```text
3. Use a two-level reduction only if atomics become visible in profiling.
```

For this partition, 160 relaxed atomics to one counter is probably not the bottleneck. If larger tally sets or smaller local ranges increase atomic pressure, write one partial sum per CTA to a temporary partials buffer and run a second reduction kernel per tally. This replaces atomics with extra global writes and an extra launch, so it is only justified when atomic serialization shows up in profiling.

```text
4. Store `total_bits` once.
```

The `total_bits` value is independent of `bitpack_id` and CTA. It can be written by a single dedicated work-item, by a separate initialization/finalization step, or on the host when collecting tallies. This is minor, but it removes redundant stores and clarifies the kernel's ownership of accumulator fields.

#### Relaxed-assumption recommendations

Relax progressively:

```text
A. Relax "tally is always a separate pass".
```

For final gates that are tallied and not otherwise consumed, fuse tallying into the producer kernel:

```text
producer computes output bitpack
producer popcounts output bitpack
producer contributes to tally accumulator
optional: skip writing output bitpack if no consumer needs it
```

This can remove one full global store and one full later global read. For a final ordinary OR gate with fan-in 3, this changes the pipeline from:

```text
gate: 3 loads + 1 store
tally: 1 load + popcount + reduction
```

to:

```text
fused: 3 loads + popcount + reduction
```

If the output buffer is not needed, the store is also eliminated. This is the most important tally optimization for final-only nodes.

```text
B. Relax "one tally counter per full pass".
```

Accumulate per-CTA partial sums in a persistent partial buffer across several passes, then reduce less frequently. This can reduce atomic pressure and host collection overhead, but it complicates iteration semantics and memory lifetime.

```text
C. Relax source-buffer residency for tally-only subgraphs.
```

If a subgraph exists solely to produce a tallied top event, compute and tally bitpacks in a streaming fashion without storing every intermediate node's full per-pass buffer. This is the natural endpoint of fusing gate and tally kernels: resident memory falls, DRAM traffic falls, and sample shape may be increased because fewer buffers need to coexist.

### Cross-kernel design critique

The current execution design is excellent for explainability and for simple partition-local visualization:

```text
every node has a resident bitpack buffer
every partition has a 3D range
every output bitpack maps cleanly to a MathBox tile
dependencies are explicit partition edges
```

That design is not throughput-optimal for `sm_120`. Its main costs are:

- It materializes every intermediate gate buffer, even when the value may be consumed once.
- It uses one local-range heuristic for kernels with very different synchronization and arithmetic profiles.
- It treats `atleast` as a bit-lane cooperative kernel even when the native `uint64` bitpack representation is already a 64-lane SIMD-within-a-register structure.
- It creates many narrow, sequentially dependent partitions in deep PDAGs. Even if each partition has many work-items, launch overhead and global-memory round trips accumulate layer by layer.

The recommended optimization path is therefore:

```text
1. Keep the current data shape.
   Add kernel-specific local-range policies and small-fan-in specializations.

2. Keep bitpack buffers but relax partitioning.
   Bucket by fan-in/k/negation pattern, and fuse simple producer-consumer chains.

3. Relax full intermediate materialization.
   Inline single-use gates and fuse final tally accumulation into producers.

4. Only for cardinality-heavy models, consider alternate layouts.
   Use cooperative tiled cardinality kernels or transposed/staged input layouts when fan-in is large enough to justify abandoning the simple row-major bitpack buffer model.
```

For the captured `nus9601` partitions, the highest-value changes are:

```text
ordinary gates:
  benchmark 256/512-thread CTAs and small-fan-in unrolling

k/n / atleast:
  replace the 64-lane-per-output kernel for fan-in 3..6 with a word-parallel SWAR/cardinality kernel

tally:
  benchmark 256/512-thread CTAs and multi-word-per-thread popcount
  fuse final tally into producer kernels when the output buffer is not otherwise needed
```