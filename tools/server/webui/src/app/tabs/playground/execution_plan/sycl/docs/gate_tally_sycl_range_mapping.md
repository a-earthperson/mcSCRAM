# SYCL range mapping for gate and tally partitions

This note extends the basic-event mapping to the two downstream kernel families in the DirectEval pipeline:

```text
basic_event kernels -> gate kernels -> tally/popcount kernels
SYCL C++ backend    -> execution_plan API -> WebGL/MathBox frontend
```

The examples below are `nus9601` execution-plan partitions on an NVIDIA GPU with `bitpack_t = uint64`. As in the basic-event note, the useful hardware analogy for this backend is:

- SYCL work-item -> CUDA thread / lane instance.
- SYCL sub-group of size 32 -> CUDA warp.
- SYCL work-group -> CUDA thread block / CTA.
- SYCL compute unit -> NVIDIA SM.

The crucial correction is that there are two different gate geometries:

- Ordinary gates (`and`, `or`, `xor`, `not`, `nand`, `nor`, `null`) use one work-item per output bitpack.
- `k/n` / `atleast` gates use one work-group per output bitpack, with one work-item lane per bit in that bitpack.

Tally kernels return to the ordinary one-work-item-per-bitpack geometry, but each work-group reduces many popcounts into one atomic add.

## Common partition semantics

`layer_manager` builds partitions from the semantic PDAG as follows:

```text
one topological layer
  -> at most one basic-event partition
  -> one gate partition per connective type present in that layer
final tally request set
  -> one tally partition after all graph layers
```

The execution-plan partition fields are produced by `make_kernel_partition_description()` in `kernel_builder.h`. For all kernel kinds it records:

```text
logical_range.x = node_indices.size()
logical_range.y = sample_shape.batch_size
logical_range.z = sample_shape.bitpacks_per_batch

bitpacks_per_node = sample_shape.batch_size * sample_shape.bitpacks_per_batch
bits_per_node     = bitpacks_per_node * bits_per_bitpack
bytes_per_node    = bitpacks_per_node * sizeof(bitpack_t)
resident_bytes    = node_count * bytes_per_node
```

For gate partitions, `resident_bytes` matches the output buffer allocation owned by the gate block. For tally partitions, the same field is best read as bytes scanned per pass for the tallied source buffers: the tally block itself owns only tally structs and points at existing basic-event or gate buffers.

The native buffer data layer remains:

```text
row    = node ordinal inside this partition block
column = sample_idx = batch_idx * bitpacks_per_batch + bitpack_idx
tile   = one bitpack_t value
point  = one bit inside the bitpack_t tile
```

For the `nus9601` examples:

```text
batch_size         = 1
bitpacks_per_node = bitpacks_per_batch = 163664
bits_per_node     = 163664 * 64 = 10474496
bytes_per_node    = 163664 * 8 = 1309312
```

## Ordinary gate kernels

Ordinary gate partitions are built by `build_kernel_for_gates_of_type()` and execute `kernel::op<OpType>::operator()`.

### Work-item

One valid ordinary-gate work-item is addressed by:

```text
global_id(0) = gate_idx
global_id(1) = batch_idx
global_id(2) = bitpack_idx
```

It writes exactly one output bitpack:

```text
sample_idx = batch_idx * bitpacks_per_batch + bitpack_idx
g.buffer[sample_idx] = result_bitpack
```

The work-item loops serially over the gate's fan-in for that one `sample_idx`. Input buffers are arranged as:

```text
inputs[0 .. negated_inputs_offset)        = positive inputs
inputs[negated_inputs_offset .. num_inputs) = negated inputs
```

`kernel_builder.h` gathers positive inputs first, negative inputs second, then `create_gate_block()` stores:

```text
negated_inputs_offset = total_inputs - negated_input_count
```

For each input buffer, the work-item reads `input[sample_idx]`. Negative inputs are bitwise inverted before being folded into the operation:

```text
ordinary OR:   result |= val or ~val
ordinary AND:  result &= val or ~val
ordinary XOR:  result ^= val or ~val
ordinary NOT:  result  = val or ~val, then final invert
ordinary NAND: AND fold, then final invert
ordinary NOR:  OR fold, then final invert
```

The result bitpack preserves the trial-bit alignment established by the basic-event buffers. Bit position `b` in all input and output bitpacks refers to the same Monte Carlo trial lane.

### nd-range

For ordinary gates:

```text
logical launch shape = (num_gates, batch_size, bitpacks_per_batch)
global_range         = logical shape rounded up to local_range multiples
```

The ordinary gate kernel bounds-checks:

```text
gate_idx    >= gates_block.count
batch_idx   >= sample_shape.batch_size
bitpack_idx >= sample_shape.bitpacks_per_batch
```

Out-of-bounds work-items return without writing.

### `or` partition example

Captured partition:

```json
{
  "id": 115,
  "layer": 63,
  "kind": "gate",
  "connective": "or",
  "operation": "or",
  "node_indices": [1569],
  "dependency_partition_ids": [9, 114],
  "node_count": 1,
  "logical_range": { "x": 1, "y": 1, "z": 163664, "product": 163664 },
  "local_range": { "x": 1, "y": 1, "z": 1024, "product": 1024 },
  "global_range": { "x": 1, "y": 1, "z": 163840, "product": 163840 },
  "work_group_count": 160,
  "work_item_count": 163840,
  "logical_work_item_count": 163664,
  "overprovisioned_work_items": 176,
  "bitpacks_per_node": 163664,
  "bits_per_node": 10474496,
  "bytes_per_node": 1309312,
  "resident_bytes": 1309312,
  "min_fan_in": 3,
  "max_fan_in": 3,
  "total_fan_in": 3,
  "positive_input_count": 3,
  "negated_input_count": 0
}
```

Interpretation:

```text
logical data layer = 1 gate row x 163664 bitpack columns
local_range        = (1, 1, 1024)
global_range       = (1, 1, 163840)
work-groups        = 1 * 1 * (163840 / 1024) = 160
padding columns    = 163840 - 163664 = 176
```

One work-group covers:

```text
1 gate row * 1024 bitpack columns = 1024 output tiles
1024 output tiles * 64 bits/tile  = 65536 output trial bits
```

On the NVIDIA device:

```text
sub-group size           = 32
work-items per group     = 1024
sub-groups per group     = 32
sub-group visual footprint ~= 1 row x 32 adjacent bitpack columns
```

This is a cache-friendly shape: a warp reads and writes 32 adjacent `uint64` words for the same gate row, i.e. 256 bytes of output stores, split naturally across two 128-byte cache-line segments.

## `k/n` / `atleast` gate kernels

`k/n` gates use the specialization `kernel::op<core::Connective::kAtleast>`. The semantic output is still one bitpack per `(gate_idx, batch_idx, bitpack_idx)`, but the SYCL work decomposition is different.

### Work-group owns the output bitpack

The specialized `get_range()` rewrites the ordinary local range:

```text
NUM_BITS = sizeof(bitpack_t) * 8

local_range.x = 1
local_range.z = NUM_BITS
global_range.z = bitpacks_per_batch * NUM_BITS
```

For `uint64`, `NUM_BITS = 64`, so one work-group has 64 work-items. The group maps to:

```text
one gate_idx
one batch_idx
one bitpack_idx
64 local lanes, one lane per bit position inside the output bitpack
```

The kernel deliberately does not use `global_id(2)` as a bitpack column. It computes:

```text
bitpack_idx = item.get_group().get_group_id(2)
lane        = item.get_local_id(2)  // 0..63 for uint64
mask        = bitpack_t(1) << lane
```

Each lane counts how many inputs are true at its bit position:

```text
cnt = 0
for each positive input: cnt += (input[sample_idx] & mask) ? 1 : 0
for each negated input:  cnt += (~input[sample_idx] & mask) ? 1 : 0
my_bit = (cnt >= k) ? mask : 0
```

Then the work-group performs:

```text
result = reduce_over_group(bit_or(my_bit))
```

Only `lane == 0` writes:

```text
g.buffer[sample_idx] = result
```

Thus:

```text
one work-item -> one bit lane of one output bitpack
one work-group -> one complete output bitpack tile
```

This is not the ordinary "one work-item -> one bitpack tile" rule.

This specialization relies on the current sample-shaper convention that flattens sample work into Z:

```text
batch_size = 1
local_range.y = 1
```

If a future sample shape allowed `batch_size > 1` and `local_range.y > 1`, the present `reduce_over_group()` would span multiple `batch_idx` values inside one work-group. The captured `nus9601` plan is safe because Y is exactly one.

### `k/n` partition example

Captured partition:

```json
{
  "id": 1,
  "layer": 1,
  "kind": "gate",
  "connective": "k/n",
  "operation": "k/n",
  "node_indices": [2796, 2670, 3137, 2674, 2672, 2577, 2787, 2858, 2541, 3014, 2622, 2662, 3118, 3029, 2668, 2950, 2569],
  "dependency_partition_ids": [0],
  "node_count": 17,
  "logical_range": { "x": 17, "y": 1, "z": 163664, "product": 2782288 },
  "local_range": { "x": 1, "y": 1, "z": 64, "product": 64 },
  "global_range": { "x": 17, "y": 1, "z": 10474496, "product": 178066432 },
  "work_group_count": 2782288,
  "work_item_count": 178066432,
  "logical_work_item_count": 2782288,
  "overprovisioned_work_items": 175284144,
  "bitpacks_per_node": 163664,
  "bits_per_node": 10474496,
  "bytes_per_node": 1309312,
  "resident_bytes": 22258304,
  "min_fan_in": 3,
  "max_fan_in": 6,
  "total_fan_in": 54,
  "positive_input_count": 54,
  "negated_input_count": 0
}
```

The output data layer is:

```text
17 gate rows x 163664 bitpack columns
= 2782288 output bitpack tiles
= 178066432 output trial bits
```

The launched SYCL lattice is:

```text
global_range = (17, 1, 163664 * 64)
             = (17, 1, 10474496)

local_range  = (1, 1, 64)
```

The work-group grid is:

```text
groups_x = 17 / 1 = 17
groups_y = 1 / 1 = 1
groups_z = 10474496 / 64 = 163664

total work-groups = 17 * 1 * 163664 = 2782288
```

That is exactly one work-group per output bitpack tile. For MathBox, this is the right conceptual hierarchy:

```text
bit lane                 -> one point inside an 8x8 uint64 tile
work-item                -> one bit lane / one point contribution
sub-group / warp          -> 32 bit lanes, i.e. half of the uint64 tile
work-group / CTA          -> one complete output bitpack tile
partition logical layer   -> 17 x 163664 output bitpack tiles
```

### Important execution-plan caveat

For this `k/n` partition:

```text
logical_work_item_count = 2782288
work_item_count         = 178066432
overprovisioned         = 175284144
```

For ordinary kernels, `overprovisioned_work_items` means padded work-items that fail the bounds check. For `k/n`, that interpretation is not correct.

The apparent overprovisioning is:

```text
178066432 - 2782288
= 2782288 * (64 - 1)
```

Those are the 63 non-leader bit lanes in every 64-lane work-group, not invalid/padded work-items. They all perform useful counting work. In this captured example, there is no Z padding: `global_range.z` is exactly `bitpacks_per_batch * 64`.

The reason is structural: `make_kernel_partition_description()` always computes `logical_range.product()` as output bitpack tiles, while `kernel::op<kAtleast>::get_range()` reports a launched work-item lattice whose Z dimension is bit lanes. For `k/n`, `logical_work_item_count` is therefore better read as "logical output bitpack count", not actual useful work-item count.

## Tally / popcount kernels

Tally partitions are built by `build_tallies_for_layer()` and execute `kernel::tally::operator()`.

The tally block stores one tally struct per requested node, and each tally struct points to the source bitpack buffer produced by either a basic-event partition or a gate partition:

```text
tally[i].buffer = source_buffers[i]
```

No new bitpack output buffer is allocated for the tally partition.

### Work-item

One valid tally work-item is addressed by:

```text
global_id(0) = tally_idx
global_id(1) = batch_idx
global_id(2) = bitpack_id
```

It reads exactly one source bitpack:

```text
sample_idx = batch_idx * bitpacks_per_batch + bitpack_id
local_sum  = popcount(tallies[tally_idx].buffer[sample_idx])
```

The work-group reduces `local_sum` across its work-items:

```text
group_sum = reduce_over_group(plus(local_sum))
```

The group leader atomically adds:

```text
tally.num_one_bits += group_sum
```

The kernel also writes:

```text
tally.total_bits = iteration * batch_size * bitpacks_per_batch * bits_per_bitpack
```

The probability statistics (`mean`, `std_err`, confidence intervals) are currently populated host-side after the passes complete. The device-side statistics calculation remains in the source as commented-out code because doing it inside every work-group would repeat the same derived calculation and race benignly on the derived fields.

### nd-range

`build_tallies_for_layer()` asks the working-set heuristic for a local range with an X limit:

```text
local_limits = (1, 0, 0)
```

Then `kernel::tally::get_range()` also forces:

```text
local_range.x = 1
```

So a tally work-group covers one tally row and a rectangular run of sample bitpack columns. For the captured `batch_size = 1` case, it is:

```text
1 tally row x local_range.z bitpack columns
```

### Tally partition example

Captured partition:

```json
{
  "id": 116,
  "layer": 64,
  "kind": "tally",
  "connective": "popcount",
  "operation": "popcount",
  "node_indices": [1569],
  "dependency_partition_ids": [115],
  "node_count": 1,
  "logical_range": { "x": 1, "y": 1, "z": 163664, "product": 163664 },
  "local_range": { "x": 1, "y": 1, "z": 1024, "product": 1024 },
  "global_range": { "x": 1, "y": 1, "z": 163840, "product": 163840 },
  "work_group_count": 160,
  "work_item_count": 163840,
  "logical_work_item_count": 163664,
  "overprovisioned_work_items": 176,
  "bitpacks_per_node": 163664,
  "bits_per_node": 10474496,
  "bytes_per_node": 1309312,
  "resident_bytes": 1309312,
  "min_fan_in": 0,
  "max_fan_in": 0,
  "total_fan_in": 0,
  "positive_input_count": 0,
  "negated_input_count": 0
}
```

Interpretation:

```text
logical source layer = 1 tallied node row x 163664 bitpack columns
local_range          = (1, 1, 1024)
global_range         = (1, 1, 163840)
work-groups          = 160
padding columns      = 176
```

One work-group scans:

```text
1024 source bitpacks
1024 * 64 = 65536 source trial bits
```

Then it emits one atomic add to the tally accumulator. With 160 work-groups, this tally partition performs 160 atomic additions per pass for node `1569`, plus one repeated `total_bits` store per group leader.

## Local-range selection differences

All three downstream examples use the same device and sample shape, but the kernel-specific launch rules produce different local ranges:

```text
ordinary gate:
  working_set chooses (1, 1, 1024)
  get_range keeps it
  one work-group = 1 node row x 1024 bitpack columns

k/n gate:
  working_set initially sees the same problem shape
  get_range overrides x = 1 and z = bits_per_bitpack
  one work-group = 1 output bitpack tile

tally:
  working_set is called with x limited to 1
  get_range enforces x = 1
  one work-group = 1 tally row x 1024 source bitpack columns
```

For the `uint64` NVIDIA case:

```text
ordinary gate work-group = 1024 work-items = 32 warps
k/n gate work-group      = 64 work-items   = 2 warps
tally work-group         = 1024 work-items = 32 warps
```

## MathBox visualization implications

The base bitpack field is still the right frontend primitive:

```text
one row    = one partition-local node/gate/tally ordinal
one column = one flattened sample bitpack
one tile   = one uint64 bitpack drawn as 8x8 points
```

But the overlay mapping must be kernel-kind aware.

### Ordinary gates and tallies

The current generic mapping is exact:

```text
row = global_id.x
col = global_id.y * logical_range.z + global_id.z
```

For ordinary gates, one tile is one work-item output. For tallies, one tile is one work-item input scanned by popcount. In both cases:

```text
work-item overlay -> one bitpack tile
sub-group overlay -> 32 adjacent bitpack tiles when local.z >= 32 and batch_size = 1
work-group overlay -> local.x rows x (local.y * local.z) columns
```

For the captured ordinary gate and tally partitions, the first work-group window is:

```text
1 row x 1024 columns
```

If the scene clamps the rendered window to `16` columns, it is showing a crop from a much wider work-group.

### `k/n` gates

The generic mapping is not exact for `k/n` gates. The output data layer is still:

```text
17 rows x 163664 columns of uint64 output tiles
```

However, the launched work-item lattice is:

```text
17 rows x (163664 * 64) bit-lane work-items
```

Mapping `global_id.z` directly to tile columns would draw one `k/n` work-group as `1 x 64` bitpack tiles. That is false. The backend maps those 64 work-items to the 64 points inside one tile:

```text
tile row    = global_id.x
tile column = group_id.z
tile point  = local_id.z
```

For the `uint64` case, a faithful overlay should show:

```text
work-item -> one point inside the 8x8 tile
sub-group -> 32 points inside the tile, i.e. one warp over half the bit positions
work-group -> one 8x8 bitpack tile
```

So the `k/n` MathBox transform should derive a display geometry distinct from `local_range`:

```text
output_tile_grid.rows = logical_range.x
output_tile_grid.cols = logical_range.y * logical_range.z
work_group_tile_shape = (1, 1)
lane_grid_shape       = 8 x 8 for uint64
warp_lane_count       = 32
```

The execution-plan values remain useful, but `local_range.z = 64` should be interpreted as bit lanes, not tile columns.

## Summary for `nus9601`

Ordinary `or` gate partition:

```text
one work-item            -> one output uint64 bitpack tile
one warp/sub-group       -> 32 adjacent output tiles
one work-group           -> 1 x 1024 output tiles
real output grid         -> 1 x 163664 tiles
launched output grid     -> 1 x 163840 work-item positions
padding                  -> 176 bitpack columns
```

`k/n` gate partition:

```text
one work-item            -> one output bit lane / one point
one warp/sub-group       -> 32 bit lanes inside one output tile
one work-group           -> one output uint64 bitpack tile
real output grid         -> 17 x 163664 tiles
launched work-item grid  -> 17 x (163664 * 64) bit-lane positions
reported overprovision   -> lane expansion, not masked padding
```

Tally partition:

```text
one work-item            -> popcount one source uint64 bitpack tile
one warp/sub-group       -> 32 adjacent source tiles
one work-group           -> popcount/reduce 1 x 1024 source tiles
atomic additions/pass    -> 160 for this one tallied node
real source grid         -> 1 x 163664 tiles
launched source grid     -> 1 x 163840 work-item positions
padding                  -> 176 bitpack columns
```
