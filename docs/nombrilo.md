# nombrilo ↔ pcfb

`nombrilo.circuit` (version 1) is Kirill Shumilov’s safetensors encoding of a
Boolean circuit with Bernoulli inputs. fuzzybits imports it 1:1 into pcfb
core-1.1 and can export pcfb back to nombrilo. The reverse direction is
**lossy in general**; this page records exactly what is preserved.

Related: [#129](https://github.com/fuzzybit-org/fuzzybits/issues/129).

## CLI

```bash
# Detected from the safetensors header, not the extension.
fzb compile model.safetensors [-o model.pcfb] [--target NAME] [--weights FILE]

# Variant files (.e1/.e2/.e3/.e4/.f1–.f6) have no weights; the sibling
# <model>.safetensors is used automatically, or pass --weights.
fzb compile model.e1.safetensors --weights model.safetensors

fzb export --to nombrilo model.pcfb -o model.safetensors
fzb export --to nombrilo model.pcfb -o model.safetensors --strict   # no extras
```

`fzb compile` still accepts OpenPSA/MEF XML. The first input is sniffed: a
`nombrilo.circuit` header takes the 1:1 path (no PDAG preprocessing, no
`compile.*` metadata). XML takes the existing MEF path.

## Python

```python
import fuzzybits

pcfb = fuzzybits.compile("model.safetensors", target="r1")
pcfb = fuzzybits.nombrilo_to_pcfb("model.safetensors", weights="model.safetensors")

plain = fuzzybits.pcfb_to_nombrilo(pcfb, extras=False)
with_extras = fuzzybits.pcfb_to_nombrilo(pcfb, extras=True)
```

## nombrilo → pcfb (1:1)

| nombrilo | pcfb |
|---|---|
| variable 0 (TRUE) | tensor 0 = `BITWISE_ONES` |
| variables 1..B (inputs) | tensors 1..B = `SAMPLE` (`p = weights[v-1]`, `event_id = v`, `source = BASIC_EVENT`, name = first non-negated name of `v`) |
| gate *i* | tensor `B+1+i` = AND / OR / XOR / `ATLEAST{k}` |
| child literal `l` | tensor `l>>1`, in `inverted_inputs` iff `l` is even |
| empty AND | `BITWISE_ONES` at the same tensor index |
| empty OR / XOR | `BITWISE_ZEROS` at the same tensor index |
| at-least-k with `k > n` | `BITWISE_ZEROS` at the same tensor index |

Queries default to every gate with no parents. `--target` / `target=` overrides
that (gate name, input name, or a name on a negated literal). A negated target
appends one `NOT` after the gates so gate tensor indices stay put.

Names are stored losslessly in `nombrilo.names` (JSON `{names, lits}`),
including several names per variable and names on negated literals. The original
`operators` / `offsets` / `children` arrays are also snapshotted under
`nombrilo.operators`, `nombrilo.offsets`, `nombrilo.children` so child order
(which pcfb splits into `inputs` vs `inverted_inputs`) round-trips exactly.

Variant files record `nombrilo.weights_source = "inherited"` and omit `weights`
on the way back. Inline files record `"inline"` and keep bit-identical
float64 weights.

Required pcfb metadata is written (`general.generator`, `general.generator_version`,
`general.endianness`, `pcfb.opset`). Also `general.source_format =
"nombrilo.circuit-1"`, `general.source_sha256`, `general.model_name`. There are
**no** `compile.*` keys.

`event_id = v` starts at 1. The PDAG runtime shifts placement so variables do
not collide with Constant index 1 (`pdag_from_pcfb`); Philox still keys off the
placed PDAG index.

## pcfb → nombrilo (lossy)

nombrilo has fewer concepts than pcfb. Structural conversion:

| pcfb | nombrilo |
|---|---|
| `NOT` / `NO_OP` | folded into literals |
| `NAND` / `NOR` / `XNOR` | AND / OR / XOR referenced through a negated literal |
| `ATMOST k` | `not ATLEAST(k+1)` |
| `EXACTLY k` | `ATLEAST(k) AND not ATLEAST(k+1)` (adds gates) |
| `ZEROS` / `ONES` | constant literals 0 / 1 |
| SAMPLE order | inputs are always variables 1..B; interleaved SAMPLEs are pulled forward |

When the pcfb was produced from nombrilo (`general.source_format =
nombrilo.circuit-1` plus the snapshots above), arrays are restored exactly
instead of being reconstructed.

### What has no home in nombrilo

| pcfb field | fate |
|---|---|
| Queries (and `Query.confidence`) | not in the circuit format; stored in extras |
| `SampleArgs.source`, `event_id` | extras; otherwise all inputs are BASIC_EVENT with `event_id = v` |
| explicit `threshold` | recomputed from float64 `p` |
| pcfb metadata (`general.generator`, `compile.*`, `fuzzybits.*`) | extras |
| non-unique op / sample names | disambiguated with `_2`, `_3`, … |

### Extras (`fuzzybits.pcfb.*`)

`fzb export --to nombrilo` (and `pcfb_to_nombrilo(..., extras=True)`) writes
JSON strings into safetensors `__metadata__`:

| key | contents |
|---|---|
| `fuzzybits.pcfb.queries` | `[{lit, kind, name, confidence}]` |
| `fuzzybits.pcfb.event_ids` | `uint32[B]` in input order |
| `fuzzybits.pcfb.sources` | `["BASIC_EVENT", …]` |
| `fuzzybits.pcfb.metadata` | original pcfb key/value table |

Safetensors readers treat `__metadata__` as `dict[str, str]` and ignore unknown
keys; the Python `safetensors` package is used in tests to confirm that.
`--strict` / `extras=False` omits these keys (plain nombrilo).

When those keys are present, `nombrilo_to_pcfb` restores queries, event ids,
sources, and metadata so **our own** `fzb compile` output can round-trip
pcfb → nombrilo → pcfb with the same SAMPLE identity (and therefore the same
`fzb run` hit counts for a given trial count).

## Dataset tests

The Aralia nombrilo corpus is owner-local and is **not** committed. Point
tests at it with:

```bash
export FUZZYBITS_NOMBRILO_DIR=/path/to/input/aralia-nombrilo
```

If the directory is absent, dataset tests `SKIP`. Round-trip identity is
**array-level** (`operators` / `offsets` / `children` / names / name_lits /
weights bits, core `__metadata__` keys). Header JSON key order and space
padding are not required to be byte-identical; the writer emits a stable
order (`format`, `version`, `base_count`, then extras) and 8-byte-aligns the
header with trailing spaces.

## ATLEAST multiplicity

nombrilo counts **true children with multiplicity**. pcfb `ATLEAST` does the
same in `exact_eval` (duplicate tensor ids in `inputs` each contribute 1).
The PDAG runtime collapses duplicate arguments (`Gate::AddArg` /
`ProcessAtleastGateDuplicateArg`), so `fzb run` can disagree with nombrilo on
this edge case. It does not appear in the Aralia nombrilo files.
