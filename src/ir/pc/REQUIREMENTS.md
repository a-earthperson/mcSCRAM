# fuzzybits::ir::pc Requirements

Status: design contract for the probabilistic circuit IR
Scope: `src/ir/pc`, plus the public API and backend boundaries that reference it
Audience: contributors designing IR ingestion, compiler/solver lowering, PC transforms, and PC serialization

This document defines the lock-in-safe requirements for `fuzzybits::ir::pc`. The PC IR is the canonical semantic
representation of probabilistic Boolean circuits in the target architecture, but it is not a promise that one physical
layout, one serializer, or one solver schedule is the IR.

The design intentionally separates:

1. Semantic commitments: what the circuit means.
2. Physical views: how consumers traverse it efficiently.
3. Transform reports: how derived circuits preserve identity and projection.
4. Serializer mappings: how files import/export the semantic circuit.
5. Backend lowerings: how solvers and compilers schedule execution.

## 1) Boundary Decision

`pc::circuit` is a public IR storage and view type owned by `src/ir/pc`, but it is not the public orchestration
protocol.

The cross-layer protocol remains:

- `api::ir_handle` for ownership/type erasure across pipeline, compiler, and solver boundaries;
- `api::ir_backend_descriptor` and structured PC shape metadata for compatibility planning;
- `api::diagnostic_list` for unsupported or lossy conversions;
- backend-private lowering code for solver/compiler execution details.

The concrete `pc::circuit` type may be used by backend implementations that explicitly opt into PC consumption. Public
compiler and solver headers must not expose PC implementation details unless the architecture contract is deliberately
revised in the same change. This keeps `pc` available as the high-performance canonical IR while preserving the
API-layer dependency rule.

Allowed:

- an IR backend constructs `pc::circuit` and wraps it in `api::ir_handle`;
- a compiler or solver backend checks the handle descriptor, then extracts a `pc::circuit` through a public `src/ir/pc`
  accessor in its private implementation;
- pipeline planning reasons over PC descriptors, capabilities, and conversion diagnostics.

Not allowed:

- public solver/compiler APIs that require users to include backend-private headers;
- direct dependencies on MEF, pcfb, or scramble headers from `src/ir/pc`;
- treating `api::ir_view` as the high-performance PC contract;
- exposing solver execution scratch state as PC semantic nodes.

## 2) Semantic Core

The v1 PC semantic core is a flat probabilistic Boolean DAG.

The initial core must support:

- dense node identifiers scoped to one circuit;
- explicit edge polarity;
- acyclic Boolean dependencies;
- basic events with probability sources;
- constants;
- Boolean connectives with direct probabilistic Boolean semantics;
- cardinality gates as a single semantic operator;
- multiple target observables.

Node ids are storage ids, not stable semantic identity. Transforms that reorder, compact, expand, or decompose nodes
must return identity/provenance maps whenever diagnostics, target projection, CCF projection, or result attribution can
be affected.

R1 intentionally fixes node and edge indexing columns to 32-bit unsigned storage
instead of templating `pc::circuit` on an index type. This is an R1 semantic-core
limit, not a statement that all future physical views must be 32-bit. The reasons
are:

- the public `pc::circuit` type should remain a single concrete IR storage type
  behind `api::ir_handle`;
- templating the circuit would immediately leak representation width into every
  compiler, solver, descriptor, serializer, smoke test, and handle-extraction
  path;
- the R1 target scale is millions of nodes, where 32-bit dense ids and offsets
  keep the hot columns compact;
- larger layouts should be introduced as explicit R2+ physical storage/view
  variants with structured shape metadata and compatibility diagnostics.

Builders must diagnose attempts to exceed the representable R1 id or offset
range. Consumers that need larger graphs must request a later large-graph view or
storage implementation rather than relying on a public template instantiation.

The initial target table models observables only:

```cpp
struct target_view {
    target_id    id;
    bool_node_id node;
    observable_role role;
};
```

Observable roles classify why a node is observable, for example `primary`, `watch`, or `projection_anchor`. Targets must
not encode requested result kinds, convergence tracking, solver policies, or report-only domain payloads.

Domain-specific structures stay outside the Boolean/probability core:

- fault trees;
- event trees;
- sequences;
- end states;
- initiating events;
- report layout metadata.

Those structures belong in IR-backend domain wrappers and typed API projection contracts. If a projection has to survive
into reporting, it must be modeled as typed API data or transform provenance, not as an opaque blob inside the PC core.

## 3) Edge Encoding

The semantic edge API is explicit:

```cpp
struct edge_ref {
    bool_node_id source;
    bool         complement;
};
```

The semantic contract must not encode complement by negating signed node ids.

Packed edge representations are allowed only as physical views. A packed representation must specify:

- unsigned storage width;
- reserved invalid value;
- maximum representable source id;
- whether id zero is valid;
- behavior at the width limit;
- conversion to and from explicit `edge_ref`;
- conversion to and from pcfb `inputs` and `inverted_inputs`;
- diagnostics for unrepresentable ids.

This avoids inheriting PDAG sign conventions, `INT_MIN` hazards, `abs()` ambiguity, sentinel collisions, and ABI leakage
from `id_t` template choices.

## 4) Boolean Opcode Scope

The v1 Boolean opcode set must be limited to operators with direct probabilistic Boolean semantics.

Required leaves:

- `basic_event`;
- `constant`.

Conditionally allowed leaf:

- `house_event`, only if house-event identity must survive ingestion after resolution. If not required for
  diagnostics/projection, it should lower to `constant`.

Allowed Boolean gates:

- `and`;
- `or`;
- `xor`;
- `not`;
- `nand`;
- `nor`;
- `xnor`.

Cardinality gates should use one semantic form:

```cpp
cardinality(low, high)
```

with `0 <= low <= high <= fanin`. `atleast(k)`, `atmost(k)`, and `exactly(k)` are views or builder conveniences over
this semantic form, not separate irreversible payload conventions.

Forbidden from canonical PC v1:

- MC execution helpers such as `sample` and `tally`;
- pcfb tensor/signal operators such as `conv_*`, `fft_*`, `adder`, `comparator`, `muxed`, and `demux`;
- backend-specific compiled artifact operations.

Such operations may appear in pcfb compatibility layers or compiled artifacts, but they are not part of the FuzzyBits
probabilistic Boolean interoperability layer unless a later RFC proves their semantics belong here.

## 5) Probability Sources

The v1 semantic core needs probability sources sufficient for current probability and Monte Carlo ingestion without
freezing a full expression DAG.

Required v1 probability sources:

- literal probability in `[0, 1]`;
- named parameter reference with declared probability domain;
- unavailable or unsupported source carrying diagnostics.

Each `basic_event` refers to one probability source. A solver may consume only the source kinds it declares in its
descriptor. If a source cannot be evaluated under the current request, planning or lowering must produce diagnostics
before execution.

The probability-source table must define:

- domain validation rules;
- finite-value requirements;
- parameter identity;
- stable diagnostic identity for unsupported imports;
- whether the source is deterministic or requires an expression evaluator.

The full expression DAG is deferred until the evaluator contract in Section 9 is complete.

## 6) Construction And Finalization

`pc::circuit` is immutable after finalization. Mutation is confined to builder
types or transform-local construction state.

The builder contract must support:

- appending nodes with dense ids;
- appending explicit `edge_ref` operands;
- declaring probability sources before or during basic-event construction;
- declaring target observables;
- validating local arity and payload invariants;
- returning structured diagnostics for semantic validation failures.

Finalization must:

- validate acyclicity or produce diagnostics;
- validate edge endpoints and polarity;
- validate cardinality bounds;
- validate probability-source domains that are checkable without a solver;
- compact builder scratch storage into read-only storage;
- compute or invalidate shape metadata;
- not impose mandatory kind partitioning;
- not require topological node ids unless explicitly requested by a derived view.

Finalized circuits must expose read-only spans or equivalent non-owning views for
high-volume traversal. They must not expose mutable node access, per-node owning
objects, or virtual dispatch in the hot traversal path.

`builder::from(circuit)` is allowed, but it is a clone-into-mutable operation.
It must not imply that sealed circuits are mutable or that transforms can edit a
shared circuit in place.

R1 storage uses `std::vector` for owned columns and summary lists. This is a
deliberate storage choice, not a semantic commitment that the IR is a dynamically
resizable runtime graph:

- finalized circuits expose borrowed spans; consumers do not mutate the vectors;
- vector columns give compact contiguous storage, stable iteration order, and
  direct compatibility with CSR-style views;
- insertion happens only during builder/finalization, outside solver hot paths;
- uniqueness constraints are enforced by construction/validation when needed,
  not by exposing tree/hash set containers in the public representation;
- small descriptor lists such as `op_codes_present` are deterministic,
  insertion-ordered enumerations, not lookup-heavy sets.

`std::set` would encode uniqueness in the container, but it would also impose
node allocations, pointer chasing, comparator policy, and non-columnar layout.
For the PC IR, uniqueness is an invariant and vectors are the storage mechanism.

## 7) Physical Views

Physical layout is not semantic identity. A valid `pc::circuit` is not required to be kind-partitioned, topologically
ordered, or serialized in pcfb order.

The implementation should expose derived or cached physical views:

### 7.1 CSR Edge View

`csr_edge_view` presents:

- node count;
- edge offsets;
- edge counts or sentinel offsets;
- explicit or packed edge payloads;
- payload accessors for cardinality and probability-source references.

The CSR view is the primary no-pointer-chasing traversal shape for generic consumers.

### 7.2 Topological View

`topological_view` presents a permutation satisfying dependency order. It may be:

- a canonical node-id layout, if the circuit was constructed that way;
- a side permutation over base node ids;
- a lowered schedule for a specific consumer.

Consumers that require dependency order must request this view or require a property saying the base layout already
satisfies it.

### 7.3 Kind Bucket View

`kind_bucket_view` groups nodes by opcode. It may use:

- contiguous ranges if the underlying storage already has that layout;
- index lists if not;
- a generated permutation for a specific backend.

Kind buckets are not canonical circuit identity. They must not invalidate topological semantics.

### 7.4 Solver Schedule View

`solver_schedule_view` is backend-specific lowering metadata. Monte Carlo, BDD, exact enumeration, and compiler backends
may require different schedules. These schedules belong to derived views or compiled artifacts, not to the semantic
circuit itself.

## 8) Shape And Capability Metadata

`api::feature_set` remains useful for coarse yes/no properties, but PC compatibility requires structured shape metadata.

The PC descriptor should include:

- node count;
- edge count;
- node id width;
- maximum representable node id for each physical view;
- opcode set present;
- probability source kinds present;
- target count and observable roles present;
- whether an expression evaluator is required;
- CCF declaration and expansion state;
- available physical views;
- whether any import losses or unsupported placeholders are present;
- approximate memory shape.

Feature strings should be reserved for coarse properties such as:

- acyclic;
- has topological view;
- has CSR edge view;
- has kind bucket view;
- has literal probability sources;
- has parameter probability sources;
- has unsupported probability sources;
- has CCF declarations;
- CCF fully expanded.

Parameterized constraints must live in structured descriptors, not in string-like capability names.

## 9) Expression DAG RFC Gate

The expression DAG must not be frozen until the evaluator semantics are specified.

The RFC must define:

- whether an expression evaluates to probability, rate, time-dependent probability, distribution over probabilities,
  symbolic parameter, or another typed quantity;
- evaluation context fields: mission time, parameter bindings, epistemic sample index, RNG stream, unit system, and
  backend options;
- domain validation for probability intervals, non-negative rates, positive scales, finite arithmetic, and invalid
  operations;
- random-variable identity, including shared deviates, independent deviates, and correlated deviates;
- deterministic expression identity for common subexpression and diagnostics;
- external function policy;
- unsupported MEF expression representation;
- serialization and round-trip rules;
- backend compatibility diagnostics.

Only after this RFC is accepted may a `basic_event` carry a mandatory `expr_ref`. Until then, basic events refer to the
narrower probability-source table.

## 10) CCF RFC Gate

CCF support must preserve provenance. A single `ccf_expanded` boolean is insufficient.

The CCF RFC must define:

- unexpanded group declarations;
- beta, alpha, MGL, phi, and custom parameterization;
- member identity;
- original basic-event identity;
- whether expansion state is none, partial, or complete;
- generated event and generated node provenance;
- original-to-expanded maps;
- expanded-to-original maps;
- adjusted independent event probabilities;
- common-shock event semantics;
- how CCF expansion affects target projection and result attribution;
- serialization/import/export behavior;
- diagnostics for unsupported or lossy CCF models.

CCF expansion must return a transform report. It must not merely toggle a property bit.

## 11) Transform API

Transforms must carry reports and account for large graphs.

The transform result shape should be:

```cpp
struct transform_result {
    circuit          output;
    transform_report report;
};
```

`transform_report` should include:

- transform id and version;
- checked preconditions;
- produced postconditions;
- diagnostics;
- old-to-new node maps where available;
- generated-node provenance;
- generated-target provenance when applicable;
- invalidated physical views;
- memory amplification estimate or measurement;
- whether the transform was semantics-preserving, normalizing, lossy, or projection-affecting.

Transforms that may materially alter graph identity include:

- constant propagation;
- null/pass-through elimination;
- negation normalization;
- cardinality decomposition;
- XOR decomposition;
- common subexpression elimination;
- topological scheduling;
- compaction;
- CCF expansion.

For 10M-node inputs, the implementation must support move-aware, fused, or builder-backed transforms where safe.
Immutability is the consumption contract for sealed circuits, not a requirement that every transform allocate a complete
copy for every phase.

## 12) MEF Ingestion Requirements

MEF-to-PC conversion must be explicit about supported and unsupported semantics.

The initial conversion subset should include:

- basic events with literal probability expressions;
- basic events with parameter probability expressions when the parameter domain is a probability;
- house events that resolve to constants or to identifiable house-event leaves if identity is required;
- Boolean gates mapping to the v1 opcode set;
- cardinality gates mapping to `cardinality(low, high)`;
- top-event observables mapping to PC targets.

Unsupported or deferred MEF features must not be silently erased:

- non-literal expression trees before the expression RFC is implemented;
- random deviates before evaluator randomness is specified;
- external functions before external function policy is specified;
- CCF groups before the CCF RFC is implemented;
- event-tree sequence projection unless represented by typed API projection data;
- unsupported connectives or malformed cardinality bounds.

The converter must return diagnostics for unsupported features and must mark the resulting PC descriptor with any
unsupported placeholders. Solver/compiler backends that require fully evaluable probabilities must reject those
placeholders during planning.

## 13) Serialization And pcfb Mapping

Serialization is not the IR contract.

pcfb remains:

- a compatibility backend;
- a candidate wire format;
- a possible source of zero-copy physical views after a separate layout RFC.

pcfb is not:

- the canonical semantic identity of `pc`;
- proof that `pc` is lossless-isomorphic to pcfb;
- the required in-memory layout for `pc::circuit`.

The pcfb mapping spec must classify each feature as:

- lossless import/export;
- lossless with index or layout translation;
- lossy but diagnosable;
- unsupported.

Known mapping hazards that must be addressed:

- pcfb tensor operators outside the PC Boolean opcode set;
- input polarity order and role preservation;
- pcfb multi-output or structural operators;
- pcfb execution order versus PC semantic ids;
- expression/probability side tables;
- CCF declarations and expansion provenance;
- target and projection metadata.

Zero-copy mmap construction is a physical-view optimization. It requires a separate contract for lifetime, alignment,
endianness, edge encoding, string storage, and unsupported schema features.

## 14) Subcircuits And Modules

Subcircuits/modules are deferred until instance semantics are specified.

The module RFC must define:

- whether ids are global or per subcircuit;
- whether a subcircuit is a definition, an instance, a partition, or a pcfb compatibility shape;
- input and output port binding;
- edge polarity across ports;
- multi-output semantics;
- target references across module boundaries;
- CCF references across module boundaries;
- flattening semantics;
- flattening provenance maps;
- view/schedule generation for module-aware and flattened consumers.

R1 must remain a flat DAG with multi-target observables. Module detection and module lowering are transforms after the
above semantics exist.

## 15) Migration Phases

The migration phases are requirements gates, not permission to add placeholders that look stable.

### R0: Architecture Boundary

- `pc::circuit` is public `src/ir/pc` storage.
- Public orchestration remains `api::ir_handle`.
- Compiler/solver public headers stay dependency-light.
- Backend implementations may opt into `fuzzybits_ir_pc`.
- Architecture docs must record this boundary.

### R1: Semantic Core

- Implement flat DAG semantics.
- Use explicit `edge_ref`.
- Add the narrow Boolean opcode set.
- Add target observables.
- Add literal and parameter probability sources.
- Add builder/finalization and read-only consumption.

### R2: Physical Views

- Add CSR, topological, kind-bucket, and schedule views.
- Keep views derived or cached.
- Do not make any one view canonical identity.

### R3: MEF Ingestion

- Convert the supported MEF subset.
- Emit diagnostics and unsupported placeholders for deferred semantics.

### R4: Expression RFC

- Freeze evaluator context, domains, units, and random-variable identity.
- Add expression DAG only after the RFC.

### R5: CCF RFC

- Add CCF declarations, expansion transform, provenance, and projection semantics.

### R6: Transform API

- Add transform reports and memory budgets.
- Replace scramble preprocessing only after semantic parity and projection behavior are proven.

### R7: Serialization

- Specify pcfb and any future serializers as mappings.
- Add import/export after semantic PC is stable.

### R8: Modules

- Add subcircuits/modules only after instance semantics, port binding, id scope, CCF, targets, and flattening are fixed.

## 16) Explicit Non-Commitments

The PC v1 requirements do not commit to:

- signed negative ids for complement edges;
- mandatory kind partitioning;
- pcfb as the canonical on-disk format;
- zero-copy mmap PC construction;
- expression DAG opcode closure;
- CCF expansion as a boolean property;
- subcircuits/modules in R1;
- MC `sample` or `tally` as PC opcodes;
- solver convergence flags inside IR targets;
- pure-copy transform implementations for large graphs.

Any of these may become valid implementation choices later, but none may become v1 ABI without a semantic RFC and
compatibility story.
