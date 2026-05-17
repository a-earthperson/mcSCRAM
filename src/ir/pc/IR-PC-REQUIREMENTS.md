# `fuzzybits::ir::pc` Requirements

Status: design contract for the probabilistic circuit IR
Scope: `src/ir/pc`, plus the public role contracts and backend boundaries that reference it
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

`pc::circuit` is a public IR storage and view type owned by `src/ir/pc`, but it is not the public orchestration protocol.

The cross-layer protocol is:

- `ir::handle` for ownership and type erasure across pipeline, compiler, and solver boundaries;
- `ir::descriptor` for provider, endpoint, storage type, form, name, and coarse capabilities;
- `pc::shape_summary` for structured PC shape metadata used by compatibility planning;
- `api::diagnostic_list` and `api::result<T>` for unsupported or lossy conversions;
- backend-private lowering code for solver/compiler execution details.

The concrete `pc::circuit` type may be used by backend implementations that explicitly opt into PC consumption. Public compiler and solver headers must not expose PC implementation details unless the architecture contract is deliberately revised in the same change. This keeps `pc` available as the high-performance canonical IR while preserving the public
dependency rule.

Allowed:

- an IR builder constructs `pc::circuit` and wraps it in `ir::handle` through `pc::circuit::to_handle()`;
- a compiler or solver backend checks the handle descriptor and storage type, then extracts a `pc::circuit` through `pc::circuit_from()` in private implementation code;
- pipeline planning reasons over endpoint descriptors, `api::capability_set`, `api::form_id`, and diagnostics.

Not allowed:

- public solver/compiler APIs that require users to include backend-private headers;
- direct dependencies on provider-private headers from `src/ir/pc`;
- treating `ir::view` as the high-performance PC traversal contract;
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

Node ids are storage ids, not stable semantic identity. Transforms that reorder, compact, expand, or decompose nodes must return identity/provenance maps whenever diagnostics, target projection, CCF projection, or result attribution can be affected.

R1 intentionally fixes node, target, probability-source, and edge-indexing columns to 32-bit unsigned storage instead of templating `pc::circuit` on an index type. This is an R1 semantic-core limit, not a statement that all future physical views must be 32-bit. The reasons are:

- the public `pc::circuit` type should remain a single concrete IR storage type behind `ir::handle`;
- templating the circuit would immediately leak representation width into every compiler, solver, descriptor, serializer, smoke test, and handle-extraction path;
- the R1 target scale is millions of nodes, where 32-bit dense ids and offsets keep hot columns compact;
- larger layouts should be introduced as explicit R2+ physical storage/view variants with structured shape metadata and compatibility diagnostics.

Builders must diagnose attempts to exceed the representable R1 id or offset range. Consumers that need larger graphs must request a later large-graph view or storage implementation rather than relying on a public template instantiation.

The current target table models observables only:

```cpp
struct target_view final {
    target_id id;
    bool_node_id node;
    observable_role role = observable_role::primary;
};
```

`observable_role` classifies why a node is observable. Current roles are `primary`, `watch`, and `projection_anchor`. Targets must not encode requested result kinds, convergence tracking, solver policies, or report-only domain payloads.

Domain-specific structures stay outside the Boolean/probability core:

- fault trees;
- event trees;
- sequences;
- end states;
- initiating events;
- report layout metadata.

Those structures belong in role-specific domain wrappers, typed projection data, or transform provenance. If a projection has to survive into result serialization, it must be modeled as typed data or provenance, not as an opaque blob inside the PC core.

## 3) Identifiers And Edge Encoding

Storage-local ids use the strongly typed `pc::id<tag_type>` template:

```cpp
using bool_node_id = id<id_tags::bool_node>;
using probability_source_id = id<id_tags::probability_source>;
using target_id = id<id_tags::target>;
```

Each id stores a `std::uint32_t`, uses `std::numeric_limits<std::uint32_t>::max()` as its invalid value, and exposes `valid()` plus `raw()` accessors. These ids are storage-local. Transforms that remap nodes are responsible for returning provenance maps rather than promising global id stability.

The semantic edge API is explicit:

```cpp
struct edge_ref final {
    bool_node_id source;
    bool complement = false;
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
- conversion to and from external format input/polarity encodings;
- diagnostics for unrepresentable ids.

This avoids signed-id hazards such as negative sentinel collisions, `INT_MIN`, `abs()` ambiguity, and ABI leakage from provider-specific id types.

## 4) Boolean Opcode Scope

The v1 Boolean opcode set must be limited to operators with direct probabilistic Boolean semantics.

Current leaves:

- `op_code::basic_event`;
- `op_code::constant`;
- `op_code::house_event`, only if house-event identity must survive ingestion after resolution. If not required for diagnostics/projection, it should lower to `op_code::constant`.

Current Boolean gates:

- `op_code::and_gate`;
- `op_code::or_gate`;
- `op_code::xor_gate`;
- `op_code::not_gate`;
- `op_code::nand_gate`;
- `op_code::nor_gate`;
- `op_code::xnor_gate`.

Cardinality gates use one semantic form:

```cpp
op_code::cardinality
```

with payload columns `cardinality_low()` and `cardinality_high()`, satisfying `0 <= low <= high <= fanin`. `atleast(k)`, `atmost(k)`, and `exactly(k)` are views or builder conveniences over this semantic form, not separate irreversible payload conventions.

Forbidden from canonical PC v1:

- execution helpers such as `sample` and `tally`;
- tensor/signal operators such as `conv_*`, `fft_*`, `adder`, `comparator`, `muxed`, and `demux`;
- backend-specific compiled artifact operations.

Such operations may appear in compatibility layers, derived views, or compiled artifacts, but they are not part of the FuzzyBits probabilistic Boolean interoperability layer unless a later RFC proves their semantics belong here.

## 5) Probability Sources

The v1 semantic core needs probability sources sufficient for current probability ingestion without freezing a full expression DAG.

Current probability source kinds:

```cpp
enum class probability_source_kind : std::uint8_t {
    literal,
    parameter,
    unsupported,
};
```

Current payload:

```cpp
struct probability_source final {
    probability_source_kind kind = probability_source_kind::unsupported;
    double literal = 0.0;
    std::string parameter_name;
    std::string diagnostic;
};
```

Factory helpers are:

- `probability_source::literal_probability(double)`;
- `probability_source::parameter_probability(std::string)`;
- `probability_source::unsupported_probability(std::string)`.

Builder helpers are:

- `builder::add_literal_probability(double)`;
- `builder::add_parameter_probability(std::string_view)`;
- `builder::add_unsupported_probability(std::string)`.

Each `op_code::basic_event` refers to one probability source through `probability_source_refs()`. A solver may consume only the source kinds it declares in its descriptor. If a source cannot be evaluated under the current request, planning or lowering must produce diagnostics before execution.

The probability-source table must define:

- domain validation rules;
- finite-value requirements;
- parameter identity;
- stable diagnostic identity for unsupported imports;
- whether the source is deterministic or requires an expression evaluator.

The full expression DAG is deferred until the evaluator semantics in Section 9 are complete.

## 6) Construction And Finalization

`pc::circuit` is immutable after finalization. Mutation is confined to `pc::builder` or transform-local construction state.

The current builder contract supports:

- `reserve_nodes(std::uint32_t)`;
- `add_literal_probability(double)`;
- `add_parameter_probability(std::string_view)`;
- `add_unsupported_probability(std::string)`;
- `add_basic_event(probability_source_id)`;
- `add_constant(bool)`;
- `add_house_event(bool)`;
- `add_gate(op_code, std::span<edge_ref const>)`;
- initializer-list overloads for small hand-built gates;
- `add_not(edge_ref)`;
- `add_and(...)`;
- `add_or(...)`;
- `add_cardinality(std::uint32_t low, std::uint32_t high, ...)`;
- `add_target(bool_node_id, observable_role = observable_role::primary)`.

Finalization is:

```cpp
auto result = std::move(builder).seal();
```

and returns:

```cpp
struct finalize_result final {
    circuit output;
    api::diagnostic_list diagnostics;
    [[nodiscard]] auto ok() const -> bool;
};
```

Finalization must:

- validate acyclicity or produce diagnostics;
- validate edge endpoints and polarity;
- validate cardinality bounds;
- validate probability-source domains that are checkable without a solver;
- compact builder scratch storage into read-only storage;
- compute or invalidate shape metadata;
- not impose mandatory kind partitioning;
- not require topological node ids unless explicitly requested by a derived view.

Finalized circuits expose read-only spans for high-volume traversal. They must not expose mutable node access, per-node owning objects, or virtual dispatch in the hot traversal path.

`builder::from(circuit)` is allowed as a future API, but it must be a clone-into-mutable operation. It must not imply that sealed circuits are mutable or that transforms can edit a shared circuit in place.

R1 storage uses `std::vector` for owned columns and summary lists. This is a deliberate storage choice, not a semantic commitment that the IR is a dynamically resizable runtime graph:

- finalized circuits expose borrowed spans; consumers do not mutate the vectors;
- vector columns give compact contiguous storage, stable iteration order, and direct compatibility with CSR-style views;
- insertion happens only during builder/finalization, outside solver hot paths;
- uniqueness constraints are enforced by construction/validation when needed, not by exposing tree/hash set containers in
  the public representation;
- small descriptor lists such as `op_codes_present` are deterministic insertion-ordered enumerations, not lookup-heavy
  sets.

`std::set` would encode uniqueness in the container, but it would also impose node allocations, pointer chasing, comparator policy, and non-columnar layout. For the PC IR, uniqueness is an invariant and vectors are the storage mechanism.

## 7) Circuit Accessors And Handle Extraction

Current `pc::circuit` accessors are:

- `valid()`;
- `operator bool()`;
- `descriptor() -> ir::descriptor const &`;
- `shape() -> shape_summary const &`;
- `node_count() -> std::uint32_t`;
- `edge_count() -> std::uint32_t`;
- `op_codes() -> std::span<op_code const>`;
- `operand_offsets() -> std::span<std::uint32_t const>`;
- `operand_counts() -> std::span<std::uint32_t const>`;
- `edges() -> std::span<edge_ref const>`;
- `probability_source_refs() -> std::span<probability_source_id const>`;
- `probability_sources() -> std::span<probability_source const>`;
- `cardinality_low() -> std::span<std::uint32_t const>`;
- `cardinality_high() -> std::span<std::uint32_t const>`;
- `constant_values() -> std::span<std::uint8_t const>`;
- `targets() -> std::span<target_view const>`;
- `operands(bool_node_id) -> std::span<edge_ref const>`;
- `to_handle() -> ir::handle`.

Current handle extraction helpers are:

- `pc::storage_type_id() -> api::type_id`;
- `pc::endpoint_id() -> api::endpoint_id`;
- `pc::make_descriptor(shape_summary const & = {}) -> ir::descriptor`;
- `pc::is_pc(ir::handle const &) -> bool`;
- `pc::circuit_from(ir::handle const &) -> circuit const *`.

Private backend implementations may use `pc::is_pc()` and `pc::circuit_from()` only after descriptor and capability checks. Generic orchestration must not downcast arbitrary handles speculatively.

## 8) Physical Views

Physical layout is not semantic identity. A valid `pc::circuit` is not required to be kind-partitioned, topologically ordered, or serialized in any external format order.

The current concrete storage already exposes CSR-like edge columns through `operand_offsets()`, `operand_counts()`, and `edges()`. Additional physical views are R2+ derived or cached views:

### 8.1 CSR Edge View

`csr_edge_view` presents:

- node count;
- edge offsets;
- edge counts or sentinel offsets;
- explicit or packed edge payloads;
- payload accessors for cardinality and probability-source references.

The CSR view is the primary no-pointer-chasing traversal shape for generic consumers.

### 8.2 Topological View

`topological_view` presents a permutation satisfying dependency order. It may be:

- a canonical node-id layout, if the circuit was constructed that way;
- a side permutation over base node ids;
- a lowered schedule for a specific consumer.

Consumers that require dependency order must request this view or require a capability saying the base layout already satisfies it.

### 8.3 Kind Bucket View

`kind_bucket_view` groups nodes by opcode. It may use:

- contiguous ranges if the underlying storage already has that layout;
- index lists if not;
- a generated permutation for a specific backend.

Kind buckets are not canonical circuit identity. They must not invalidate topological semantics.

### 8.4 Solver Schedule View

`solver_schedule_view` is backend-specific lowering metadata. Monte Carlo, BDD, exact enumeration, and compiler backends may require different schedules. These schedules belong to derived views or compiled artifacts, not to the semantic circuit itself.

## 9) Shape And Capability Metadata

PC compatibility requires both coarse capabilities and structured shape metadata.

The current `shape_summary` is:

```cpp
struct shape_summary final {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    bool acyclic = true;
    bool has_csr_edge_view = false;
    bool has_unsupported_probability_sources = false;
    std::vector<op_code> op_codes_present;
    std::vector<probability_source_kind> probability_source_kinds_present;
    std::vector<observable_role> observable_roles_present;
};
```

The current descriptor is:

```cpp
ir::descriptor{
    .provider = api::provider_id("fuzzybits"),
    .endpoint = pc::endpoint_id(),
    .storage_type = pc::storage_type_id(),
    .form = ir::forms::pc(),
    .name = "probabilistic circuit",
    .capabilities = api::capability_set{
        api::capability_id("ir.probabilistic_boolean_dag"),
        api::capability_id("ir.explicit_edge_polarity"),
        api::capability_id("ir.targets"),
        api::capability_id("ir.literal_probabilities"),
    },
};
```

R2+ shape metadata should add, as needed:

- node id width;
- maximum representable node id for each physical view;
- target count;
- whether an expression evaluator is required;
- CCF declaration and expansion state;
- available physical views beyond CSR-like columns;
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

## 10) Expression DAG RFC Gate

The expression DAG must not be frozen until the evaluator semantics are specified.

The RFC must define:

- whether an expression evaluates to probability, rate, time-dependent probability, distribution over probabilities, symbolic parameter, or another typed quantity;
- evaluation context fields: mission time, parameter bindings, epistemic sample index, RNG stream, unit system, and backend options;
- domain validation for probability intervals, non-negative rates, positive scales, finite arithmetic, and invalid operations;
- random-variable identity, including shared deviates, independent deviates, and correlated deviates;
- deterministic expression identity for common subexpression and diagnostics;
- external function policy;
- unsupported external expression representation;
- serialization and round-trip rules;
- backend compatibility diagnostics.

Only after this RFC is accepted may a `basic_event` carry a mandatory expression reference. Until then, basic events refer to the narrower probability-source table.

## 11) CCF RFC Gate

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

## 12) Transform API

Transforms must carry reports and account for large graphs.

The transform result shape should be:

```cpp
struct transform_result {
    circuit output;
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

## 13) External Risk-Model Ingestion Requirements

External risk-model-to-PC conversion must be explicit about supported and unsupported semantics.

The initial conversion subset should include:

- basic events with literal probability expressions;
- basic events with parameter probability expressions when the parameter domain is a probability;
- house events that resolve to constants or to identifiable house-event leaves if identity is required;
- Boolean gates mapping to the v1 opcode set;
- cardinality gates mapping to `op_code::cardinality` with `cardinality_low()` and `cardinality_high()` payloads;
- top-event or query observables mapping to PC targets.

Unsupported or deferred external-model features must not be silently erased:

- non-literal expression trees before the expression RFC is implemented;
- random deviates before evaluator randomness is specified;
- external functions before external function policy is specified;
- CCF groups before the CCF RFC is implemented;
- event-tree sequence projection unless represented by typed projection data;
- unsupported connectives or malformed cardinality bounds.

The converter must return diagnostics for unsupported features and must mark the resulting PC descriptor or shape summary with any unsupported placeholders. Solver/compiler backends that require fully evaluable probabilities must reject those placeholders during planning.

## 14) Serialization And External Format Mapping

Serialization is not the IR contract.

External formats are:

- compatibility backends;
- candidate wire formats;
- possible sources of zero-copy physical views after separate layout RFCs.

External formats are not:

- the canonical semantic identity of `pc`;
- proof that `pc` is lossless-isomorphic to any particular file schema;
- the required in-memory layout for `pc::circuit`.

Each format mapping spec must classify every feature as:

- lossless import/export;
- lossless with index or layout translation;
- lossy but diagnosable;
- unsupported.

Known mapping hazards that must be addressed:

- operators outside the PC Boolean opcode set;
- input polarity order and role preservation;
- multi-output or structural operators;
- execution order versus PC semantic ids;
- expression/probability side tables;
- CCF declarations and expansion provenance;
- target and projection metadata.

Zero-copy mmap construction is a physical-view optimization. It requires a separate contract for lifetime, alignment, endianness, edge encoding, string storage, and unsupported schema features.

## 15) Subcircuits And Modules

Subcircuits/modules are deferred until instance semantics are specified.

The module RFC must define:

- whether ids are global or per subcircuit;
- whether a subcircuit is a definition, an instance, a partition, or an external-format compatibility shape;
- input and output port binding;
- edge polarity across ports;
- multi-output semantics;
- target references across module boundaries;
- CCF references across module boundaries;
- flattening semantics;
- flattening provenance maps;
- view/schedule generation for module-aware and flattened consumers.

R1 must remain a flat DAG with multi-target observables. Module detection and module lowering are transforms after the above semantics exist.

## 16) Roadmap Gates

The roadmap gates are requirements gates, not permission to add placeholders that look stable.

### R0: Architecture Boundary

- `pc::circuit` is public `src/ir/pc` storage.
- Public orchestration uses `ir::handle`.
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

### R3: External Risk-Model Ingestion

- Convert the supported external risk-model subset.
- Emit diagnostics and unsupported placeholders for deferred semantics.

### R4: Expression RFC

- Freeze evaluator context, domains, units, and random-variable identity.
- Add expression DAG only after the RFC.

### R5: CCF RFC

- Add CCF declarations, expansion transform, provenance, and projection semantics.

### R6: Transform API

- Add transform reports and memory budgets.
- Replace ad hoc preprocessing only after semantic parity and projection behavior are proven.

### R7: Serialization

- Specify external serializers as mappings.
- Add import/export after semantic PC is stable.

### R8: Modules

- Add subcircuits/modules only after instance semantics, port binding, id scope, CCF, targets, and flattening are fixed.

## 17) Explicit Non-Commitments

The PC v1 requirements do not commit to:

- signed negative ids for complement edges;
- mandatory kind partitioning;
- any file format as the canonical on-disk format;
- zero-copy mmap PC construction;
- expression DAG opcode closure;
- CCF expansion as a boolean property;
- subcircuits/modules in R1;
- execution `sample` or `tally` as PC opcodes;
- solver convergence flags inside IR targets;
- pure-copy transform implementations for large graphs.

Any of these may become valid implementation choices later, but none may become v1 ABI without a semantic RFC and compatibility story.
