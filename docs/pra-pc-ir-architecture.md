# PRA And PC IR Architecture

Status: design analysis  
Scope: `ir::pc`, a future `ir::pra`, PRA-to-PC lowering, target projection, and result serialization  
Context: `src/ir/pc/IR-PC-REQUIREMENTS.md`, source architecture contracts, OpenPSA/FTP backend refactors

Note: the referenced `out-of-scope/KUREAS` schema content was not visible in this workspace snapshot. The SAPHIRE MAR-D discussion below is therefore based on the expected shape of MAR-D/PRA project data: fault trees, event trees, sequences, end states, basic events, gates, common-cause groups, parameters, attributes, and external project metadata.

## Executive Position

The current `ir::pc` requirements are directionally correct and should not be weakened by adding domain metadata columns to `pc::circuit`. PC is the computational target IR: a compact probabilistic Boolean DAG with local node ids, local target ids, probability sources, and physical traversal views. It should not become the canonical owner of fault trees, event trees, sequence/end-state structure, CCF declarations, SAPHIRE/OpenPSA attributes, report layout, or source-format provenance.

The missing piece is a first-class PRA domain IR and a typed lowering/projection artifact between PRA and PC:

```text
OpenPSA / SAPHIRE MAR-D / FTP / other PRA formats
  -> io reader
  -> ir::pra::model
  -> compiler::pra_to_pc
  -> compiler::artifact(pra_pc_lowering)
       - ir::pra::model handle
       - ir::pc::circuit handle or owned circuit
       - typed projection/lowering maps
       - transform report and diagnostics
  -> PC-oriented solver/compiler
  -> solver results keyed to computational targets
  -> projection stage
  -> PRA-domain results keyed to fault trees, sequences, end states, events, etc.
  -> io writer
```

This gives us a clean path that satisfies the PC contract, avoids one-off bespoke IRs per PRA input format, and avoids a full node-for-node reconciliation burden. We preserve only the sparse semantic anchors that projection and attribution actually require.

## Is The Existing PC Plan Sufficient?

It is sufficient as a boundary contract. It explicitly says:

- PC targets are observables, not domain payloads.
- PC node ids are storage ids, not stable semantic identity.
- domain structures such as fault trees, event trees, sequences, end states, initiating events, and report metadata stay outside the Boolean/probability core.
- transforms that affect identity must return provenance maps.
- event-tree sequence projection must be represented by typed projection data.
- CCF expansion requires provenance, not a boolean property bit.

It is not sufficient as a complete architecture because it does not yet define the positive owner of that typed projection data. If the project stops at `pc::target_view{target_id, node, role}`, then OpenPSA/SAPHIRE result serialization will either lose information or force solvers and writers to infer domain semantics from local PC ids. That is exactly the failure mode the PC requirements are trying to avoid.

The implementation path is therefore:

1. keep `ir::pc` narrow;
2. add one canonical `ir::pra` domain IR;
3. add a PRA-to-PC lowering artifact with explicit sparse maps;
4. make pipeline/result projection consume those maps.

## Design Goals

The design should optimize for these properties:

- one canonical PRA model, not a bespoke OpenPSA model, SAPHIRE model, FTP model, and future MAR-D model;
- PC remains a compact, high-performance computational IR;
- domain identity is stable across parsing, lowering, solving, projection, and serialization;
- lossy import, unsupported semantics, and expansion choices are diagnostics, not hidden behavior;
- result projection is typed and queryable before execution;
- maps are sparse and semantic-anchor based, not mandatory all-node provenance;
- large models can be lowered and transformed without prohibitive copy amplification.

## Core Separation

### `ir::pc`

`ir::pc` owns only computational Boolean/probability semantics:

- dense local node ids;
- explicit edge polarity;
- probability source table;
- local PC target table;
- shape metadata and physical views;
- transform reports for computational graph rewrites.

It may retain generic computational facts such as probability source parameter names because those are part of the evaluable probability contract. It should not retain external model topology, report labels, event-tree paths, CCF group definitions, file source locations, or arbitrary attributes.

### `ir::pra`

`ir::pra` should own the canonical PRA domain model:

- project metadata and source provenance;
- elements with stable domain ids and display names;
- fault trees, components, gates, basic events, house events;
- event trees, functional events, branches, sequences, end states, initiating events;
- parameters and typed expressions;
- common-cause group declarations and member relationships;
- substitution/rule/alignment/change-set-like structures where they affect model semantics;
- source-format attributes as structured extension records;
- domain query targets.

This is not a "bespoke IR for OpenPSA" or a "bespoke IR for SAPHIRE". It is a single PRA semantic model. OpenPSA MEF, SAPHIRE MAR-D JSON, FTP, and other formats become IO adapters into this model. Format-specific details live either in source provenance/extension attributes or in backend-private parser models until they are normalized.

### `pra_pc_lowering`

The lowering artifact connects the two IRs:

```cpp
struct pra_pc_lowering {
    ir::handle pra_model;
    ir::handle pc_model;
    pra::pc_projection projection;
    transform_report report;
};
```

The exact storage type can live under a compiler backend or under `ir::pra` if it is treated as a reusable IR conversion product. The public orchestration type should be `compiler::artifact` or another type-erased role handle, not direct concrete storage exposure.

The important design point is that a PC solver can remain PC-oriented while the pipeline retains the projection context needed to express results in PRA terms.

## PRA Model Shape

The `ir::pra::model` should use arena-style dense ids, column-friendly vectors, and typed references. It should not expose a mutable object graph or provider-native parser objects.

Suggested high-level structure:

```cpp
namespace fuzzybits::ir::pra {

struct model {
    project_metadata project;

    std::vector<element_metadata> metadata;
    std::vector<attribute> attributes;
    std::vector<source_location> sources;

    std::vector<basic_event> basic_events;
    std::vector<house_event> house_events;
    std::vector<gate> gates;
    std::vector<fault_tree> fault_trees;
    std::vector<component> components;

    std::vector<functional_event> functional_events;
    std::vector<branch> branches;
    std::vector<sequence> sequences;
    std::vector<end_state> end_states;
    std::vector<event_tree> event_trees;
    std::vector<initiating_event> initiating_events;

    std::vector<parameter> parameters;
    std::vector<expression_node> expressions;
    std::vector<formula_node> formulas;

    std::vector<ccf_group> ccf_groups;
    std::vector<ccf_member> ccf_members;

    std::vector<domain_target> targets;
};

}
```

The actual implementation should split headers by concern, but conceptually the model needs these categories.

### Metadata

Every domain entity that can appear in diagnostics, reports, or serialized output should have an `element_metadata` entry:

```cpp
struct element_metadata {
    element_id id;
    element_kind kind;
    std::string stable_name;
    std::string qualified_name;
    std::string label;
    attribute_range attributes;
    source_location_range sources;
};
```

`stable_name` and `qualified_name` are domain identities, not PC node names. For OpenPSA this may correspond to scoped names. For SAPHIRE MAR-D this may correspond to project-defined IDs. For FTP this may be a minimal generated PRA subset. For AIGER/BLIF, there may be no PRA model at all unless the user requests a generic symbol projection sidecar.

### Boolean Formulas

Fault-tree gates and event-tree branch conditions should reference a PRA formula arena. Formula nodes should reference domain event ids, not PC node ids:

```cpp
struct formula_node {
    formula_kind kind;
    formula_operand_range operands;
    cardinality_bounds cardinality;
};

struct formula_operand {
    formula_operand_kind kind; // event, formula, constant
    event_ref event;
    formula_id formula;
    bool complement;
    bool constant_value;
};
```

This gives the lowering pass freedom to share subgraphs, normalize, decompose operators, or expand CCF without losing the domain expression that generated the computational graph.

### Probability Expressions

`ir::pc` currently has literal, parameter, and unsupported probability sources. That is enough for R1 PC, but not enough for OpenPSA and MAR-D as source domains.

`ir::pra` should own typed probability/rate/expression semantics:

- literal probability;
- named parameter;
- rate with mission-time conversion;
- time-dependent probability expressions;
- distributions/deviates for uncertainty;
- external functions as explicit unsupported or policy-gated expressions;
- unit and domain metadata.

Lowering to PC may produce:

- literal PC probability sources when fully evaluable under the current request;
- parameter PC probability sources when symbolic parameters are allowed;
- unsupported PC probability sources plus diagnostics when a solver cannot evaluate the source.

This keeps evaluator policy out of PC while still making compatibility checkable.

### CCF

CCF belongs first in `ir::pra`, not PC.

The PRA model should represent unexpanded declarations:

- group id and model kind: beta, alpha, MGL, phi, custom;
- member basic events;
- factor expressions;
- distribution/exposure metadata;
- source format provenance.

Expansion is a lowering transform:

```text
pra::model
  -> ccf_expansion transform
  -> pra::model or pra_to_pc lowering with generated events/nodes
  -> transform report
```

The report must include original-to-generated and generated-to-original mappings. This is necessary for importance and product results: a generated common-shock event is computationally real, but a user may need attribution rolled up to the original CCF group, original member events, or both.

## Projection Model

`observable_role::projection_anchor` is useful but insufficient by itself. It says why a PC target exists; it does not say what domain entity it represents or how to aggregate results.

Add a typed projection model owned by the lowering artifact:

```cpp
namespace fuzzybits::ir::pra {

struct pc_projection {
    std::vector<target_binding> target_bindings;
    std::vector<event_binding> event_bindings;
    std::vector<generated_binding> generated_bindings;
    std::vector<aggregation_binding> aggregation_bindings;
};

struct target_binding {
    domain_target_id domain_target;
    pc::target_id pc_target;
    projection_kind kind;
    domain_ref subject;
};

struct event_binding {
    domain_event_ref domain_event;
    pc::bool_node_id pc_node;
    event_binding_kind kind; // basic event, house event, generated CCF event, gate anchor
};

struct generated_binding {
    generated_entity_ref generated;
    domain_ref origin;
    transform_id transform;
};

}
```

This table should be sparse. It should contain only anchors needed for:

- target probability projection;
- cut set/product attribution;
- importance attribution;
- sequence/end-state projection;
- CCF roll-up;
- diagnostics and traceability.

It should not attempt to map every intermediate PC node to a domain entity. Most intermediate nodes are compiler artifacts: decomposed cardinality gates, normalized negations, shared subexpressions, BDD-derived variables, generated CCF shocks, or optimization results. Requiring total correspondence would constrain transforms and damage performance.

## Domain Targets

PRA targets should be first-class domain queries:

```cpp
enum class domain_target_kind : std::uint8_t {
    fault_tree_top,
    gate,
    basic_event,
    sequence,
    end_state,
    initiating_event,
    event_tree_path,
    custom_formula,
};

struct domain_target {
    domain_target_id id;
    domain_target_kind kind;
    domain_ref subject;
    formula_id condition;
    observable_role role;
};
```

Lowering creates one or more PC targets for each requested domain target. Usually this is one-to-one:

```text
fault tree top event -> PC target
```

Some domain targets may be one-to-many or require aggregation:

```text
end state -> set of sequence/path PC targets -> aggregation binding
event tree path probability -> conjunction of branch conditions -> generated PC target
CCF group attribution -> generated basic events + original member/group roll-up
```

The aggregation binding is the place to represent those relationships, not `pc::target_view`.

## Result Projection

Solvers should not know about OpenPSA, SAPHIRE, or PRA report structure. They should compute result values over their computational input.

The pipeline should retain a projection context:

```text
solver result keyed to PC target/event/product
  + pra_pc_lowering.projection
  -> projected result keyed to domain_target/domain_entity
```

This likely requires changes to current result contracts:

- `solver::request` should eventually carry requested target ids, not just result kinds.
- probability results should preserve the computational target identity without lossy stringification of raw local ids.
- product and importance results need event identity references that can map through `event_binding`.
- pipeline result values should distinguish computational result records from projected result records.

An acceptable intermediate implementation is:

1. PC solvers return PC-local target ids.
2. Pipeline detects a `pra_pc_lowering` artifact and applies `pc_projection`.
3. Result writers receive projected records plus the original PRA model handle.

The long-term API should make this explicit rather than relying on ad hoc result-id strings.

## Avoiding Extreme Reconciliation Costs

The key performance rule is sparse provenance.

Do not require:

- every PC node to have a PRA entity;
- every PRA formula node to survive as a PC node;
- every transform to materialize full old-to-new maps for all nodes;
- PC node ids to be stable across compaction, scheduling, or decomposition;
- report serialization to walk PC internals for domain labels.

Do require:

- every requested domain target has a binding to the PC target(s) that implement it;
- every domain basic event that can appear in product or importance results has an event binding;
- every generated CCF event has provenance to its group/member/origin;
- every transform that changes anchor identity updates or invalidates projection maps;
- every unsupported or lossy mapping produces diagnostics.

For large models, maps should use compact representations:

- sorted vectors for sparse `domain_id -> pc_id` and `pc_id -> domain_id` lookups;
- range/permutation maps for transforms that preserve order with predictable remapping;
- generated-node ranges for decompositions;
- optional hash indexes built lazily outside hot solver loops;
- no owning strings in hot PC columns.

## Where Names Belong

There are three classes of names:

1. computational parameter names;
2. generic source symbols;
3. PRA domain identities.

Computational parameter names can remain in PC probability sources because solvers and evaluators need them.

Generic source symbols, such as AIGER input/output symbols or FTP names without rich PRA semantics, should live in a lightweight symbol/provenance sidecar if they need to round-trip. That sidecar should not be part of the PC hot columns. It can be a future `ir::annotation` or `pc::symbol_map` view keyed by PC ids.

PRA domain identities belong in `ir::pra`. OpenPSA/SAPHIRE labels, attributes, qualified names, event-tree branch names, end-state names, and report metadata should never be stored as PC node payloads.

## Endpoint Implications

Rich PRA readers should not lower directly to raw PC as their only output.

Recommended endpoint split:

- `io/backend/fuzzybits/openpsa` reads OpenPSA input XML into `ir::pra::model`.
- future `io/backend/fuzzybits/saphire_mard` reads MAR-D JSON into `ir::pra::model`.
- `io/backend/fuzzybits/ftp` can either read directly to PC for the historical simple subset or read to a minimal PRA model if target naming/projection must round-trip.
- `io/backend/fuzzybits/aiger` should remain PC-oriented; AIGER is a circuit format, not a PRA project format.
- `compiler/backend/fuzzybits/pra_to_pc` lowers `ir::pra::model` to `pra_pc_lowering`.
- OpenPSA/SAPHIRE writers that serialize model input should consume `ir::pra::model`, not reconstruct a PRA model from raw PC.
- result/report writers should consume projected result values plus the PRA model/projection context.

This separation also fixes the current OpenPSA writer problem: a writer from raw PC to OpenPSA MEF cannot be meaningfully lossless for PRA project XML. It can only emit a synthetic fault tree unless given PRA domain data.

## Pipeline Implications

The pipeline should plan around domain and computational stages:

```text
read OpenPSA
  -> ir::pra::model
compile pra_to_pc
  -> pra_pc_lowering artifact
solve PC probability/products/importance
  -> computational results
project results
  -> PRA-domain results
write report/input/export
```

Planning descriptors need enough information to know:

- a reader produces `ir.pra.model`;
- `pra_to_pc` accepts `ir.pra.model` and produces `ir.probabilistic_boolean_dag` plus projection capabilities;
- a solver accepts raw PC or a PC-bearing artifact;
- a writer requires projected PRA results, raw PRA model, raw PC, or some combination.

Capabilities should remain coarse, but constraints need structured fields. Examples:

- supports event-tree sequence projection;
- requires fully evaluated literal probabilities;
- supports parameter probabilities;
- supports unexpanded CCF declarations;
- requires CCF expansion;
- emits CCF attribution roll-up;
- preserves source attributes;
- preserves external ids.

Parameterized constraints should not be encoded as string capability names.

## Proposed Module Layout

Suggested target/module structure:

```text
src/ir/pra/
  descriptor.h
  id.h
  metadata.h
  expression.h
  formula.h
  model.h
  projection.h
  handle.h
  pra.h

src/compiler/backend/fuzzybits/pra_to_pc/
  descriptor.h
  compile.cpp
  detail/lowering.cpp
  detail/ccf_expansion.cpp
  detail/projection_builder.cpp

src/io/backend/fuzzybits/openpsa/
  descriptor.h
  read.cpp          # OpenPSA -> ir::pra::model
  write.cpp         # ir::pra::model -> OpenPSA, where supported
  detail/...        # parser/private MEF mapping

src/io/backend/fuzzybits/saphire_mard/
  descriptor.h
  read.cpp          # MAR-D JSON -> ir::pra::model
  write.cpp         # optional
  detail/...
```

Whether `pra_pc_lowering` lives in `ir/pra` or the compiler backend is an ownership decision:

- put it under `ir/pra` if multiple compilers/solvers/writers need the projection storage as a stable public role contract;
- keep it under the compiler backend if it is initially a private artifact with only pipeline-mediated access.

Given the architecture goals, I would start with a small public `ir::pra::projection` value contract and allow compiler artifacts to own it. That makes projection inspectable without exposing lowering internals.

## Minimal R1 Implementation Plan

### Step 1: Add `ir::pra` Identity And Model Skeleton

Implement dense ids, metadata, basic events, gates, fault trees, formulas, parameters, and domain targets. Do not implement full event-tree or CCF semantics in the first slice unless needed by tests.

The first useful subset:

- basic events with literal/parameter/unsupported probability expressions;
- house events;
- gates with Boolean/cardinality formulas;
- fault trees with top-event targets;
- stable metadata and source ids;
- domain target table.

### Step 2: Make OpenPSA Import Produce PRA

Change OpenPSA reader output from "MEF parsed then PC-not-implemented" to a PRA model handle.

Keep the MEF parser private. Convert MEF structures into `ir::pra` at the IO boundary. Unsupported OpenPSA features should survive as diagnostics and explicit unsupported records where possible.

### Step 3: Add PRA-To-PC Lowering

Lower the subset above to PC:

- each basic event becomes a PC basic-event node;
- each house event becomes a constant or house-event node depending on projection needs;
- each gate/formula becomes PC Boolean/cardinality nodes;
- each requested domain target becomes a PC target;
- emit `target_binding` and `event_binding`.

Return a compiler artifact containing PC plus projection.

### Step 4: Update Solver/Pipeline Result Identity

Stop treating `std::to_string(pc_target.raw())` as a stable public target id. Keep PC-local target identity until projection, then emit domain target ids.

The minimal bridge can be pipeline-owned:

- solver emits PC target result;
- pipeline looks up `target_binding`;
- pipeline emits domain probability result.

### Step 5: Add Event-Tree Projection

Represent event-tree paths/sequences/end states in PRA first, then lower requested sequence/end-state targets into generated PC target formulas. Add aggregation bindings for many-path end states.

This should not require event-tree-specific solvers.

### Step 6: Add CCF Expansion With Provenance

Implement CCF declaration import before expansion. Add an explicit lowering option:

- reject unexpanded CCF for solvers that cannot consume it;
- expand CCF when requested and supported;
- record generated event/node provenance;
- project results back to original group/member entities.

## Why This Satisfies The User-Facing Requirements

### No Format-Bespoke PRA IRs

There is one PRA IR. OpenPSA, SAPHIRE MAR-D, FTP, and future formats map into it. Format-specific details are either normalized into core PRA semantics or retained as typed extension/source metadata. We do not create one IR per file format.

### No PRA Payload In PC

PC remains small and solver-friendly. Domain names and project metadata do not enter PC hot traversal columns. Solvers do not parse fault trees, event trees, CCF groups, or report attributes.

### No Extreme PC/PRA Reconciliation

Projection maps are sparse and anchor-oriented. We map targets, leaf events, generated CCF events, and selected gate anchors. We do not force total PC-node/domain-entity correspondence.

### Event Trees Stay Projection, Not Solver Families

Sequence and end-state probabilities become domain targets lowered to PC targets and aggregation bindings. Solvers compute probability/product/importance over PC; projection maps results back to sequences/end states.

### MAR-D Has A Path

MAR-D project JSON should map to `ir::pra` entities and attributes. The computational subset lowers to PC. MAR-D-specific fields that affect semantics become typed PRA fields; report/UI-only fields become source/extension metadata; unsupported fields produce diagnostics.

## Main Risks

The largest risk is under-specifying `ir::pra` expressions. If we import rate/time/deviate/external-function semantics as strings, later solvers will either reject too much or silently evaluate incorrectly. The expression model should be typed early, even if many variants lower to unsupported PC probability sources initially.

The second risk is result identity. Current solver results are too target-light for projected workflows. Fixing this after many solvers exist will be expensive. The pipeline should introduce requested targets and projection-aware result records before product/importance solvers grow.

The third risk is making `pra_pc_lowering` private for too long. Writers and projection stages need a stable way to inspect target/event bindings. The projection value contract should become public early, even if the lowering algorithm remains private.

The fourth risk is treating CCF expansion as preprocessing without provenance. CCF affects products, importance, attribution, and serialization. It must be a transform with explicit reports from the first implementation.

## Recommendation

Proceed with a canonical `ir::pra` and a `pra_to_pc` compiler artifact. Do not add domain metadata to `ir::pc::circuit`. Do not make OpenPSA/SAPHIRE writers reconstruct domain XML from raw PC. Do not require complete PC-node/domain-entity mapping.

The durable architecture is:

```text
PRA owns domain semantics.
PC owns computational Boolean/probability semantics.
The lowering artifact owns projection and provenance.
Pipeline owns planning and result projection.
IO owns external serialization.
Solvers own algorithms over declared computational contracts.
```

That design can support OpenPSA, SAPHIRE MAR-D, FTP subsets, direct circuit formats, CCF, event-tree projection, and future result families without turning PC into a PRA object graph or coupling solvers to external project schemas.
