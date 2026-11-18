# `fuzzybits::ir::pra` Requirements

Status: initial domain contract  
Scope: `src/ir/pra`, PRA IO adapters, PRA-to-PC lowering, projection, and PRA-domain result serialization

## Boundary

`ir::pra::model` is the canonical in-memory PRA domain model. It is not an OpenPSA MEF object model, a SAPHIRE MAR-D
object model, a KUREAS editor state, or a PC graph with names attached. External schema details either normalize into
typed PRA semantics or remain source-provenance/attribute records.

The PRA IR must not parse files, evaluate probability expressions, expand CCF groups, run solvers, or serialize reports.
Those actions belong to IO, compiler, solver, pipeline, and writer endpoints.

## Coverage Targets

The public model must be able to name and relate these information classes:

1. Project metadata, source artifacts, schema/tool versions, stable names, labels, descriptions, and external attributes.
2. Hazards, hazard groups, initiating events, initiator frequencies, and event-tree linkage.
3. Event trees, functional events, branch states, branch conditions, sequence paths, end states, transfers, and path
   frequencies.
4. Fault trees, gates, basic events, house events, gate inputs, transfers, top events, and component/failure-mode links.
5. Systems analysis concepts such as systems, components, functions, FMEA-style failure modes, and criteria formulas.
6. Parameters and expressions for probabilities, frequencies, rates, mission-time conversion, distributions, deviates,
   and unsupported external functions.
7. Unexpanded common-cause groups, members, factors, expansion state, and generated-event provenance anchors.
8. Post-processing rules including substitutions, recovery/deletion/addition rules, alignments, phases, cut-set filters,
   truncation settings, roll-ups, and imported product caches.
9. Domain targets and sparse projection bindings to lowered PC targets/events.

## Evidence From Formats

OpenPSA MEF provides explicit fault-tree, event-tree, model-data, CCF, substitution, rule, alignment, phase, consequence,
and attribute layers.

SAPHIRE MAR-D provides equivalent semantics through separate decks: `.FTL` fault-tree algebra, `.ETL` event-tree logic
matrices and node substitutions, `.BEI` basic-event quantification, `.BED` event descriptions, `.GTD` gate descriptions,
`.MTD` initiating-event taxonomy, `.FTA` analysis settings, and `.FTC` / `.SQC` / `.ESC` cut-set caches.

KUREAS `KNOW_FTL` contributes a JSON fault-tree editor representation. `KNOW_SYS` and hazard-analysis files add systems,
components, functions, failure modes, FMEA/reliability-library links, hazard hierarchy, and report/UI metadata. The core
IR should normalize only stable PRA semantics; editor state and layout remain attributes or extensions.

## Non-Commitments

The first `ir::pra` contract does not commit to:

- an evaluator for all probability/expression nodes;
- a total OpenPSA or SAPHIRE round-trip guarantee;
- a mandatory CCF expansion algorithm;
- a node-for-node PRA-to-PC map;
- canonical imported cut sets as solver truth;
- UI/report layout semantics as core PRA math.

When a reader encounters unsupported or ambiguous semantics, it should preserve the source identity and produce explicit
unsupported expression/formula/rule records plus diagnostics.

## Projection Requirement

PRA-to-PC lowering must produce sparse anchors, not a full reconciliation map. Required anchors are requested domain
targets, domain basic events needed for products/importance, generated CCF events, and aggregation bindings for
sequences/end states/hazard or system roll-ups. Intermediate PC nodes remain compiler artifacts unless they are needed
for projection, diagnostics, or provenance.

## Deferred Implementation Gates

The current code defines the public PRA value contract only. The following executable pieces remain intentionally
unimplemented until their input/output semantics and compatibility obligations are precise enough to avoid baking
schema-specific behavior into `ir::pra`.

### Model Construction And Validation

Still needed:

- a builder or construction API that assigns dense ids, fills `table_range` slices, validates references, and emits
  diagnostics;
- invariant checks for missing metadata, invalid ranges, dangling references, duplicate stable names where uniqueness is
  required, malformed formula/expression operands, and invalid target subjects.

Implementation gate:

- implement only after the first reader or compiler endpoint has concrete construction needs that cannot be met by plain
  value initialization;
- define which invariants are hard IR validity requirements and which are endpoint-specific compatibility constraints;
- define diagnostics ids for structural errors before adding silent normalization or repair behavior.

### External Readers

Still needed:

- OpenPSA MEF input reader producing `ir::pra::model`;
- SAPHIRE MAR-D reader mapping `.FTL`, `.ETL`, `.BEI`, `.BED`, `.GTD`, `.MTD`, `.FTA`, `.FTC`, `.SQC`, and `.ESC` records;
- KUREAS native readers for `KNOW_FTL`, `KNOW_SYS`, and hazard-analysis files where they carry stable PRA semantics;
- FTP/simple fault-tree adapter deciding whether to produce minimal PRA, PC directly, or both.

Implementation gate:

- each reader must have a feature mapping table classifying fields as lossless core semantics, preserved extension
  attributes, diagnosable loss, or unsupported;
- parser-native structures must stay private to the IO backend;
- unsupported or ambiguous source constructs must produce diagnostics and explicit unsupported records, not guessed
  formulas or probabilities;
- at least one corpus-backed smoke/round-trip expectation must exist for every source format introduced.

### External Writers

Still needed:

- PRA-model writers for OpenPSA MEF and, later, SAPHIRE/KUREAS formats where round-trip semantics are meaningful;
- report/result writers that consume projected PRA-domain results plus the original model/projection context.

Implementation gate:

- implement only after the reader-side mapping table identifies which fields are round-trippable and which are
  extension-only;
- writers must reject raw PC handles when PRA domain data is required;
- writers must distinguish model serialization from result/report serialization;
- lossy export must be declared through diagnostics, not hidden by synthetic defaults.

### Probability And Expression Evaluation

Still needed:

- an evaluator contract for probability, frequency, rate, duration, distribution, and deviate expressions;
- evaluation context fields such as mission time, phase/alignment, units, parameter bindings, epistemic sample index, RNG
  stream identity, and external-function policy;
- lowering policy from PRA expressions to PC literal, parameter, or unsupported probability sources.

Implementation gate:

- implement only after expression domains, units, validation rules, random-variable identity, correlation semantics, and
  external-function policy are specified;
- solvers must declare which expression classes they can consume directly or through lowering;
- any expression that cannot be evaluated under a request must remain visible through diagnostics and unsupported PC/PRA
  placeholders.

### PRA-To-PC Lowering

Still needed:

- a compiler endpoint accepting `ir::pra::model` and producing a PC-bearing artifact plus `pc_projection`;
- Boolean lowering for fault-tree gates, formulas, transfers, house events, event-tree branch/path targets, and selected
  post-processing rules;
- target and event binding generation for probability, products, importance, and sequence/end-state projection.

Implementation gate:

- implement only after the supported formula/operator subset is enumerated and mapped to PC opcodes;
- define lowering options for house-event preservation, transfer flattening, generated target creation, unsupported
  probability handling, and CCF expansion policy;
- every requested domain target must either receive a binding to PC target(s) or a diagnostic explaining why it cannot be
  lowered;
- every transform that changes observable identity must update projection bindings or explicitly invalidate them through
  a transform report.

### Event-Tree Semantics And Projection

Still needed:

- precise branch-state semantics for success/failure/transfer/bypass/split states;
- conversion of event-tree paths and sequences into domain targets and generated PC formulas;
- aggregation from sequence/path targets to end states, hazards, initiators, systems, or consequence groups.

Implementation gate:

- implement only after SAPHIRE ETL logic-grid encodings and OpenPSA fork/path semantics are normalized to the same path
  model;
- define whether branch split fractions are probabilities, frequencies, weights, or conditional modifiers in each
  context;
- aggregation rules must state whether they are exact disjoint sums, rare-event approximations, weighted sums, or
  format-specific report roll-ups.

### Common-Cause Failure Support

Still needed:

- import of unexpanded beta-factor, alpha-factor, MGL, phi-factor, shock, and custom CCF declarations;
- CCF expansion transform producing generated events/nodes and original-to-generated/generated-to-original provenance;
- attribution policy for probability, product, and importance results involving generated common-shock events.

Implementation gate:

- implement only after parameterization semantics, member identity, adjusted independent probabilities, and generated
  event semantics are specified;
- expansion must be a compiler transform with a transform report, never a boolean flag on the model;
- result projection must define roll-up behavior to CCF groups, original member events, generated events, or combinations
  of those views.

### Post-Processing Rules

Still needed:

- semantics for substitutions, recovery/deletion/addition rules, alignments, phases, truncation/filter settings,
  cut-set-processing directives, and consequence collection rules;
- decision about which rules are model semantics, which are solver/compiler options, and which are report-only metadata.

Implementation gate:

- implement only after each rule kind has a declared subject, evaluation phase, ordering rule, and effect on targets,
  formulas, products, or projections;
- rules that alter Boolean/probability semantics must produce transform reports and provenance;
- report-only rules must remain attributes or extension records unless a writer explicitly consumes them.

### Imported Product/Cut-Set Caches

Still needed:

- representation and validation of imported fault-tree, sequence, end-state, and consequence cut-set decks;
- compatibility rules for when cached products may be trusted, recomputed, filtered, or rejected.

Implementation gate:

- implement only after the result/product identity model can distinguish solver-produced products from imported caches;
- cached products must carry source provenance, truncation/filter settings, and target identity;
- no solver may treat imported products as canonical truth unless its descriptor declares that behavior and validates the
  cache against the requested model/options.

### Result Projection

Still needed:

- pipeline or compiler-stage projection from PC-local solver records to PRA-domain result records;
- probability, product, importance, uncertainty, and roll-up result identities keyed by `domain_target_id` or
  `domain_ref`;
- diagnostics when solver outputs cannot be projected because bindings were missing or invalidated.

Implementation gate:

- implement only after solver results preserve PC target/event identity without lossy stringification;
- `pc_projection` must be carried alongside the PC artifact through pipeline planning and execution;
- projection must be deterministic for one-to-one bindings and explicitly declare aggregation semantics for many-to-one
  or one-to-many bindings.

### Domain-Specific Extension Growth

Still needed:

- a policy for promoting extension attributes into first-class `ir::pra` fields as new domains mature;
- compatibility tests proving promotion does not break existing readers, writers, or projection consumers.

Implementation gate:

- promote a field only when at least two endpoints or one solver/compiler workflow need typed access;
- maintain source provenance for promoted fields if external round-trip or diagnostics depend on it;
- avoid adding schema-specific columns for UI/report layout unless they affect PRA semantics or a stable writer contract.
