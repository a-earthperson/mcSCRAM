The provided code is best understood as a **PDAG pre-compiler** for downstream knowledge-compilation backends. It does **not** itself construct a BDD/ZBDD/MOCUS artifact in the shown code; instead, it rewrites a Boolean/fault-tree PDAG into a smaller, more backend-suitable Boolean DAG, annotates it with coherence/module/order metadata, and optionally normalizes the connective basis.

The core KC goals are:

1. **Boolean-equivalence preserving simplification**: remove constants/null gates, collapse identities, simplify duplicate/complement arguments.
2. **Connective normalization**: progressively eliminate `NOT`/`NAND`/`NOR`, optionally expand `XOR` and `K/N` threshold gates, and represent negation as signed edges/root complement where possible.
3. **Structural compression before compilation**: common-subexpression elimination, associative flattening, common-argument extraction, distributivity/absorption, and redundancy removal.
4. **Module detection / decomposition support**: identify subgraphs that can be treated as modules to control downstream compilation blow-up.
5. **Coherence and ordering metadata**: compute whether the resulting PDAG is monotone/coherent and assign a compilation/evaluation order.
6. **Backend-specific preparation**:
   - BDD: general Boolean function compilation; can tolerate noncoherence.
   - ZBDD/MOCUS: set-family / cut-set style compilation; strongly benefits from coherent monotone formulas.
   - DirectEval: configurable simplification without necessarily forcing a canonical KC-friendly normal form.

---

## 1. Representation-level semantics and invariants

The PDAG represents Boolean formulas using:

- `Gate` nodes with connective type:
  - `kAnd`, `kOr`, `kAtleast`, `kXor`, `kNot`, `kNand`, `kNor`, `kNull`.
- `Variable` nodes for basic events.
- One global `Constant` node whose positive signed index means true and negative signed index means false.
- **Signed argument indices**:
  - positive index = positive literal/subformula,
  - negative index = complemented literal/subformula.
- `Pdag::complement_`, a root-level complement bit.

A `kNull` gate is not Boolean “null” in the algebraic sense; it is an identity/wrapper gate used to encode pass-through structure, root triviality, constants, and normalized `NOT` gates before they are propagated away.

The distinction between **normal** and **coherent** is crucial:

### `normal`

In this implementation, `Pdag::normal_` means roughly:

> The graph contains only the basic `AND`/`OR` Boolean basis, modulo signed edges and root complement.

Construction sets `normal_ = false` when it sees anything other than `kAnd` or `kOr`, including `kAtleast`, `kXor`, `kNot`, `kNand`, `kNor`, `kNull`, and complex MEF connectives.

After `RunPhaseThree()`, the code explicitly sets:

```cpp
graph_->normal(true);
```

meaning the intended postcondition is: nontrivial gates are now `AND`/`OR` only.

### `coherent`

`coherent` means monotone-positive structure:

> No root complement, no negative arguments, no negating/nonmonotone gate types, and all descendants coherent.

`pdag::MarkCoherence()` computes this structurally. It marks gates incoherent if they are `XOR`, `NOR`, `NOT`, or `NAND`, or if they contain negative arguments, or if any child gate is incoherent.

After full normalization, coherence effectively means:

> The graph is a positive `AND`/`OR` DAG over positive variables.

This matters especially for ZBDD and MOCUS-style cut-set compilation.

---

## 2. `pdag_ops`: utility passes and their KC role

| Method | What it does | KC role |
|---|---:|---|
| `pdag::Transform(Pdag*, ops...)` | Variadic pass driver. Before each pass, calls mutable `graph->IsTrivial()`, which may collapse root `kNull` wrappers or terminate the pipeline. | Orchestrates preprocessing passes and short-circuits when the formula becomes trivial. Important: `IsTrivial()` is not a pure predicate. |
| `pdag::OrderArguments<T>(Gate*)` | Collects child gates or variables and sorts them by descending parent count/fanout. | Heuristic ordering: high-fanout arguments are visited earlier in topological ordering. This can affect downstream BDD/ZBDD size. |
| `pdag::TopologicalOrder(Pdag*)` | DFS postorder from root; orders children before parents. Variables are ordered when first encountered. | Produces a total compilation/evaluation schedule. Used by BDD/ZBDD/MOCUS finalization. Also used before `Atleast` expansion because that expansion chooses the highest-ordered argument. |
| `pdag::LayeredTopologicalOrder(Pdag*)` | Computes depth layers: variables/constants get order `0`; each gate gets `1 + max(child depth)`. | Levelized schedule. Not used in the shown pipelines. Could be useful for parallel or breadth-level direct evaluation, but currently unused. |
| `pdag::MarkCoherence(Pdag*)` | Recursively annotates each gate and the whole graph as coherent or incoherent. | Computes monotonicity/coherence metadata for downstream KC. Essential for deciding whether ZBDD/MOCUS cut-set semantics are directly valid. |

Important subtlety: `TopologicalOrder()` asserts that visited gates are not constant gates. Therefore it expects null/constant cleanup to have already happened, except for trivial graphs short-circuited by `Transform()`.

---

## 3. High-level preprocessing phases

The base `Preprocessor::Run()` is defined, even though `Run()` is pure virtual in the class. Derived preprocessors explicitly call `Preprocessor::Run()`.

The base pipeline is:

```cpp
RunPhaseOne();
RunPhaseTwo();
if (!graph_->normal())
    RunPhaseThree();
```

wrapped in `pdag::Transform`, so each phase may be skipped if the graph becomes trivial.

---

### Phase I: cleanup and partial negation normalization

```cpp
void Preprocessor::RunPhaseOne()
```

Purpose:

1. Remove registered `kNull` gates and constants.
2. Collapse trivial root structures if possible.
3. If the graph is not coherent, normalize explicit negative connectives:
   - `NAND` → signed edge over `AND`
   - `NOR` → signed edge over `OR`
   - `NOT` → `kNull` wrapper, later propagated
4. Leave `XOR` and `Atleast` gates intact.

Implementation path:

```cpp
if (graph_->HasNullGates())
    graph_->RemoveNullGates();

SANITY_ASSERT;

if (!graph_->coherent())
    NormalizeGates(false); // equivalent to NormalizeGates(NONE)
```

KC role:

- Converts explicit negating connectives into edge polarity.
- Simplifies the Boolean circuit without committing to full expansion.
- Prepares the graph for structural sharing/factoring passes.

Postcondition, for nontrivial graphs:

- No registered null gates or constants.
- `NOT`/`NAND`/`NOR` generally eliminated as explicit gate types.
- `XOR` and `Atleast` may remain.
- Coherence is not recomputed here; the `coherent_` flag may be stale until `MarkCoherence()`.

---

### Phase II: structural compression and Boolean factoring

```cpp
void Preprocessor::RunPhaseTwo()
```

Pipeline:

```cpp
while (ProcessMultipleDefinitions()) continue;
DetectModules();
while (CoalesceGates(false)) continue;
MergeCommonArgs();
DetectDistributivity();
DetectModules();
BooleanOptimization();
DecomposeCommonNodes();
DetectModules();
while (CoalesceGates(false)) continue;
DetectModules();
```

KC role:

This is the main graph-compression phase. It aims to reduce the size of the downstream compiled representation by reducing repeated subformulas, exposing modules, flattening associative gates, factoring common arguments, applying distributivity/absorption, and specializing repeated common nodes.

Phase II does **not** guarantee normality or coherence. It works on the existing connective set, with many passes only acting on `AND`/`OR`/`NAND`/`NOR`-like structures.

Postcondition, for nontrivial well-formed graphs:

- No constants/null gates expected.
- Modules detected/created.
- Many equivalent repeated subgraphs merged.
- Some common factors extracted.
- No guarantee of DNNF, determinism, smoothness, read-once form, canonical form, or coherence.

---

### Phase III: full connective normalization to `AND`/`OR`

```cpp
void Preprocessor::RunPhaseThree()
```

Pipeline:

```cpp
assert(!graph_->normal());

NormalizeGates(true); // ALL
graph_->normal(true);

if (graph_->IsTrivial())
    return;

RunPhaseTwo();
```

Purpose:

- Fully normalize the graph into the backend’s basic Boolean basis.
- Expand:
  - `XOR`
  - `Atleast`
  - residual `NOT`/`NAND`/`NOR`
- Then rerun Phase II because expansion creates new structure worth optimizing.

KC role:

This is the pass that makes the graph suitable for BDD/ZBDD-style compilation over a simple `AND`/`OR` basis.

Postcondition, assuming the `normal_` flag was accurate and the graph is nontrivial:

- Nontrivial gates should be `kAnd` or `kOr`.
- Negation is represented by signed edges and/or root complement.
- `graph_->normal() == true`.
- Coherence is not guaranteed.
- Complemented subgraphs may remain.

---

### Phase IV: complement propagation / De Morgan normalization

```cpp
void Preprocessor::RunPhaseFour()
```

Used by ZBDD/MOCUS and optionally by DirectEval at high levels.

Pipeline:

1. If `graph_->complement()` is true, push root complement into the root gate:
   - `OR` ↔ `AND`
   - negate all arguments
   - clear root complement
2. Recursively replace negative gate arguments using De Morgan complements.
3. Rerun Phase II.

Core method:

```cpp
PropagateComplements(graph_->root(), false, &complements);
```

KC role:

- Tries to convert complemented subgraphs into explicit complementary structures.
- Moves the graph closer to negation-normal form.
- This is especially relevant for ZBDD/MOCUS because monotone set-family compilation prefers positive structure.

Important limitation:

Phase IV does **not** guarantee final coherence. If the Boolean function genuinely depends on negated variables, the final graph remains incoherent.

Also, because Phase IV calls `RunPhaseTwo()` afterward, later Boolean optimization may reintroduce complemented gate arguments. Therefore, for noncoherent formulas, Phase IV should be viewed as an optimization/normalization attempt, not a hard NNF guarantee.

---

### Phase V: aggressive coalescing for set/cut-set style compilation

```cpp
void Preprocessor::RunPhaseFive()
```

Pipeline:

```cpp
while (CoalesceGates(true)) continue;

if (graph_->IsTrivial()) return;

RunPhaseTwo();

if (graph_->IsTrivial()) return;

while (CoalesceGates(true)) continue;

if (graph_->IsTrivial()) return;
```

KC role:

- Performs aggressive associative flattening, including flattening shared same-type gates when `common=true`.
- This can destroy some sharing but may expose flatter `AND`/`OR` structures useful for ZBDD/MOCUS cut-set compilation.

Difference from normal coalescing:

```cpp
CoalesceGates(false)
```

preserves shared subgraphs.

```cpp
CoalesceGates(true)
```

may flatten through shared subgraphs, except modules are still preserved.

---

## 4. Low-level method map: how each method contributes

### 4.1 Orchestration and backend entry points

| Method | Role |
|---|---|
| `Preprocessor::Preprocessor(Pdag*)` | Stores graph pointer. |
| `Preprocessor::Preprocessor(Pdag*, optional<Settings>)` | Stores graph pointer and DirectEval-style settings. |
| `Preprocessor::operator()()` | Timed entry point; calls virtual `Run()`. |
| `Preprocessor::Run()` | Base phase pipeline: Phase I, Phase II, conditional Phase III. |
| `CustomPreprocessor<Bdd>::Run()` | Base preprocessing, then `MarkCoherence`, then `TopologicalOrder`. |
| `CustomPreprocessor<Zbdd>::Run()` | Base preprocessing, then conditional Phase IV, Phase V, `MarkCoherence`, `TopologicalOrder`. |
| `CustomPreprocessor<Mocus>::Run()` | ZBDD preprocessing, then `InvertOrder()`. |
| `CustomPreprocessor<mc::DirectEval>::Run()` | Settings-controlled pipeline with compilation levels and expansion flags. |
| `CustomPreprocessor<Mocus>::InvertOrder()` | Reorders final node orders for MOCUS traversal semantics rather than standard topological order. |
| `CustomPreprocessor<mc::DirectEval>::InvertOrder()` | Defined but not used by the shown DirectEval pipeline. |
| `CustomPreprocessor<mc::DirectEval>::remove_null_gates()` | Defined but not called. Therefore `settings_->keep_null_gates()` currently has no visible effect in the shown pipeline. |

---

### 4.2 Normalization methods

| Method | Role |
|---|---|
| `NormalizeGates(bool full)` | Wrapper: `full=true` means `ALL`; `false` means `NONE`. |
| `NormalizeGates(NormalizationType)` | Main normalization driver. Removes null gates, optionally topologically orders for threshold expansion, handles root negation, notifies parents of negating gates, recursively normalizes gates, removes null gates again. |
| `NotifyParentsOfNegativeGates()` | Rewrites parent edges to `NAND`/`NOR`/`NOT` children as negative edges, allowing those children to become `AND`/`OR`/`Null`. |
| `NormalizeGate()` | Recursive local rewrite: `NOR→OR`, `NAND→AND`, `NOT→Null`, optional `XOR` and `Atleast` expansion. |
| `NormalizeXorGate()` | Rewrites binary XOR as `(x ∧ ¬y) ∨ (¬x ∧ y)`. |
| `NormalizeAtleastGate()` | Expands `Atleast(k, {x} ∪ R)` using the recurrence `Atleast(k, {x}∪R) = (x ∧ Atleast(k−1, R)) ∨ Atleast(k, R)`, with bases `k=1 → OR`, `k=n → AND`. |
| `PropagateComplements()` | Materializes De Morgan complements of negatively referenced `AND`/`OR` child gates, using a cache so each complemented gate is created once per propagation run. |

KC significance:

- `NormalizeXorGate()` and `NormalizeAtleastGate()` can cause size growth.
- Threshold expansion is potentially exponential in the threshold gate fan-in.
- Full normalization is needed when a downstream backend only expects `AND`/`OR` plus signed edges.

---

### 4.3 Common-subexpression elimination and associative coalescing

| Method | Role |
|---|---|
| `Preprocessor::GateSet` | Hash/equality table for detecting structurally identical gates by type, signed argument set, and `min_number` for `Atleast`. |
| `ProcessMultipleDefinitions()` | Iteratively merges duplicate non-module gates. |
| `DetectMultipleDefinitions()` | DFS over graph, inserts gates into `GateSet`, records duplicates. |
| `ReplaceGate()` | Replaces all parent references to one gate with another gate. |
| `CoalesceGates(bool common)` | Public wrapper for associative flattening. |
| `CoalesceGates(const GatePtr&, bool common)` | Recursively flattens same-type positive child gates into parent gates, subject to module/sharing constraints. |

KC significance:

- Reduces repeated compilation of equivalent subfunctions.
- Turns nested associative structures into flatter n-ary gates.
- `common=false` preserves shared subgraphs; `common=true` may sacrifice sharing for flatter Boolean structure.

---

### 4.4 Module detection and module construction

| Method | Role |
|---|---|
| `DetectModules()` | Assigns DFS timings, then discovers/creates module boundaries. |
| `AssignTiming()` | DFS traversal assigning enter/exit/revisit times to gates and variables. |
| `FindModules()` | Uses timing intervals to detect whether a gate’s subgraph is internally contained or externally overlapping. |
| `ProcessModularArgs()` | For `AND`/`OR`/`NAND`/`NOR`, groups arguments into module candidates. |
| `CreateNewModule()` | Creates a new module gate containing a subset of arguments. |
| `FilterModularArgs()` | Removes modular candidates that overlap with non-modular intervals. |
| `GroupModularArgs()` | Groups modular arguments by overlapping timing intervals. |
| `CreateNewModules()` | Materializes grouped module gates. |
| `GatherModules()` | BFS collection of module gates, starting from root. |

KC significance:

- Modules are intended to isolate subgraphs that can be compiled/evaluated as units.
- This is not a formal d-DNNF decomposability guarantee.
- Module detection is based on graph traversal intervals and sharing structure, not explicit variable-disjointness.

Important implementation detail:

`ProcessModularArgs()` has:

```cpp
default:
    assert("More complex gates are considered impossible to sub-modularize!");
```

This assertion is ineffective because the string literal is truthy. As written, complex gates simply fall through without useful assertion failure. This is likely unintended.

---

### 4.5 Common-argument extraction

| Method | Role |
|---|---|
| `MergeCommonArgs()` | Runs common-argument merging for both `AND` and `OR`. |
| `MergeCommonArgs(Connective op)` | Performs merge pass for a specific connective. |
| `MarkCommonArgs()` | Counts positive/negative occurrences of arguments among gates of the same connective. |
| `GatherCommonArgs()` | Finds gates that have at least two arguments common with other same-op gates. |
| `FilterMergeCandidates()` | Removes redundant candidates and reuses already-existing subgates when one candidate’s full argument set is contained in another. |
| `GroupCandidatesByArgs()` | Groups merge candidates by overlapping common-argument index intervals and connected intersections. |
| `GroupCommonParents()` | Builds map from common argument sets to the parent gates sharing those arguments. |
| `GroupCommonArgs()` | Chooses compatible merge groups. |
| `FindOptionGroup()` | Selects a chain of compatible common-argument options. |
| `FindBaseOption()` | Heuristic selection of a promising base merge option based on extra parent counts. |
| `TransformCommonArgs()` | Materializes a new gate containing common arguments and replaces those arguments in all common parents. |

Example transformation:

```text
AND(a,b,c) and AND(a,b,d)
→ let m = AND(a,b)
→ AND(m,c) and AND(m,d)
```

Dual for `OR`.

KC significance:

- Factors repeated argument sets into explicit shared gates.
- Reduces downstream decision-diagram blow-up caused by repeated identical argument combinations.
- Heuristic, not canonical.

---

### 4.6 Distributivity and absorption

| Method | Role |
|---|---|
| `DetectDistributivity()` | Public pass wrapper. |
| `DetectDistributivity(const GatePtr&)` | Recursively searches for distributivity candidates. |
| `HandleDistributiveArgs()` | Coordinates filtering, grouping, cloning, and transformation. |
| `FilterDistributiveArgs()` | Applies absorption and removes redundant distributive candidates. |
| `GroupDistributiveArgs()` | Groups distributive options analogously to common-argument grouping. |
| `TransformDistributiveArgs()` | Applies distributive factoring recursively. |

Boolean laws implemented include:

```text
x ∧ (x ∨ y) = x
x ∨ (x ∧ y) = x
```

and factorizations such as:

```text
(a ∨ b) ∧ (a ∨ c)
→ a ∨ (b ∧ c)
```

and the dual:

```text
(a ∧ b) ∨ (a ∧ c)
→ a ∧ (b ∨ c)
```

KC significance:

- Tries to move the formula toward a smaller factored representation.
- Can expose shared substructure and reduce repeated compilation.
- Not guaranteed to reduce size in all cases, but intended as a compression heuristic.

---

### 4.7 Boolean optimization via state propagation

| Method | Role |
|---|---|
| `BooleanOptimization()` | Main pass over common gates/variables. |
| `GatherCommonNodes()` | Finds gates and variables with more than one parent. |
| `ProcessCommonNode()` | For a shared node, tries to hoist/specialize its state through ancestors. |
| `MarkAncestors()` | Marks ancestors of a common node up to a module root. |
| `PropagateState()` | Propagates assumed Boolean state of the common node through marked ancestors. |
| `DetermineGateState()` | Determines whether a gate output is forced true/false under child state counts. |
| `CollectStateDestinations()` | Finds gates where the propagated state can be materialized. |
| `CollectRedundantParents()` | Finds direct parent occurrences of the common node that can be replaced. |
| `ProcessRedundantParents()` | Replaces redundant occurrences with constants. |
| `ProcessStateDestinations()` | Adds or wraps destination gates so the common node state is represented once. |
| `ClearStateMarks()` | Clears temporary descendant/opti marks. |

Conceptually, this is a Shannon-conditioning-like optimization.

If a shared node `x` forces some ancestor to true, the code can rewrite a region into something analogous to:

```text
target = x ∨ target[x := false]
```

or, for false-forcing cases,

```text
target = ¬x ∧ target[x := true]
```

depending on polarity and gate type.

KC significance:

- Reduces repeated occurrences of common nodes by hoisting their decisive state.
- Can improve BDD/ZBDD compilation by reducing redundant paths.
- Sensitive to the correctness of the `coherent_` flag and sign structure.

---

### 4.8 Decomposition of common nodes

| Method | Role |
|---|---|
| `DecomposeCommonNodes()` | Processes common gates/variables in reverse gathered order. |
| `DecompositionProcessor::operator()` | Entry for decomposing one common node. |
| `DecompositionProcessor::MarkDestinations()` | Marks ancestor destinations up to modules. |
| `DecompositionProcessor::ProcessDestinations()` | For candidate parent gates, determines the specializing state and processes ancestors. |
| `DecompositionProcessor::ProcessAncestors()` | Replaces occurrences of the common node with constants within suitable ancestor regions; may clone when needed. |
| `DecompositionProcessor::IsAncestryWithinGraph()` | Checks whether a gate’s ancestry is contained inside a root region using timing intervals and parent closure. |
| `DecompositionProcessor::ClearAncestorMarks()` | Clears ancestry memoization marks. |
| `RegisterToClear()` | Returns whether a gate is constant or `kNull`; used during decomposition cleanup logic. |

KC significance:

- Attempts to reduce cross-sharing that prevents modular decomposition.
- Specializes repeated nodes under certain parents to improve separability.
- This is heuristic; it does not guarantee decomposable negation normal form.

Important limitation:

In `ProcessAncestors()`, the cloning branch checks:

```cpp
GatePtr clone = gate->Clone();
if (preprocessor_->RegisterToClear(clone)) {
    ...
    clone->ProcessConstantArg(node_, state);
}
```

Because `RegisterToClear(clone)` is checked **before** `ProcessConstantArg()`, this branch will usually not execute for ordinary non-null/non-constant clones. That appears to severely limit the intended cloning behavior. It may be a bug or at least an implementation limitation.

---

### 4.9 Ordering utilities

| Method | Role |
|---|---|
| `GatherNodes()` | Collects reachable gates and variables, used by order inversion. |
| `CustomPreprocessor<Mocus>::InvertOrder()` | Partitions gates into modules and non-modules, reverses non-module gate order, shifts module/variable orders. |
| `CustomPreprocessor<mc::DirectEval>::InvertOrder()` | Same logic as MOCUS but unused in DirectEval pipeline. |

MOCUS order after `InvertOrder()` is **not** a standard topological order. It is a MOCUS-specific traversal order.

---

## 5. Compilation-variant pipelines

### 5.1 BDD

```cpp
void CustomPreprocessor<Bdd>::Run() {
    Preprocessor::Run();
    pdag::Transform(graph_, &pdag::MarkCoherence, &pdag::TopologicalOrder);
}
```

Pipeline:

1. Phase I
2. Phase II
3. Phase III if `!graph_->normal()`
4. Mark coherence
5. Assign topological order

KC interpretation:

- Prepares a general Boolean function for BDD compilation.
- BDDs can naturally represent noncoherent Boolean functions, so the pipeline does not try to force monotonicity.
- It fully normalizes non-`AND`/`OR` connectives through Phase III.
- It computes coherence only as metadata, not as a constraint.

Expected nontrivial postcondition, assuming the PDAG’s `normal_` flag was accurate:

- Gates are `AND`/`OR`.
- Negation may remain as signed edges and/or root complement.
- Constants/nulls removed, except trivial-root representation.
- `graph_->coherent()` is accurate after `MarkCoherence()`.
- Node orders are topological.

Not guaranteed:

- Coherence.
- NNF with negation only at variables.
- DNNF/decomposability/determinism/smoothness.
- Minimal or canonical PDAG structure.

---

### 5.2 ZBDD

```cpp
void CustomPreprocessor<Zbdd>::Run() {
    Preprocessor::Run();
    pdag::Transform(
        graph_,
        [this](Pdag *) {
            if (!graph_->coherent())
                RunPhaseFour();
        },
        [this](Pdag *) { RunPhaseFive(); },
        &pdag::MarkCoherence,
        &pdag::TopologicalOrder);
}
```

Pipeline:

1. Base preprocessing: Phase I, Phase II, conditional Phase III.
2. If graph is marked incoherent, run Phase IV.
3. Run Phase V.
4. Mark coherence.
5. Assign topological order.

KC interpretation:

- ZBDD compilation is usually most effective for sparse families of sets, e.g. cut sets.
- Coherent monotone formulas map naturally to set families.
- Phase IV tries to push complements and make noncoherent structure more explicit.
- Phase V aggressively flattens/factors for set-style compilation.

Expected nontrivial postcondition:

- Normal `AND`/`OR` basis, assuming `normal_` was accurate.
- Additional complement propagation and aggressive coalescing have been attempted.
- Final coherence flag is accurate.
- Final order is topological.

Not guaranteed:

- Coherence for genuinely nonmonotone formulas.
- Absence of negative variables.
- Strict NNF after the whole ZBDD pipeline, because Phase II/Phase V can introduce additional signed references.
- Suitability for monotone cut-set semantics unless `MarkCoherence()` says the graph is coherent.

KC consequence:

If the downstream ZBDD compiler assumes a monotone set family, upstream or downstream code must reject or specially handle cases where final `graph->coherent()` is false.

---

### 5.3 MOCUS

```cpp
void CustomPreprocessor<Mocus>::Run() {
    CustomPreprocessor<Zbdd>::Run();
    pdag::Transform(graph_, [this](Pdag *) { InvertOrder(); });
}
```

Pipeline:

1. Full ZBDD preprocessing.
2. MOCUS-specific order inversion.

KC interpretation:

- Structurally, MOCUS receives the same kind of graph as ZBDD.
- The difference is ordering:
  - ZBDD keeps topological order.
  - MOCUS reverses/reweights order for MOCUS traversal/cut-set enumeration.

Expected nontrivial postcondition:

- Same structural form as ZBDD.
- Coherence flag computed before order inversion.
- Order is no longer standard topological order.

Not guaranteed:

- Same as ZBDD: coherence is not forced.
- If downstream MOCUS assumes coherent fault trees, it must check final coherence.

---

### 5.4 DirectEval

DirectEval has a separate settings-controlled pipeline:

```cpp
const int compilation_target = settings_->compilation_level();
```

Important: `settings_` is dereferenced unconditionally. A DirectEval preprocessor must be constructed with settings.

#### DirectEval flags

| Setting | Implementation effect |
|---|---|
| `expand_atleast_gates()` | Before level cutoff, runs `NormalizeGates(ATLEAST)` if `!graph_->normal()`. |
| `expand_xor_gates()` | Before level cutoff, runs `NormalizeGates(XOR)` if `!graph_->normal()`. |
| `compilation_level()` | Chooses how many additional preprocessing blocks run. |
| `keep_null_gates()` | Referenced only in unused `remove_null_gates()`; no effect in the shown `Run()`. |

Important oddity:

At levels `>= 6`, if **either** `expand_atleast_gates()` or `expand_xor_gates()` is true, `RunPhaseThree()` is called, and Phase III performs `NormalizeGates(ALL)`. Therefore, at high enough level, either expansion flag can trigger full expansion of both `XOR` and `Atleast`.

#### DirectEval level pipeline

The levels are cumulative.

| Level | Actual implementation |
|---:|---|
| `<= 0` | Only optional selected `Atleast`/`XOR` expansion flags run. Then return. If both flags are false, graph is essentially raw. |
| `1` | Adds `NormalizeGates(NONE)`. This removes explicit `NOT`/`NAND`/`NOR` forms but does not expand `XOR`/`Atleast` unless flags already did. |
| `2` | Adds basic compression: repeated CSE, module detection, non-common coalescing, common-argument merge, aggressive coalescing. |
| `3` | Adds advanced compression: CSE, modules, coalescing, common-arg merge, distributivity, module detection, Boolean optimization, decomposition, final coalescing. |
| `4` | After all previous blocks, runs `RunPhaseOne()`. |
| `5` | Adds `RunPhaseTwo()`. |
| `6` | If `!graph_->normal()` and at least one expansion flag is true, runs `RunPhaseThree()`, fully normalizing to `AND`/`OR`. |
| `7` | If `!graph_->coherent()` and at least one expansion flag is true, runs `RunPhaseFour()`. |
| `>= 8` | Adds `RunPhaseFive()`. |

DirectEval KC interpretation:

- DirectEval is not necessarily compiling to a canonical KC language.
- It is a configurable PDAG simplifier/evaluator preparation pipeline.
- Low levels may preserve high-level connectives such as `XOR` and `Atleast`.
- Higher levels increasingly resemble the BDD/ZBDD preprocessing pipeline, but DirectEval never calls final `MarkCoherence()` or `TopologicalOrder()` in the shown code.

DirectEval postcondition caveats:

- No DirectEval level computes final coherence metadata.
- No DirectEval level assigns final topological order.
- Unless level `>= 6` and expansion flags allow Phase III, `XOR`/`Atleast` may remain.
- At low levels, root `kNull` wrappers may remain because some single-pass `Transform()` calls do not have a subsequent pass that forces `IsTrivial()` collapse.
- `normal_` may be stale after selected `NormalizeGates(ATLEAST)` or `NormalizeGates(XOR)`, because those calls do not set `graph_->normal(true)`.

---

## 6. Variant differences from a KC perspective

| Variant | KC target interpretation | Handles noncoherence? | Normalization strength | Ordering |
|---|---|---:|---|---|
| DirectEval | Direct Boolean circuit/PDAG evaluation with optional simplification. Not necessarily a canonical KC language. | Yes, if downstream evaluator supports remaining connectives/signs. | Configurable; may leave complex gates. Full normalization only at high level with expansion flags. | No final order assigned in shown pipeline. |
| BDD | General Boolean-function KC. BDDs naturally support negation/nonmonotonicity. | Yes. | Full `AND`/`OR` normalization for non-normal graphs. | Standard topological order. |
| ZBDD | Sparse set-family / cut-set-oriented KC. Best for coherent monotone formulas. | Partially; final graph may remain incoherent and must be checked. | BDD-like normalization plus complement propagation and aggressive coalescing. | Standard topological order. |
| MOCUS | Cut-set enumeration style processing. Structurally same as ZBDD preprocessing. | Same caveats as ZBDD. | Same as ZBDD. | MOCUS-specific inverted order, not topological. |

---

## 7. Prerequisites and constraints

### Global graph prerequisites

The passes assume:

1. The graph is acyclic.
2. Parent maps are correct.
3. Root has no parents.
4. Gate argument sets are consistent with `gate_args_`, `variable_args_`, and `constant_`.
5. Gate shapes satisfy:
   - `kNull`/`kNot`: one argument.
   - `kXor`: exactly two arguments.
   - `kAtleast`: `min_number > 1` and `args.size() > min_number`.
   - `AND`/`OR`/`NAND`/`NOR`: more than one argument.
6. Constants and null gates have been cleaned before passes that assert `SANITY_ASSERT`.

Most of these are enforced only by `assert()`. In release builds with assertions disabled, malformed graphs may fail later or silently produce invalid transformations.

### External pointer constraint

`Pdag::IsTrivial()` asserts:

```cpp
assert(root_.use_count() == 1 && "Graph gate pointers outside of the graph!");
```

Because `pdag::Transform()` calls mutable `IsTrivial()`, users should not retain external owning `GatePtr`s to the root during preprocessing.

### Stage-specific constraints

| Stage/pass | Constraint |
|---|---|
| `RunPhaseTwo()` | Expects no constants/null gates and structurally valid gates. |
| `NormalizeAtleastGate()` | Requires topological order; `NormalizeGates(ATLEAST/ALL)` sets it first. |
| `RunPhaseFour()` / `PropagateComplements()` | Assumes complemented child gates are `AND`/`OR`. It should be run after full normalization. |
| `TopologicalOrder()` | Assumes no constant gates in the nontrivial graph. |
| `MarkCoherence()` | Assumes no constant gates in reachable nontrivial structure. |

### Semantic constraints by backend

#### BDD

No monotonicity/coherence prerequisite is necessary at the Boolean-function level.

However, if upstream manually mutates the PDAG, the `normal_` flag must be accurate or Phase III may be skipped incorrectly.

#### ZBDD

If the intended semantics are monotone cut sets / set families, the formula should be coherent or should be rejected after final `MarkCoherence()` if incoherent.

ZBDD can technically be used for more general encodings, but this preprocessor does not guarantee a clean positive set-family representation for noncoherent formulas.

#### MOCUS

Same coherence concern as ZBDD, usually stronger: classical MOCUS-style minimal cut-set enumeration assumes coherent positive fault trees.

#### DirectEval

DirectEval’s required input constraints depend on level/flags and the evaluator’s supported gate set.

If the downstream DirectEval engine cannot handle `XOR`, `Atleast`, signed gate references, `kNull`, or root complement, the chosen DirectEval level/flags must eliminate those constructs. The shown pipeline does not universally guarantee that unless configured carefully.

---

## 8. Postcondition summary

Assuming nontrivial graphs and accurate `normal_`/`coherent_` tracking from the provided `Pdag` constructor:

| Pipeline | Cleaned constants/nulls? | Only `AND`/`OR` gates? | Coherence guaranteed? | Coherence annotated? | Order assigned? |
|---|---:|---:|---:|---:|---:|
| BDD | Yes | Yes | No | Yes | Topological |
| ZBDD | Yes | Yes | Only if formula is actually monotone | Yes | Topological |
| MOCUS | Yes | Yes | Only if formula is actually monotone | Yes | MOCUS-inverted |
| DirectEval level `<=0` | No general guarantee | No | No | No | No |
| DirectEval level `1` | Partial cleanup; root null wrapper possible | No, unless flags expand all needed types | No | No | No |
| DirectEval level `2–3` | Better cleanup/compression, but no final metadata | No, unless flags/earlier passes expand | No | No | No |
| DirectEval level `4–5` | Stronger cleanup via Phase I/II | Not necessarily; `XOR`/`Atleast` may remain | No | No | No |
| DirectEval level `>=6` with expansion flag | Yes | Yes after Phase III | No | No | No |
| DirectEval level `>=7` with expansion flag | Yes | Yes | No; Phase IV only propagates complements | No | No |
| DirectEval level `>=8` | Yes, plus aggressive coalescing | Depends on whether Phase III ran | No | No | No |

Trivial graphs are special: the representation may remain a root `kNull` over a variable or constant by design.

---

## 9. Important implementation limitations and downstream implications

### 1. These passes do not produce a canonical KC representation

The PDAG after preprocessing is not a BDD, not a ZBDD, not d-DNNF, not deterministic, not smooth, and not canonical. It is merely a reduced/factored Boolean DAG intended to make later compilation cheaper.

### 2. Coherence is detected, not enforced

For genuinely nonmonotone formulas, no pass can make the formula coherent without changing semantics or introducing a richer variable encoding. ZBDD/MOCUS downstream code must check final `graph->coherent()` if monotone cut-set semantics are required.

### 3. `normal_` and `coherent_` can be stale

Only `RunPhaseThree()` explicitly sets `normal(true)`. Selected DirectEval normalization of `Atleast`/`XOR` does not update `normal_`.

Only `pdag::MarkCoherence()` recomputes coherence. DirectEval never calls it in the shown code.

Downstream code should not trust these flags after DirectEval preprocessing unless it explicitly reruns the relevant marking pass.

### 4. DirectEval flags are asymmetric and partly surprising

At low levels, `expand_atleast_gates()` and `expand_xor_gates()` only expand selected connective families.

At level `>=6`, either flag can cause `RunPhaseThree()`, which expands all non-normal connectives.

### 5. `keep_null_gates()` appears unused

`CustomPreprocessor<mc::DirectEval>::remove_null_gates()` references `settings_->keep_null_gates()`, but `remove_null_gates()` is not called in `Run()`. Therefore, in the shown implementation, that setting has no effect.

### 6. MOCUS final order is not topological

After `CustomPreprocessor<Mocus>::Run()`, the graph structure is ZBDD-like, but node order is intentionally inverted. Downstream code must not assume child-before-parent topological order for MOCUS.

### 7. Module flags are not globally cleared before detection

`DetectModules()` marks new modules but does not obviously clear all previous `module_` flags. Many transformations appear designed to preserve or explicitly adjust module flags, but module status should be understood as heuristic metadata maintained across transformations, not a freshly recomputed proof of decomposability.

### 8. Some implementation details look suspicious

Notable examples:

- `assert("More complex gates are considered impossible to sub-modularize!")` is effectively a no-op.
- The decomposition cloning path checks `RegisterToClear(clone)` before applying `ProcessConstantArg()`, so it likely rarely executes as intended.
- `CustomPreprocessor<mc::DirectEval>::InvertOrder()` is implemented but not used.
- DirectEval dereferences `settings_` unconditionally.

These do not necessarily make the whole pipeline incorrect, but they matter for maintenance, testing, and interpreting guarantees.

---

## 10. Practical guidance

If the downstream target is:

### General BDD

Use the BDD preprocessor. It gives a cleaned, normalized `AND`/`OR` PDAG with signed edges and topological order. Do not require coherence.

### ZBDD for cut sets

Use the ZBDD preprocessor, but require:

```cpp
pdag::MarkCoherence(graph);
assert(graph->coherent());
```

or equivalent final validation if the ZBDD backend assumes monotone set-family semantics.

### MOCUS

Use the MOCUS preprocessor only when the downstream algorithm expects the MOCUS-specific order. Do not treat the final order as topological. Coherence should be checked for classical cut-set enumeration.

### DirectEval

Choose the level based on what the evaluator can handle:

- If it can evaluate arbitrary PDAG connectives, low levels are acceptable.
- If it needs only `AND`/`OR`, use level `>=6` with expansion enabled.
- If it needs coherence metadata or a topological order, explicitly run `pdag::MarkCoherence()` and/or `pdag::TopologicalOrder()` afterward.
- Do not rely on `keep_null_gates()` in the shown implementation.