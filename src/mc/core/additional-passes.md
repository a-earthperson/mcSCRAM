# Additional DirectEval PDAG Preprocessing Passes

This note records additional PDAG simplification opportunities after reading:

- `src/mcSCRAM/src/mc/core/compiler.md`
- `src/mcSCRAM/src/pdag.cc`
- `src/mcSCRAM/src/pdag.h`
- `src/mcSCRAM/src/preprocessor.cc`
- `src/mcSCRAM/src/preprocessor.h`
- `src/mcSCRAM/src/mc/core/preprocessor.cc`

The focus is the DirectEval case, where the evaluator can retain signed
arguments/root complement and can evaluate `ATLEAST` and binary `XOR` directly.
That changes the best optimization target: for DirectEval, many reductions
should preserve high-level connectives instead of expanding them to `AND`/`OR`.

## Current Coverage

The existing implementation already covers many local and structural Boolean
reductions:

- `Gate::AddArg` handles duplicate arguments, complementary arguments, and
  constants. This includes `AND(x, x) = x`, `OR(x, x) = x`, `XOR(x, x) = false`,
  `AND(x, ~x) = false`, `OR(x, ~x) = true`, `XOR(x, ~x) = true`, and
  `ATLEAST(k, x, ~x, R) = ATLEAST(k - 1, R)`.
- `Gate::AddArg<Constant>` propagates constants through every connective,
  including threshold boundary reductions such as `ATLEAST(k, true, R) =
  ATLEAST(k - 1, R)` and `ATLEAST(k, false, R) = ATLEAST(k, R)`.
- `NormalizeGates(NONE)` removes explicit `NOT`/`NAND`/`NOR` by using signed
  edges/root complement, without expanding `XOR` or `ATLEAST`.
- `ProcessMultipleDefinitions` performs exact structural CSE by connective,
  signed argument set, and threshold value.
- `CoalesceGates` flattens positive same-type `AND`/`OR` descendants, preserving
  shared gates unless `common=true`.
- `MergeCommonArgs`, `DetectDistributivity`, `BooleanOptimization`, and
  `DecomposeCommonNodes` already cover common argument extraction, positive
  absorption/distributivity, and some Shannon-style specialization of common
  nodes.

Therefore the remaining promising DirectEval optimizations are not basic
idempotence, constants, exact CSE, or ordinary associative flattening. They are
mostly reductions that become available because DirectEval can keep complements,
thresholds, and XORs as first-class operations.

## Important DirectEval Pipeline Observation

`src/mcSCRAM/src/mc/core/preprocessor.cc` currently defines two lambdas in
`CustomPreprocessor<mc::DirectEval>::Run()`: `result` and `result2`. Only
`result()` is executed.

That matters because `result2` contains the detailed staged level comments from
the compiler documentation, including selected level-0 `ATLEAST`/`XOR`
expansion. The executed `result` pipeline is different:

- level `<= 0`: returns immediately; selected `ATLEAST`/`XOR` expansion is not
  run.
- level `1`: runs `RunPhaseOne()`.
- level `>= 2`: repeatedly runs `RunPhaseOne()`, `RunPhaseTwo()`, optional full
  `RunPhaseThree()`, optional `RunPhaseFour()`, and unconditional
  `RunPhaseFive()` for each pass from `2` through `compilation_level()`.

Any additional pass should be designed against the executed pipeline, or this
dead-code discrepancy should be resolved first. In particular, DirectEval level
`>= 2` is already more aggressive than the level table in `compiler.md` implies,
because `RunPhaseFive()` is reached unconditionally inside each repeated pass.

## Recommended Additional Passes

### 1. Complement-Aware CSE

Add a CSE pass that canonicalizes gates up to an external sign, then replaces a
duplicate with either a positive or negative reference to the canonical gate.
The current `GateSet` only detects exact matches. It misses semantically
identical gates related by De Morgan, XOR parity, or threshold complement.

Examples:

```text
AND(a, b) and OR(~a, ~b)
=> keep one node and reference the other as a complemented edge

XOR(a, b) and XOR(~a, b)
=> one canonical XOR(a, b), with one complemented reference

ATLEAST(k, a, b, c) and ATLEAST(3 - k + 1, ~a, ~b, ~c)
=> one canonical threshold, with one complemented reference
```

For `AND`/`OR`, the complement-normal form is:

```text
~AND(args) = OR(~args)
~OR(args) = AND(~args)
```

For binary `XOR`, argument signs are parity:

```text
XOR(~a, ~b) = XOR(a, b)
XOR(~a, b)  = ~XOR(a, b)
```

For threshold gates with `n` arguments:

```text
~ATLEAST(k, args) = ATLEAST(n - k + 1, ~args)
```

Implementation requirements:

- Introduce a replacement helper that can replace `gate` by `replacement` with
  an extra complement bit. `ReplaceGate` currently preserves the old parent-edge
  sign but cannot intentionally flip it.
- If the replaced gate is the root, set `graph_->root(replacement)` and toggle
  `graph_->complement()` when the canonical representative has opposite
  polarity.
- Preserve module/watched-origin semantics by default. The existing exact CSE
  skips modules; complement-aware CSE should also skip module gates unless there
  is an explicit policy to merge modules.
- Run after `RunPhaseOne()`/`NormalizeGates(NONE)`, when explicit
  `NOT`/`NAND`/`NOR` have been converted to signed edges. It should also be
  useful immediately before any optional `RunPhaseThree()` expansion, because it
  can reduce the number of high-level gates before expansion.

Expected effect:

- Reduces node count by merging dual subgraphs that are currently distinct.
- Reduces edge count when De Morgan duals are replaced by signed references
  instead of materialized complement structures.
- Can flatten effective levels when an entire complement subgraph disappears
  behind a signed edge.

This is the highest-value missing pass for DirectEval.

### 2. Signed De Morgan Coalescing

Extend associative coalescing to flatten negative opposite-type children without
materializing complement gates.

Current `CoalesceGates` handles:

```text
AND(..., AND(a, b), ...) => AND(..., a, b, ...)
OR(..., OR(a, b), ...)   => OR(..., a, b, ...)
```

For DirectEval, the signed-edge representation also permits:

```text
AND(..., ~OR(a, b), ...) => AND(..., ~a, ~b, ...)
OR(..., ~AND(a, b), ...) => OR(..., ~a, ~b, ...)
```

This directly applies De Morgan at the parent edge. It is different from
`RunPhaseFour()`: it does not need to clone or materialize a complement gate, and
it only flattens where doing so reduces local structure.

Implementation requirements:

- Only flatten through a negative child when:
  - parent is `AND` and child is `OR`, or parent is `OR` and child is `AND`;
  - the child is not a module;
  - either the child has a single parent, or the pass is explicitly running in
    an aggressive/shared mode analogous to `CoalesceGates(common=true)`.
- Transfer each child argument with sign inverted.
- Use existing `AddArg` semantics so direct contradictions and constants are
  still handled by the gate itself.

Expected effect:

- Reduces depth for formulas with complemented subexpressions.
- Often removes an entire child gate when it was single-parent.
- Avoids the node growth that can come from full complement propagation.

### 3. Contextual Unit Propagation Into Children

Add a local conditioning pass for `AND`/`OR` parents that already contain a
literal or subgate also used, with opposite polarity, inside a child.

The existing `DetectDistributivity` catches positive absorption:

```text
x & (x | y) = x
x | (x & y) = x
```

It does not catch the complementary/resolution form:

```text
x & (~x | y) = x & y
x | (~x & y) = x | y
```

More generally, for a parent argument `L`:

```text
AND(L, OR(~L, R)) => AND(L, OR(R))
OR(L, AND(~L, R)) => OR(L, AND(R))
```

If the child degenerates to one argument, it becomes a `kNull` wrapper and is
removed by existing cleanup. If it degenerates to an empty contradictory case,
the existing constant/null machinery should represent the result as a constant.

Implementation requirements:

- Apply only to positive child gates of the opposite connective first:
  `AND` parent with `OR` children, `OR` parent with `AND` children.
- For each direct parent argument `L`, remove `~L` from eligible children.
- Clone shared child gates before mutation unless running in an aggressive mode.
- Preserve module boundaries.

Expected effect:

- Reduces edges inside child gates.
- Can collapse child gates and thereby reduce levels.
- Complements `DetectDistributivity` without requiring full SAT/BDD reasoning.

This pass is local, deterministic, and should be substantially cheaper than the
existing general common-node optimizations.

### 4. Threshold Context Reduction

Exploit direct `ATLEAST` support by simplifying a threshold gate under a parent
that already fixes one of its arguments.

For a literal/subformula `L` in threshold argument set `R`:

```text
AND(L, ATLEAST(k, L, R))  = AND(L, ATLEAST(k - 1, R))
AND(~L, ATLEAST(k, L, R)) = AND(~L, ATLEAST(k, R))

OR(L, ATLEAST(k, L, R))   = OR(L, ATLEAST(k, R))
OR(~L, ATLEAST(k, L, R))  = OR(~L, ATLEAST(k - 1, R))
```

These identities are exact Boolean conditioning. They reduce threshold fan-in
without expanding the threshold.

Boundary cases should use existing threshold simplification rules:

- `ATLEAST(0, R) = true`
- `ATLEAST(k, R)` with `k > |R|` is false
- `ATLEAST(1, R) = OR(R)`
- `ATLEAST(|R|, R) = AND(R)`
- a one-argument `OR`/`AND` should become `kNull`

Implementation requirements:

- Handle signed threshold arguments by treating the signed index as the literal.
- If the threshold child is shared, clone before mutation unless in aggressive
  mode.
- Run after complement-aware CSE or local polarity canonicalization, so the same
  literal is represented consistently.

Expected effect:

- Reduces threshold edge count.
- Often lowers threshold gates to `OR`, `AND`, `kNull`, or constants.
- Avoids the exponential blow-up of `NormalizeAtleastGate()`.

### 5. Threshold Chain Simplification

Detect multiple threshold gates over the same signed argument set and simplify
their `AND`/`OR` combinations by threshold dominance.

For the same argument set `X`:

```text
OR(ATLEAST(k, X), ATLEAST(l, X))  = ATLEAST(min(k, l), X)
AND(ATLEAST(k, X), ATLEAST(l, X)) = ATLEAST(max(k, l), X)
```

With complemented threshold arguments:

```text
OR(~T_k, ~T_l)  = ~T_max(k, l)
AND(~T_k, ~T_l) = ~T_min(k, l)
```

Mixed-polarity cases sometimes become constants:

```text
OR(T_k, ~T_l)  = true  when k <= l
AND(T_k, ~T_l) = false when k >= l
```

When `k > l`, `OR(T_k, ~T_l)` represents a gap predicate
`count(X) < l or count(X) >= k`; when `k < l`, `AND(T_k, ~T_l)` represents
`k <= count(X) < l`. Those are not representable as a single existing PDAG
connective, so the pass should leave them unchanged unless a future range-gate
representation is added.

Expected effect:

- Removes redundant threshold gates and parent edges.
- Can replace multiple thresholds with one canonical threshold plus signed
  references.
- Particularly useful for cardinality constraints, which the constructor already
  represents as combinations of positive and complemented `ATLEAST` gates.

### 6. Threshold Self-Argument Absorption

Simplify a threshold gate that contains another threshold over exactly the same
argument set.

Let `T_l(X) = ATLEAST(l, X)`. Then:

```text
ATLEAST(k, X, T_l(X)) =
  ATLEAST(k - 1, X)  if l < k
  ATLEAST(k, X)      if l >= k
```

For a complemented nested threshold:

```text
ATLEAST(k, X, ~T_l(X)) =
  ATLEAST(k, X)      if l < k
  ATLEAST(k - 1, X)  if l >= k
```

These identities follow by case analysis on `count(X)`: the nested threshold
contributes exactly one additional vote, and whether that vote is present is
itself a threshold predicate over the same count.

Expected effect:

- Removes one threshold node/edge and may lower `k`.
- Can cascade into `OR`/`AND`/constant boundary reductions.
- Avoids expanding either threshold.

This is narrower than arbitrary nested-threshold reasoning, but it is exact,
cheap, and well matched to DirectEval.

### 7. XOR Polarity Canonicalization and Parity CSE

Binary `XOR` has stronger algebraic structure than the current exact CSE uses.
At minimum, normalize argument signs and perform CSE modulo XOR polarity:

```text
XOR(~a, ~b) = XOR(a, b)
XOR(~a, b)  = ~XOR(a, b)
XOR(a, ~b)  = ~XOR(a, b)
```

For larger XOR-only regions, use parity collection:

```text
XOR(XOR(a, b), XOR(a, c)) = XOR(b, c)
XOR(a, XOR(a, b))         = b
XOR(a, ~a)                = true
```

Because the current `kXor` invariant is binary, a full XOR-region pass should
not create n-ary XOR gates unless the invariant and evaluator are changed.
Instead it can collect operands modulo two, track one complement parity bit, and
rebuild a balanced binary XOR tree. Rebalancing reduces depth from a chain-like
`O(n)` form to `O(log n)` while cancellations reduce nodes.

Implementation requirements:

- Skip shared/module XOR children unless cloning/rebuilding is explicitly
  allowed.
- Preserve a complement bit on the parent edge or root complement after
  canonicalization.
- Use a stable operand ordering so rebuilt trees are deterministic and CSE can
  find duplicates.

Expected effect:

- Reduces nodes in XOR-heavy formulas with repeated operands.
- Reduces layers for long associative XOR chains.
- Avoids the four-gate expansion used by `NormalizeXorGate()`.

### 8. Reverse High-Level Gate Recognition

Add a reverse-normalization pass that recognizes expanded `AND`/`OR` structures
equivalent to native DirectEval gates and replaces them with `XOR` or
`ATLEAST`. This is the dual of `NormalizeXorGate()` and
`NormalizeAtleastGate()`: instead of expanding high-level connectives for
BDD/ZBDD-style backends, DirectEval should sometimes reintroduce them to reduce
depth, node count, and edge count.

#### Binary XOR recognition

The most direct patterns are the two standard binary XOR normal forms:

```text
OR(AND(a, ~b), AND(~a, b)) = XOR(a, b)
AND(OR(a, b), OR(~a, ~b))  = XOR(a, b)
```

The XNOR variants should be represented as a complemented XOR edge/root:

```text
OR(AND(a, b), AND(~a, ~b)) = ~XOR(a, b)
AND(OR(~a, b), OR(a, ~b))  = ~XOR(a, b)
```

This can be applied to a whole parent gate or to a subset of two child gates
inside a larger parent:

```text
OR(AND(a, ~b), AND(~a, b), z) => OR(XOR(a, b), z)
```

Implementation requirements:

- Match two children of the opposite connective, each with exactly two signed
  arguments.
- Require the two child argument sets to contain the same two underlying nodes
  and either opposite parity (XOR) or equal parity (XNOR).
- If the matched child gates are shared, remove only this parent edge; do not
  mutate the child gates.
- Represent XNOR by adding the new `XOR` gate with a negative sign, or by
  toggling `graph_->complement()` if the whole root is replaced.
- Mark `graph_->normal(false)` if a new `kXor` gate is introduced.

Expected effect:

- Replaces two binary child gates and two parent edges with one `XOR` child.
- Collapses a two-level XOR encoding into one DirectEval gate.
- Gives later XOR parity CSE a canonical target.

#### Threshold recognition from DNF/CNF

The exact monotone DNF and CNF encodings of a threshold gate are:

```text
ATLEAST(k, X) =
  OR({ AND(S) : S subset X, |S| = k })

ATLEAST(k, X) =
  AND({ OR(S) : S subset X, |S| = |X| - k + 1 })
```

The same identities hold when `X` is a set of signed PDAG arguments, provided
the candidate literal universe does not contain both `x` and `~x`. If the
universe is all negated literals, the result may be kept as
`ATLEAST(k, ~X)` or canonicalized through the threshold-complement identity:

```text
ATLEAST(k, ~X) = ~ATLEAST(|X| - k + 1, X)
```

A recognizer can therefore replace:

```text
OR(AND(a, b), AND(a, c), AND(b, c))
=> ATLEAST(2, a, b, c)

AND(OR(a, b), OR(a, c), OR(b, c))
=> ATLEAST(2, a, b, c)
```

Implementation requirements:

- For an `OR` parent, group positive `AND` children by uniform child arity `r`.
  Let `X` be the union of their signed argument sets. If the child set equals
  every `r`-combination of `X`, replace the group with `ATLEAST(r, X)`.
- For an `AND` parent, group positive `OR` children by uniform child arity `r`.
  Let `X` be the union of their signed argument sets. If the child set equals
  every `r`-combination of `X`, replace the group with
  `ATLEAST(|X| - r + 1, X)`.
- Use a cap on `binomial(|X|, r)` before enumerating combinations. The pass
  should verify exact coverage only when the candidate child count equals the
  expected binomial count and the count is below the cap.
- Reject or defer groups whose literal universe contains both `x` and `~x`.
  Those can often be simplified, but the exact recognition problem becomes a
  signed-cardinality/range problem rather than a simple threshold recognizer.
- Preserve module boundaries and avoid mutating shared matched child gates.
- Mark `graph_->normal(false)` when introducing `kAtleast`.

Expected effect:

- Replaces `binomial(n, k)` conjunction nodes plus their edges with one
  threshold gate.
- Collapses common two-level threshold encodings into a single DirectEval node.
- Can dramatically reduce the output of prior normalization or imported models
  that spell thresholds as pure `AND`/`OR` formulas.

#### Integration caveat

This pass is valuable only when DirectEval is allowed to keep high-level gates.
If a later DirectEval step calls `RunPhaseThree()` with expansion enabled, the
new `XOR`/`ATLEAST` gates will be expanded again. Therefore reverse recognition
should either run late in the DirectEval pipeline, or run only for settings where
high-level gates are intentionally preserved.

## Lower-Priority Or Riskier Passes

### Consensus and Subsumption Beyond Local Children

General consensus identities can reduce formulas such as:

```text
(a & b) | (~a & c) | (b & c) = (a & b) | (~a & c)
```

The dual forms also apply. However, detecting useful consensus opportunities in
a shared PDAG is combinatorial and can easily overlap with the existing
distributivity, common-argument, and decomposition passes. A narrowly scoped
version may be worthwhile later, but a general consensus pass should not be a
near-term DirectEval preprocessor because it needs a cost model and can become
expensive.

### General Boolean Equivalence CSE

Replacing all semantically equivalent subgraphs would require a canonical
representation or equivalence checking. A BDD/SAT-based equivalence cache could
find more reductions than structural CSE, but it would turn preprocessing into a
knowledge-compilation problem itself. Exact Boolean circuit minimization is not
a realistic preprocessing target for large fault trees.

### Full CNF/DNF Expansion

Expanding to CNF, DNF, or combinations-of-thresholds can expose absorption, but
it is often exponential and defeats the DirectEval advantage of retaining
`ATLEAST`, `XOR`, and signed edges. Expansion should remain opt-in and should be
avoided as a prerequisite for the passes above.

### Unconditional Flattening Through Shared Gates

Flattening always reduces syntactic depth but can increase edge count and
destroy sharing. The current `CoalesceGates(common)` distinction is important.
Any new flattening pass should keep the same policy: conservative mode for
single-parent children, aggressive mode only when the caller explicitly accepts
the sharing/depth trade-off.

## Suggested Integration Order

For DirectEval, the most useful order is:

1. Run null/negative normalization: `RunPhaseOne()` or `NormalizeGates(NONE)`.
2. Run complement-aware CSE.
3. Run signed De Morgan coalescing and contextual unit propagation.
4. Run threshold context/chain/self-argument reductions.
5. Run XOR polarity/parity canonicalization.
6. Run reverse high-level gate recognition when DirectEval is preserving
   `XOR`/`ATLEAST`.
7. Run exact CSE and ordinary coalescing again.
8. Run the existing Phase II-style structural passes.

This order keeps high-level connectives intact, reduces them before any optional
expansion, and then gives the existing exact structural passes another chance to
merge the newly canonicalized graph.

## Net Assessment

There are additional feasible optimizations worth implementing for DirectEval.
They are not broad new Boolean minimization algorithms; they are targeted,
semantics-preserving rewrites that exploit PDAG signed edges and native
`ATLEAST`/`XOR` support.

The strongest candidates are:

- complement-aware CSE;
- signed De Morgan coalescing;
- local complementary unit propagation;
- threshold context and threshold-chain reductions;
- XOR polarity/parity canonicalization.
- reverse recognition of expanded `XOR` and `ATLEAST` groups.

The main infeasible direction is exact global minimization or arbitrary
equivalence-based CSE. That would be too expensive and would duplicate the role
of downstream knowledge compilation. The DirectEval-specific path should instead
prefer cheap canonicalization, local conditioning, and high-level connective
reductions that avoid expanding the graph.
