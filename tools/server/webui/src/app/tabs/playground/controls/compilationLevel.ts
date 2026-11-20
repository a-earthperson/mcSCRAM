/**
 * DirectEval `compilation_level` (0–8) as implemented in
 * `src/mc/core/preprocessor.cc` in fuzzybit-org/fuzzybits. Levels are cumulative: Cn runs
 * every pass in C0…Cn−1, then the additions for n.
 */

export type CompilationLevelOption = {
  level: number;
  /** Shown in the &lt;select&gt;; keep readable at ~42–52 characters. */
  label: string;
  /** One-line summary for tooltips / docs. */
  summary: string;
};

/** Ordered options for the playground compiler level control. */
export const COMPILATION_LEVEL_OPTIONS: readonly CompilationLevelOption[] = [
  {
    level: 0,
    label: "C0 — raw tree (optional K/N·XOR normalize only)",
    summary:
      "No structural passes. Optionally expand or merge K/N and XOR connectives when those flags are set, then ship the parsed PDAG.",
  },
  {
    level: 1,
    label: "C1 — NULL/NOT → complemented edges",
    summary:
      "NormalizeGates(NONE): replace NULL gates with edges and NOT gates with complemented argument edges.",
  },
  {
    level: 2,
    label: "C2 — CSE, modules, coalesce, merge args",
    summary:
      "ProcessMultipleDefinitions, DetectModules, CoalesceGates (non-shared then shared), MergeCommonArgs. Default compilation_level (2).",
  },
  {
    level: 3,
    label: "C3 — distributivity, Boolean opt, decompose",
    summary:
      "Extended structural block: distributivity detection, Shannon Boolean optimization, decompose common nodes, then coalesce; optional RecognizeAtleast/Xor groups.",
  },
  {
    level: 4,
    label: "C4 — +Phase I (null/constant cleanup)",
    summary:
      "Rakhimov Phase I (RunPhaseOne): remove residual null/constant wrappers after C2–C3 rewrites.",
  },
  {
    level: 5,
    label: "C5 — +Phase II (structural block rerun)",
    summary:
      "Rakhimov Phase II (RunPhaseTwo): rerun the full CSE → modules → coalesce → merge → distributivity → Boolean-opt pipeline on the cleaned graph.",
  },
  {
    level: 6,
    label: "C6 — +Phase III (expand K/N·XOR → AND/OR)",
    summary:
      "Phase III when expand_atleast or expand_xor is set: fully normalize connectives to AND/OR, mark graph normal, then rerun Phase II.",
  },
  {
    level: 7,
    label: "C7 — +Phase IV (push complements toward leaves)",
    summary:
      "Phase IV when expand flags are set and the graph is non-coherent: De Morgan / complement propagation toward basic events.",
  },
  {
    level: 8,
    label: "C8 — +Phase V (NNF coalesce, full pipeline)",
    summary:
      "Phase V: aggressive shared coalescing, Phase II, coalesce again — reaches negation normal form when K/N·XOR expansion is enabled.",
  },
] as const;

export const MIN_COMPILATION_LEVEL = 0;
export const MAX_COMPILATION_LEVEL = COMPILATION_LEVEL_OPTIONS.length - 1;

export function clampCompilationLevel(level: number): number {
  if (!Number.isFinite(level)) return 2;
  return Math.min(MAX_COMPILATION_LEVEL, Math.max(MIN_COMPILATION_LEVEL, Math.round(level)));
}

export function compilationLevelOption(level: number): CompilationLevelOption {
  return COMPILATION_LEVEL_OPTIONS[clampCompilationLevel(level)];
}

export function compilationLevelLabel(level: number): string {
  return compilationLevelOption(level).label;
}

export function compilationLevelSummary(level: number): string {
  return compilationLevelOption(level).summary;
}

export const COMPILATION_LEVEL_FIELD_HINT = [
  "Cumulative DirectEval preprocessor depth (compilation_level 0–8). Each Cn runs all passes through Cn−1, then adds the row below.",
  "C0–C3 are the native mcSCRAM structural passes; C4–C8 add Rakhimov's five-phase NNF pipeline (Phase I–V), staggered one phase per level.",
  "C6–C7 run only when Expand VOT or Expand XOR is enabled; C8 completes the pipeline toward AND/OR NNF when those expansions are on.",
].join(" ");
