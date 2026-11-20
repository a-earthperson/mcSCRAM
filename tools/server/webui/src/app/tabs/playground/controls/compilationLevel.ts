/**
 * DirectEval `compilation_level` (0–8) as implemented in
 * `src/mc/core/preprocessor.cc` in fuzzybit-org/fuzzybits. Levels are cumulative: Cn runs
 * every pass in C0…Cn−1, then the additions for n.
 */

export type CompilationLevelOption = {
  level: number;
  /** Shown in the &lt;select&gt;; keep it short. */
  label: string;
  /** One-line summary for tooltips / docs. */
  summary: string;
};

/** Ordered options for the playground compiler level control. */
export const COMPILATION_LEVEL_OPTIONS: readonly CompilationLevelOption[] = [
  {
    level: 0,
    label: "C0 — Pass-through",
    summary:
      "Pass-through only.",
  },
  {
    level: 1,
    label: "C1 — Replace NULL/NOT",
    summary:
      "Replace NULL gates with edges; NOT gates with complemented edges.",
  },
  {
    level: 2,
    label: "C2 — Find dupes, detect modules",
    summary:
      "Remove duplicates, detect modules, coalesce gates (non-shared then shared), merge common arguments.",
  },
  {
    level: 3,
    label: "C3 — detect distributivity",
    summary:
      "Detect distributivity, decompose common nodes, then coalesce.",
  },
  {
    level: 4,
    label: "C4 — Rerun C1",
    summary:
      "Rerun C1 → C2 → C3.",
  },
  {
    level: 5,
    label: "C5 — Rerun C2 → C3",
    summary:
      "Rerun the C2 → C3 pipeline on the cleaned graph.",
  },
  {
    level: 6,
    label: "C6 — Expand VOT/XOR → C2 → C3",
    summary:
      "Rewrite VOT and XOR gates as AND/OR then rerun C2 → C3.",
  },
  {
    level: 7,
    label: "C7 — Push NOTs down",
    summary:
      "De Morgan / complement propagation toward basic events.",
  },
  {
    level: 8,
    label: "C8 — Full NNF",
    summary:
      "The SCRAM default.",
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

export type CompilerId = "fuzzybits" | "scram";

export type CompilerOption = {
  id: CompilerId;
  label: string;
  minLevel: number;
  maxLevel: number;
  defaultLevel: number;
};

/** Each compiler owns a contiguous range of levels; the first is the default compiler. */
export const COMPILER_OPTIONS: readonly CompilerOption[] = [
  { id: "fuzzybits", label: "FuzzyBits", minLevel: 0, maxLevel: 3, defaultLevel: 2 },
  { id: "scram", label: "SCRAM", minLevel: 4, maxLevel: 8, defaultLevel: 8 },
] as const;

export function compilerForLevel(level: number): CompilerOption {
  const clamped = clampCompilationLevel(level);
  return COMPILER_OPTIONS.find((c) => clamped >= c.minLevel && clamped <= c.maxLevel) ?? COMPILER_OPTIONS[0];
}

export function compilerOption(id: string): CompilerOption {
  return COMPILER_OPTIONS.find((c) => c.id === id) ?? COMPILER_OPTIONS[0];
}

export function compilationLevelsFor(compiler: CompilerOption): readonly CompilationLevelOption[] {
  return COMPILATION_LEVEL_OPTIONS.filter((o) => o.level >= compiler.minLevel && o.level <= compiler.maxLevel);
}

export const COMPILATION_LEVEL_FIELD_HINT = [
  "Pick a compiler and the compilation level.",
  "FuzzyBits is optimized for Monte Carlo sampling.",
  "SCRAM adds Rakhimov's five-phase NNF pipeline.",
  "Levels are cumulative.",
].join(" ");
