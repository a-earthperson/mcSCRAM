import { benchmarkSectionHrefs, benchmarkSections, benchmarkSectionHref, benchmarksBasePath, benchmarksNavItem } from "./benchmarks/routes";
import { playgroundNavItem, playgroundPath } from "./playground/routes";

export { benchmarkSections, benchmarkSectionHref, benchmarksBasePath, playgroundPath };

export const topLevelNavSections = [
  playgroundNavItem,
  benchmarksNavItem,
] as const;

export function knownAppPaths(): string[] {
  return [
    playgroundPath,
    benchmarksBasePath,
    ...benchmarkSectionHrefs,
  ];
}
