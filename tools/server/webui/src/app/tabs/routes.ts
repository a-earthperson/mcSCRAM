import { benchmarksNavItem, benchmarksPath } from "./benchmarks/routes";
import { playgroundNavItem, playgroundPath } from "./playground/routes";

export { benchmarksPath, playgroundPath };

export const topLevelNavSections = [
  playgroundNavItem,
  benchmarksNavItem,
] as const;

export function knownAppPaths(): string[] {
  return [playgroundPath, benchmarksPath];
}
