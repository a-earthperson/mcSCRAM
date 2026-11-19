import { playgroundNavItem, playgroundPath } from "./playground/routes";

export { playgroundPath };

export const topLevelNavSections = [
  playgroundNavItem,
] as const;

export function knownAppPaths(): string[] {
  return [playgroundPath];
}
