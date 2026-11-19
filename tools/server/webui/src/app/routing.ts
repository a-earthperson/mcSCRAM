import { playgroundPath } from "./tabs/playground/routes";
import { knownAppPaths } from "./tabs/routes";

/** Map `/` and unknown paths to the playground. */
export function normalizeAppPath(url: string, basename = ""): string {
  const { pathname, search, hash } = new URL(url, "http://ssr.local");
  const normalizedBase = basename.replace(/\/$/, "");
  const pathWithoutBase = normalizedBase && pathname.startsWith(normalizedBase)
    ? pathname.slice(normalizedBase.length) || "/"
    : pathname;
  const bare = pathWithoutBase.replace(/\/$/, "") || "/";
  const resolved = bare === "/" || !knownAppPaths().includes(bare)
    ? playgroundPath
    : pathWithoutBase;
  return `${normalizedBase}${resolved}${search}${hash}`;
}
