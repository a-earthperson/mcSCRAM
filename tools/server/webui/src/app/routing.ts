import {
  benchmarkSectionHref,
  benchmarksBasePath,
  defaultBenchmarkSection,
} from "./tabs/benchmarks/routes";
import { playgroundPath } from "./tabs/playground/routes";
import { knownAppPaths } from "./tabs/routes";

/** Map `/` and unknown paths to the default tab route for SSR + CSR consistency. */
export function normalizeAppPath(url: string, basename = ""): string {
  const { pathname, search, hash } = new URL(url, "http://ssr.local");
  const normalizedBase = basename.replace(/\/$/, "");
  const pathWithoutBase = normalizedBase && pathname.startsWith(normalizedBase)
    ? pathname.slice(normalizedBase.length) || "/"
    : pathname;
  const bare = pathWithoutBase.replace(/\/$/, "") || "/";
  const resolved = bare === "/"
    ? playgroundPath
    : bare === benchmarksBasePath
      ? benchmarkSectionHref(defaultBenchmarkSection)
      : knownAppPaths().includes(bare)
        ? pathWithoutBase
        : playgroundPath;
  return `${normalizedBase}${resolved}${search}${hash}`;
}
