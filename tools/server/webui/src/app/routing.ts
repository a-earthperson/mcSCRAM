import { defaultSection, sectionHref, sections } from "./tabs/lazyTabs";

export function knownAppPaths(): string[] {
  return sections.map((section) => sectionHref(section.path));
}

/** Map `/` and unknown paths to the default tab route for SSR + CSR consistency. */
export function normalizeAppPath(url: string, basename = ""): string {
  const { pathname, search, hash } = new URL(url, "http://ssr.local");
  const normalizedBase = basename.replace(/\/$/, "");
  const pathWithoutBase = normalizedBase && pathname.startsWith(normalizedBase)
    ? pathname.slice(normalizedBase.length) || "/"
    : pathname;
  const bare = pathWithoutBase.replace(/\/$/, "") || "/";
  const resolved = bare === "/" || !knownAppPaths().includes(bare) ? sectionHref(defaultSection) : pathWithoutBase;
  return `${normalizedBase}${resolved}${search}${hash}`;
}
