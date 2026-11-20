/**
 * Split catalog model ids into the owner's `<system>/<format>/<model…>`
 * folders and derive display labels from `relativePath`.
 *
 * Catalog ids are POSIX paths relative to a server search root. Segment 1 is
 * System, segment 2 is Format, and everything after that (joined with `/`) is
 * Model. Short ids from flat layouts (`baobab1`, `aralia/chinese`) keep the
 * leaf as Model and fill missing parents with {@link MISSING_PATH_LEVEL} so
 * they stay selectable. System and format names are never hard-coded.
 *
 * XML ids have already dropped `.xml`; `.pcfb` / `.safetensors` ids keep the
 * suffix. The file extension used for labels and the Format badge always
 * comes from `relativePath` (the real filename), not the id.
 */

/** Neutral group for a missing System or Format segment. */
export const MISSING_PATH_LEVEL = "—";

export type CatalogModelRef = {
  modelId: string;
  relativePath: string;
};

export type ParsedModelPath = {
  modelId: string;
  relativePath: string;
  system: string;
  format: string;
  /** Remaining id segments after System/Format, joined with `/`. */
  model: string;
  /** Final `relativePath` extension, without the dot; empty if none. */
  extension: string;
  /** `model` with the final `relativePath` extension stripped. */
  label: string;
};

export type ModelOption = {
  modelId: string;
  /** Stem, plus a muted extension suffix when two files share a stem. */
  label: string;
  extension: string;
};

export type ModelPathIndex = {
  readonly entries: readonly ParsedModelPath[];
  readonly systems: readonly string[];
  formatsFor(system: string): readonly string[];
  modelsFor(system: string, format: string): readonly ModelOption[];
  parse(modelId: string): ParsedModelPath | undefined;
};

export type DisplayedModelSelection = {
  system: string;
  format: string;
  extension: string;
  systems: readonly string[];
  formats: readonly string[];
  models: readonly ModelOption[];
};

function posixSegments(path: string): string[] {
  return path.split("/").filter((part) => part.length > 0);
}

function basename(path: string): string {
  const parts = posixSegments(path);
  return parts[parts.length - 1] ?? "";
}

/** Final extension of `relativePath`'s basename; empty when there isn't one. */
export function fileExtension(relativePath: string): string {
  const name = basename(relativePath);
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return "";
  return name.slice(dot + 1);
}

/**
 * Strip only the final `relativePath` extension from the model id remainder.
 * `nus9601.f5.safetensors` → `nus9601.f5`; XML ids that already dropped
 * `.xml` are unchanged. Multi-segment remainders keep leading directories.
 */
export function stripFinalExtension(model: string, extension: string): string {
  if (!extension) return model;
  const parts = posixSegments(model);
  if (parts.length === 0) return model;
  const last = parts[parts.length - 1] ?? "";
  const suffix = `.${extension}`;
  if (last.length <= suffix.length) return model;
  if (!last.toLowerCase().endsWith(suffix.toLowerCase())) return model;
  parts[parts.length - 1] = last.slice(0, last.length - suffix.length);
  return parts.join("/");
}

/**
 * Right-align short ids onto `<system>/<format>/<model>`: the leaf is always
 * Model. A single segment (`baobab1`) is a bare model; two segments
 * (`aralia/chinese`) are Format + Model with System filled as `—`.
 */
export function splitModelId(modelId: string): Pick<ParsedModelPath, "system" | "format" | "model"> {
  const parts = posixSegments(modelId);
  if (parts.length === 0) {
    return { system: MISSING_PATH_LEVEL, format: MISSING_PATH_LEVEL, model: MISSING_PATH_LEVEL };
  }
  if (parts.length === 1) {
    return { system: MISSING_PATH_LEVEL, format: MISSING_PATH_LEVEL, model: parts[0] };
  }
  if (parts.length === 2) {
    return { system: MISSING_PATH_LEVEL, format: parts[0], model: parts[1] };
  }
  return { system: parts[0], format: parts[1], model: parts.slice(2).join("/") };
}

export function parseCatalogModel(entry: CatalogModelRef): ParsedModelPath {
  const { system, format, model } = splitModelId(entry.modelId);
  const extension = fileExtension(entry.relativePath);
  return {
    modelId: entry.modelId,
    relativePath: entry.relativePath,
    system,
    format,
    model,
    extension,
    label: stripFinalExtension(model, extension),
  };
}

function comparePathLevel(a: string, b: string): number {
  if (a === MISSING_PATH_LEVEL && b !== MISSING_PATH_LEVEL) return 1;
  if (b === MISSING_PATH_LEVEL && a !== MISSING_PATH_LEVEL) return -1;
  return a.localeCompare(b);
}

function uniqueSorted(values: Iterable<string>): string[] {
  return [...new Set(values)].sort(comparePathLevel);
}

function pairKey(system: string, format: string): string {
  return `${system}\0${format}`;
}

function disambiguateModelOptions(entries: readonly ParsedModelPath[]): ModelOption[] {
  const stemCounts = new Map<string, number>();
  for (const entry of entries) {
    stemCounts.set(entry.label, (stemCounts.get(entry.label) ?? 0) + 1);
  }
  return entries.map((entry) => {
    const colliding = (stemCounts.get(entry.label) ?? 0) > 1;
    const suffix = entry.extension ? entry.extension.toUpperCase() : entry.modelId;
    return {
      modelId: entry.modelId,
      extension: entry.extension,
      label: colliding ? `${entry.label} · ${suffix}` : entry.label,
    };
  });
}

export function buildModelPathIndex(models: readonly CatalogModelRef[]): ModelPathIndex {
  const entries = models.map(parseCatalogModel);
  const byId = new Map(entries.map((entry) => [entry.modelId, entry]));
  const systems = uniqueSorted(entries.map((entry) => entry.system));

  const formatsBySystem = new Map<string, string[]>();
  const modelsByPair = new Map<string, ParsedModelPath[]>();
  for (const system of systems) {
    const underSystem = entries.filter((entry) => entry.system === system);
    formatsBySystem.set(system, uniqueSorted(underSystem.map((entry) => entry.format)));
    for (const entry of underSystem) {
      const key = pairKey(system, entry.format);
      const group = modelsByPair.get(key);
      if (group) group.push(entry);
      else modelsByPair.set(key, [entry]);
    }
  }

  for (const group of modelsByPair.values()) {
    group.sort((a, b) => a.label.localeCompare(b.label) || a.modelId.localeCompare(b.modelId));
  }

  const optionCache = new Map<string, ModelOption[]>();

  return {
    entries,
    systems,
    formatsFor(system: string) {
      return formatsBySystem.get(system) ?? [];
    },
    modelsFor(system: string, format: string) {
      const key = pairKey(system, format);
      const cached = optionCache.get(key);
      if (cached) return cached;
      const options = disambiguateModelOptions(modelsByPair.get(key) ?? []);
      optionCache.set(key, options);
      return options;
    },
    parse(modelId: string) {
      return byId.get(modelId);
    },
  };
}

export function displayedModelSelection(index: ModelPathIndex, selectedId: string): DisplayedModelSelection {
  const selected = index.parse(selectedId);
  const systems = index.systems;
  const system = selected?.system ?? systems[0] ?? MISSING_PATH_LEVEL;
  const formats = index.formatsFor(system);
  const format = selected && selected.system === system && formats.includes(selected.format)
    ? selected.format
    : (formats[0] ?? MISSING_PATH_LEVEL);
  const models = index.modelsFor(system, format);
  return {
    system,
    format,
    extension: selected?.extension ?? models[0]?.extension ?? "",
    systems,
    formats,
    models,
  };
}

/** Keep the current id if it still lives under `system`; else the first model there. */
export function modelIdAfterSystemChange(index: ModelPathIndex, selectedId: string, system: string): string {
  const current = index.parse(selectedId);
  if (current && current.system === system) return selectedId;
  const format = index.formatsFor(system)[0];
  if (format === undefined) return "";
  return index.modelsFor(system, format)[0]?.modelId ?? "";
}

/** Keep the current id if it still lives under the current system + `format`. */
export function modelIdAfterFormatChange(index: ModelPathIndex, selectedId: string, format: string): string {
  const current = index.parse(selectedId);
  const system = current?.system ?? index.systems[0];
  if (system === undefined) return "";
  if (current && current.system === system && current.format === format) return selectedId;
  return index.modelsFor(system, format)[0]?.modelId ?? "";
}

export function formatExtensionBadge(extension: string): string {
  return extension.trim().toUpperCase();
}
