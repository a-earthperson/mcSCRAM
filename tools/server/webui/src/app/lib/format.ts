export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.001) || abs >= 100_000) return value.toExponential(digits);
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: digits + 1,
  }).format(value);
}

export function formatNumberFixed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const abs = Math.abs(value);
  return new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: digits + 1,
  }).format(value);
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

export function formatDurationMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 1_000) return `${formatNumber(value, 3)} ms`;
  if (value < 60_000) return `${formatNumber(value / 1_000, 3)} s`;
  return `${formatNumber(value / 60_000, 3)} min`;
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${formatNumber(size, 2)} ${units[index]}`;
}

export function githubPath(path: string): string {
  return `https://github.com/a-earthperson/mcSCRAM-benchmarks/blob/main/${path}`;
}

export function classNames(...names: Array<string | false | null | undefined>): string {
  return names.filter(Boolean).join(" ");
}
