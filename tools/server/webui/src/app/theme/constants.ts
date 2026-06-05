import { getInkPalette } from "../components/plot/plotTokens";

export const HOST_SHORT = "Core i7 10700";

export const DEVICE_ORDER = [
  HOST_SHORT,
  "UHD 630",
  "GTX 1660 S",
  "RTX 4090",
  "RTX 5090",
  "L40S",
];

export function getDeviceColor(deviceShort: string): string {
  const p = getInkPalette();
  const colors: Record<string, string> = {
    [HOST_SHORT]: p.paynes,
    "UHD 630": p.ochre,
    "GTX 1660 S": p.indigo,
    "RTX 4090": p.madder,
    "RTX 5090": p.sage,
    "L40S": p.indigoDeep,
  };
  return colors[deviceShort] ?? p.ink;
}

export function getFlagColor(flagId: string): string {
  const p = getInkPalette();
  const colors: Record<string, string> = {
    C0: p.paynes,
    C1: p.indigo,
    C2: p.madder,
    C3: p.sage,
    C4: p.ochre,
    C8: p.indigoDeep,
  };
  return colors[flagId] ?? p.ink;
}

export const ACTIVE_COMPILER_FLAGS = ["C0", "C1", "C2", "C3", "C4", "C8"];
