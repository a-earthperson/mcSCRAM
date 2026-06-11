import type { Point3 } from "./types";

/** Square view volume so one field unit maps to the same screen distance on X and Y. */
export const VIEW_RANGE: [[number, number], [number, number], [number, number]] = [
  [-1, 1],
  [-1, 1],
  [-1, 1],
];

export const VIEW_SCALE: Point3 = [1, 1, 1];
