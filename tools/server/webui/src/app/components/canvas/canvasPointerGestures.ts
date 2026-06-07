export type ClientPoint = {
  clientX: number;
  clientY: number;
};

/** Euclidean distance between two pointer positions in screen space. */
export function pointerDistance(a: ClientPoint, b: ClientPoint): number {
  return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
}

/** Screen-space center between two pointers. */
export function pointerMidpoint(a: ClientPoint, b: ClientPoint): ClientPoint {
  return {
    clientX: (a.clientX + b.clientX) / 2,
    clientY: (a.clientY + b.clientY) / 2,
  };
}

/** First two entries from an active pointer map (insertion order). */
export function firstTwoPointers(
  pointers: ReadonlyMap<number, ClientPoint>,
): readonly [ClientPoint, ClientPoint] | null {
  const values = [...pointers.values()];
  if (values.length < 2) return null;
  return [values[0], values[1]];
}
