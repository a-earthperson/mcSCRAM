import type { FaultTreeLayout, FaultTreePayload, RenderEdge, RenderNode } from "./types";

const LEAF_SPACING = 135;
const LEVEL_SPACING = 118;
const MARGIN_X = 80;
const MARGIN_Y = 70;

/** Layered DAG layout for fault-tree canvas rendering. */
export function layoutFaultTree(tree: FaultTreePayload): FaultTreeLayout {
  const nodeById = new Map(tree.nodes.map((node) => [node.id, node]));
  const childrenBySource = new Map<string, string[]>();
  const parentsByTarget = new Map<string, string[]>();
  for (const edge of tree.edges) {
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    const children = childrenBySource.get(edge.source) ?? [];
    children.push(edge.target);
    childrenBySource.set(edge.source, children);
    const parents = parentsByTarget.get(edge.target) ?? [];
    parents.push(edge.source);
    parentsByTarget.set(edge.target, parents);
  }

  const rootIds = tree.top && nodeById.has(tree.top)
    ? [tree.top]
    : [...nodeById.keys()].filter((id) => !(parentsByTarget.get(id)?.length));
  const reachable = new Set<string>();
  const stack = [...rootIds];
  while (stack.length) {
    const id = stack.pop()!;
    if (reachable.has(id)) continue;
    reachable.add(id);
    for (const child of childrenBySource.get(id) ?? []) {
      stack.push(child);
    }
  }
  if (!reachable.size) {
    for (const id of nodeById.keys()) reachable.add(id);
  }

  const indegree = new Map<string, number>();
  for (const id of reachable) indegree.set(id, 0);
  for (const id of reachable) {
    for (const child of childrenBySource.get(id) ?? []) {
      if (reachable.has(child)) indegree.set(child, (indegree.get(child) ?? 0) + 1);
    }
  }

  const depth = new Map<string, number>();
  const order: string[] = [];
  const ordered = new Set<string>();
  const queue = [...reachable].filter((id) => (indegree.get(id) ?? 0) === 0).sort((a, b) => {
    if (a === tree.top) return -1;
    if (b === tree.top) return 1;
    return (nodeById.get(a)?.label ?? a).localeCompare(nodeById.get(b)?.label ?? b);
  });
  for (const id of queue) depth.set(id, 0);
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const id = queue[cursor];
    if (ordered.has(id)) continue;
    order.push(id);
    ordered.add(id);
    const parentDepth = depth.get(id) ?? 0;
    for (const child of childrenBySource.get(id) ?? []) {
      if (!reachable.has(child)) continue;
      depth.set(child, Math.max(depth.get(child) ?? 0, parentDepth + 1));
      const nextIn = (indegree.get(child) ?? 0) - 1;
      indegree.set(child, nextIn);
      if (nextIn === 0) queue.push(child);
    }
  }
  for (const id of reachable) {
    if (!ordered.has(id)) {
      depth.set(id, depth.get(id) ?? 0);
      order.push(id);
      ordered.add(id);
    }
  }

  const layers = new Map<number, string[]>();
  for (const id of order) {
    const layer = depth.get(id) ?? 0;
    const ids = layers.get(layer) ?? [];
    ids.push(id);
    layers.set(layer, ids);
  }

  const layerKeys = [...layers.keys()].sort((a, b) => a - b);
  const position = new Map<string, number>();
  for (const layer of layerKeys) {
    const ids = layers.get(layer)!;
    ids.sort((a, b) => {
      const pa = parentsByTarget.get(a)?.map((parent) => position.get(parent)).filter((value): value is number => value !== undefined) ?? [];
      const pb = parentsByTarget.get(b)?.map((parent) => position.get(parent)).filter((value): value is number => value !== undefined) ?? [];
      const ba = pa.length ? pa.reduce((sum, value) => sum + value, 0) / pa.length : Number.POSITIVE_INFINITY;
      const bb = pb.length ? pb.reduce((sum, value) => sum + value, 0) / pb.length : Number.POSITIVE_INFINITY;
      if (ba !== bb) return ba - bb;
      return (nodeById.get(a)?.label ?? a).localeCompare(nodeById.get(b)?.label ?? b);
    });
    ids.forEach((id, index) => position.set(id, index));
  }

  const renderNodes: RenderNode[] = [];
  const indexById = new Map<string, number>();
  for (const layer of layerKeys) {
    const ids = layers.get(layer)!;
    const width = Math.max(0, (ids.length - 1) * LEAF_SPACING);
    ids.forEach((id, index) => {
      const node = nodeById.get(id);
      if (!node) return;
      const renderIndex = renderNodes.length;
      indexById.set(id, renderIndex);
      renderNodes.push({
        id,
        node,
        x: MARGIN_X + index * LEAF_SPACING - width / 2,
        y: MARGIN_Y + layer * LEVEL_SPACING,
      });
    });
  }

  let minX = 0;
  let maxX = MARGIN_X * 2;
  let maxY = 220;
  for (const node of renderNodes) {
    minX = Math.min(minX, node.x - MARGIN_X);
    maxX = Math.max(maxX, node.x + MARGIN_X);
    maxY = Math.max(maxY, node.y + MARGIN_Y);
  }
  if (minX < 0) {
    for (const node of renderNodes) node.x -= minX;
    maxX -= minX;
  }

  const renderEdges: RenderEdge[] = [];
  for (const edge of tree.edges) {
    const source = indexById.get(edge.source);
    const target = indexById.get(edge.target);
    if (source === undefined || target === undefined) continue;
    renderEdges.push({ source, target });
  }

  return {
    nodes: renderNodes,
    edges: renderEdges,
    width: maxX,
    height: maxY,
  };
}
