import type { DagPayload, FaultTreePayload } from "../../components/canvas";

export function formatCounts(counts: DagPayload["counts"]): string {
  return `${counts?.gate ?? 0} gates · ${counts?.basic_event ?? 0} basic events`;
}

export function formatFaultTreeCounts(tree: FaultTreePayload): string {
  const gates = tree.nodes.filter((node) => node.kind === "gate").length;
  const basicEvents = tree.nodes.filter((node) => node.kind === "basic_event").length;
  const houseEvents = tree.nodes.filter((node) => node.kind === "house_event").length;
  const eventSummary = houseEvents
    ? `${basicEvents} basic events · ${houseEvents} house events`
    : `${basicEvents} basic events`;
  return `${gates} gates · ${eventSummary} · ${tree.edges.length} edges`;
}
