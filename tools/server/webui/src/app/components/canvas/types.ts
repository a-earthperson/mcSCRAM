export type FaultTreeNode = {
  id: string;
  kind: "gate" | "basic_event";
  label: string;
  connective?: string;
  min?: string | null;
};

export type FaultTreeEdge = {
  source: string;
  target: string;
};

/** Server-provided fault-tree graph for canvas rendering. */
export type FaultTreePayload = {
  model: string | null;
  source?: string;
  top: string | null;
  nodes: FaultTreeNode[];
  edges: FaultTreeEdge[];
};

export type RenderNode = {
  id: string;
  node: FaultTreeNode;
  x: number;
  y: number;
};

export type RenderEdge = {
  source: number;
  target: number;
};

export type FaultTreeLayout = {
  nodes: RenderNode[];
  edges: RenderEdge[];
  width: number;
  height: number;
};

export type ViewTransform = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

export type CanvasSize = {
  width: number;
  height: number;
};
