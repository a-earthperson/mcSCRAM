export type FaultTreeNode = {
  id: string;
  kind: "gate" | "basic_event" | "house_event";
  label: string;
  description?: string | null;
  connective?: string;
  min?: string | null;
  max?: string | null;
  flavor?: string | null;
  value?: string | null;
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

export type DagNodeType =
  | "basic_event"
  | "unknown_event"
  | "and_gate"
  | "nand_gate"
  | "or_gate"
  | "nor_gate"
  | "xor_gate"
  | "xnor_gate"
  | "atleast_gate"
  | "not_gate"
  | string;

export enum DagNode {
  basic_event = "basic_event",
  unknown_event = "unknown_event",
  and_gate = "and_gate",
  nand_gate = "nand_gate",
  or_gate = "or_gate",
  nor_gate = "nor_gate",
  xor_gate = "xor_gate",
  xnor_gate = "xnor_gate",
  atleast_gate = "atleast_gate",
  not_gate = "not_gate",
  null_gate = "null_gate",
  constant_event = "constant_event",
}

export type DagBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
};

export type DagPayload = {
  version: 1;
  model: string | null;
  source?: string | null;
  top: number | null;
  bounds: DagBounds;
  nodeTypes: DagNodeType[];
  nodes: {
    count: number;
    x: number[];
    y: number[];
    type: number[];
    layer?: number[];
    partition?: number[];
  };
  edges: {
    count: number;
    source: number[];
    target: number[];
    /** 1 when the native PDAG edge target is signed negative (complemented argument). */
    complemented?: number[];
  };
  counts?: {
    gate?: number;
    basic_event?: number;
    unknown_event?: number;
    constant?: number;
  };
  graphKind?: string;
  kernel?: {
    partition_count?: number;
    partitions?: Array<{
      id: number;
      layer: number;
      kind: string;
      connective: string;
      node_indices: number[];
      node_count: number;
    }>;
    sample_shape?: {
      batch_size: number;
      bitpacks_per_batch: number;
      bits_per_bitpack?: number;
      trials_per_pass?: number;
      total_iterations: number;
    };
  };
};
