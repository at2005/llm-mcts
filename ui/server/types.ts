export type NodePayload = {
  workerId: string | number;
  id: string | number;
  parentId: string | number | null;
  contents: number[];
  visits: number;
  value: number;
};

export type StoredNode = {
  workerId: string;
  id: string;
  parentId: string | null;
  contents: number[];
  visits: number;
  value: number;
  decodedState: string;
  createdAt: string;
};

export type PublicNode = StoredNode;

export type EdgeRecord = {
  id: string;
  source: string;
  target: string;
};

export type WorkerTreeSnapshot = {
  workerId: string;
  nodes: PublicNode[];
  edges: EdgeRecord[];
};

export type WorkerTreeMetadata = {
  workerId: string;
  nodeCount: number;
  edgeCount: number;
  rootNodeIds: string[];
  firstNodeAt: string | null;
  latestNodeAt: string | null;
};

export type WsMessage =
  | { type: "tree_snapshot"; workers: WorkerTreeSnapshot[]; workerIds: string[] }
  | { type: "node_added"; workerId: string; node: PublicNode; edge: EdgeRecord | null }
  | { type: "worker_tree_reset"; workerId: string }
  | { type: "tree_reset" };
