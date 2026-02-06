export type TreeNode = {
  workerId: string;
  id: string;
  parentId: string | null;
  contents: number[];
  visits: number;
  value: number;
  decodedState: string;
  createdAt: string;
};

export type TreeEdge = {
  id: string;
  source: string;
  target: string;
};

export type WorkerTree = {
  workerId: string;
  nodes: TreeNode[];
  edges: TreeEdge[];
};

export type TreeSnapshot = {
  workers: WorkerTree[];
  workerIds: string[];
};

export type WsMessage =
  | ({ type: "tree_snapshot" } & TreeSnapshot)
  | { type: "node_added"; workerId: string; node: TreeNode; edge: TreeEdge | null }
  | { type: "worker_tree_reset"; workerId: string }
  | { type: "tree_reset" };
