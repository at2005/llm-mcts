export type TreeNode = {
  id: string;
  parentId: string | null;
  contents: number[];
  visits: number;
  value: number;
  decodedContent: string;
  cumulativeText: string;
  createdAt: string;
};

export type TreeEdge = {
  id: string;
  source: string;
  target: string;
};

export type TreeSnapshot = {
  nodes: TreeNode[];
  edges: TreeEdge[];
};

export type WsMessage =
  | ({ type: "tree_snapshot" } & TreeSnapshot)
  | { type: "node_added"; node: TreeNode; edge: TreeEdge | null }
  | { type: "tree_reset" };
