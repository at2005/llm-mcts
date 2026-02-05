export type NodePayload = {
  id: string | number;
  parentId: string | number | null;
  contents: number[];
  visits: number;
  value: number;
};

export type StoredNode = {
  id: string;
  parentId: string | null;
  contents: number[];
  visits: number;
  value: number;
  decodedContent: string;
  cumulativeText: string;
  cumulativeTokens: number[];
  createdAt: string;
};

export type PublicNode = Omit<StoredNode, "cumulativeTokens">;

export type EdgeRecord = {
  id: string;
  source: string;
  target: string;
};

export type WsMessage =
  | { type: "tree_snapshot"; nodes: PublicNode[]; edges: EdgeRecord[] }
  | { type: "node_added"; node: PublicNode; edge: EdgeRecord | null }
  | { type: "tree_reset" };
