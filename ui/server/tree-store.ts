import { EdgeRecord, NodePayload, PublicNode, StoredNode } from "./types";

const EDGE_PREFIX = "edge";

export class TreeStore {
  private readonly nodes = new Map<string, StoredNode>();

  getSnapshot(): { nodes: PublicNode[]; edges: EdgeRecord[] } {
    const nodes = [...this.nodes.values()];
    const edges = nodes
      .filter((node) => node.parentId)
      .map((node) => this.buildEdge(node.parentId as string, node.id));

    return { nodes, edges };
  }

  getNode(nodeId: string): StoredNode | undefined {
    return this.nodes.get(nodeId);
  }

  reset(): void {
    this.nodes.clear();
  }

  addNode(node: NodePayload, decodedState: string): PublicNode {
    const normalizedParent = node.parentId === null ? null : String(node.parentId);
    const stored: StoredNode = {
      id: String(node.id),
      parentId: normalizedParent,
      contents: node.contents,
      visits: node.visits,
      value: node.value,
      decodedState,
      createdAt: new Date().toISOString()
    };

    this.nodes.set(stored.id, stored);
    return stored;
  }

  buildEdge(source: string, target: string): EdgeRecord {
    return {
      id: `${EDGE_PREFIX}-${source}-${target}`,
      source,
      target
    };
  }
}
