import { EdgeRecord, NodePayload, PublicNode, StoredNode } from "./types";

const EDGE_PREFIX = "edge";

export class TreeStore {
  private readonly nodes = new Map<string, StoredNode>();

  getSnapshot(): { nodes: PublicNode[]; edges: EdgeRecord[] } {
    const nodes = [...this.nodes.values()].map((node) => this.toPublicNode(node));
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

  addNode(node: NodePayload, decodedContent: string, cumulativeText: string, cumulativeTokens: number[]): PublicNode {
    const normalizedParent = node.parentId === null ? null : String(node.parentId);
    const stored: StoredNode = {
      id: String(node.id),
      parentId: normalizedParent,
      contents: node.contents,
      visits: node.visits,
      value: node.value,
      decodedContent,
      cumulativeText,
      cumulativeTokens,
      createdAt: new Date().toISOString()
    };

    this.nodes.set(stored.id, stored);
    return this.toPublicNode(stored);
  }

  public toPublicNode(node: StoredNode): PublicNode {
    const { cumulativeTokens: _unused, ...publicNode } = node;
    return publicNode;
  }

  buildEdge(source: string, target: string): EdgeRecord {
    return {
      id: `${EDGE_PREFIX}-${source}-${target}`,
      source,
      target
    };
  }
}
