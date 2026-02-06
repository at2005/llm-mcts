import { EdgeRecord, NodePayload, PublicNode, StoredNode, WorkerTreeSnapshot } from "./types";

const EDGE_PREFIX = "edge";

export class TreeStore {
  private readonly workerTrees = new Map<string, Map<string, StoredNode>>();

  getSnapshot(): { workers: WorkerTreeSnapshot[]; workerIds: string[] } {
    const workerIds = [...this.workerTrees.keys()].sort();
    const workers = workerIds.map((workerId) => {
      const nodes = [...this.workerTrees.get(workerId)!.values()];
      const edges = nodes
        .filter((node) => node.parentId)
        .map((node) => this.buildEdge(node.parentId as string, node.id));
      return { workerId, nodes, edges };
    });

    return { workers, workerIds };
  }

  getNode(workerId: string, nodeId: string): StoredNode | undefined {
    return this.workerTrees.get(workerId)?.get(nodeId);
  }

  reset(): void {
    this.workerTrees.clear();
  }

  resetWorker(workerId: string): boolean {
    return this.workerTrees.delete(workerId);
  }

  addNode(node: NodePayload, decodedState: string): PublicNode {
    const normalizedWorkerId = String(node.workerId);
    const normalizedParent = node.parentId === null ? null : String(node.parentId);
    const stored: StoredNode = {
      workerId: normalizedWorkerId,
      id: String(node.id),
      parentId: normalizedParent,
      contents: node.contents,
      visits: node.visits,
      value: node.value,
      decodedState,
      createdAt: new Date().toISOString()
    };

    let workerTree = this.workerTrees.get(normalizedWorkerId);
    if (!workerTree) {
      workerTree = new Map<string, StoredNode>();
      this.workerTrees.set(normalizedWorkerId, workerTree);
    }

    workerTree.set(stored.id, stored);
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
