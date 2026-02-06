import { Edge, Node, Position } from "reactflow";

const NODE_WIDTH = 180;
const NODE_HEIGHT = 56;
const H_GAP = 32;
const V_GAP = 56;
const ROOT_GAP = 96;

type LayoutState = {
  xById: Map<string, number>;
  depthById: Map<string, number>;
  childrenById: Map<string, string[]>;
  visiting: Set<string>;
  nextX: number;
};

function ensureChildrenMap(nodeIds: string[], edges: Edge[]): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const id of nodeIds) {
    map.set(id, []);
  }

  for (const edge of edges) {
    const source = edge.source;
    const target = edge.target;
    if (!map.has(source) || !map.has(target) || source === target) {
      continue;
    }
    map.get(source)!.push(target);
  }

  return map;
}

function assignDepths(roots: string[], childrenById: Map<string, string[]>): Map<string, number> {
  const depthById = new Map<string, number>();
  const queue: string[] = [];

  for (const root of roots) {
    depthById.set(root, 0);
    queue.push(root);
  }

  while (queue.length > 0) {
    const current = queue.shift()!;
    const currentDepth = depthById.get(current) ?? 0;
    const children = childrenById.get(current) ?? [];

    for (const child of children) {
      const nextDepth = currentDepth + 1;
      const prevDepth = depthById.get(child);
      if (prevDepth === undefined || nextDepth < prevDepth) {
        depthById.set(child, nextDepth);
        queue.push(child);
      }
    }
  }

  return depthById;
}

function assignX(id: string, state: LayoutState): number {
  const cached = state.xById.get(id);
  if (cached !== undefined) {
    return cached;
  }

  if (state.visiting.has(id)) {
    const x = state.nextX;
    state.nextX += NODE_WIDTH + H_GAP;
    state.xById.set(id, x);
    return x;
  }

  state.visiting.add(id);
  const children = state.childrenById.get(id) ?? [];

  let x: number;
  if (children.length === 0) {
    x = state.nextX;
    state.nextX += NODE_WIDTH + H_GAP;
  } else {
    const childXs = children.map((childId) => assignX(childId, state));
    const minX = Math.min(...childXs);
    const maxX = Math.max(...childXs);
    x = (minX + maxX) / 2;
  }

  state.visiting.delete(id);
  state.xById.set(id, x);
  return x;
}

export function layoutElements(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) {
    return { nodes, edges };
  }

  const nodeIds = nodes.map((node) => node.id);
  const childrenById = ensureChildrenMap(nodeIds, edges);
  const hasParent = new Set<string>();

  for (const [, targets] of childrenById.entries()) {
    for (const target of targets) {
      hasParent.add(target);
    }
  }

  const roots = nodeIds.filter((id) => !hasParent.has(id));
  if (roots.length === 0) {
    roots.push(nodeIds[0]);
  }

  const depthById = assignDepths(roots, childrenById);
  const state: LayoutState = {
    xById: new Map<string, number>(),
    depthById,
    childrenById,
    visiting: new Set<string>(),
    nextX: 0
  };

  for (const root of roots) {
    assignX(root, state);
    state.nextX += ROOT_GAP;
  }

  for (const id of nodeIds) {
    if (!state.xById.has(id)) {
      state.xById.set(id, state.nextX);
      state.nextX += NODE_WIDTH + H_GAP;
    }
    if (!state.depthById.has(id)) {
      state.depthById.set(id, 0);
    }
  }

  const layoutedNodes = nodes.map((node) => {
    const x = state.xById.get(node.id) ?? 0;
    const depth = state.depthById.get(node.id) ?? 0;

    return {
      ...node,
      position: {
        x,
        y: depth * (NODE_HEIGHT + V_GAP)
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top
    };
  });

  return { nodes: layoutedNodes, edges };
}
