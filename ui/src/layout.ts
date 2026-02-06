import dagre from "dagre";
import { Edge, Node, Position } from "reactflow";

const NODE_WIDTH = 180;
const NODE_HEIGHT = 56;

export function layoutElements(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: "TB", nodesep: 24, ranksep: 56, marginx: 20, marginy: 20 });

  nodes.forEach((node) => {
    graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target);
  });

  dagre.layout(graph);

  const positioned = nodes.map((node) => {
    const pos = graph.node(node.id);
    return {
      ...node,
      position: {
        x: (pos?.x ?? 0) - NODE_WIDTH / 2,
        y: (pos?.y ?? 0) - NODE_HEIGHT / 2
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top
    };
  });

  const minX = positioned.reduce((acc, node) => Math.min(acc, node.position.x), Number.POSITIVE_INFINITY);
  const minY = positioned.reduce((acc, node) => Math.min(acc, node.position.y), Number.POSITIVE_INFINITY);

  const layoutedNodes = positioned.map((node) => ({
    ...node,
    position: {
      x: node.position.x - (Number.isFinite(minX) ? minX : 0),
      y: node.position.y - (Number.isFinite(minY) ? minY : 0)
    }
  }));

  return { nodes: layoutedNodes, edges };
}
