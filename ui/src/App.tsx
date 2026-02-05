import { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  MiniMap,
  Node,
  NodeMouseHandler
} from "reactflow";
import { fetchTreeSnapshot, resetTree } from "./api";
import { layoutElements } from "./layout";
import { TreeEdge, TreeNode, WsMessage } from "./types";

function dedupeNodes(nodes: TreeNode[]): TreeNode[] {
  const unique = new Map<string, TreeNode>();
  for (const node of nodes) {
    unique.set(node.id, node);
  }
  return [...unique.values()];
}

function dedupeEdges(edges: TreeEdge[]): TreeEdge[] {
  const unique = new Map<string, TreeEdge>();
  for (const edge of edges) {
    unique.set(edge.id, edge);
  }
  return [...unique.values()];
}

export default function App() {
  const [nodes, setNodes] = useState<TreeNode[]>([]);
  const [edges, setEdges] = useState<TreeEdge[]>([]);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [socketState, setSocketState] = useState<"connecting" | "live" | "offline">("connecting");

  useEffect(() => {
    let active = true;

    fetchTreeSnapshot()
      .then((snapshot) => {
        if (!active) {
          return;
        }
        setNodes(dedupeNodes(snapshot.nodes));
        setEdges(dedupeEdges(snapshot.edges));
      })
      .catch((cause) => {
        if (!active) {
          return;
        }
        setError(cause instanceof Error ? cause.message : "Failed to load initial tree");
      });

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${wsProtocol}://${window.location.host}/ws`);

    socket.onopen = () => {
      if (active) {
        setSocketState("live");
      }
    };

    socket.onclose = () => {
      if (active) {
        setSocketState("offline");
      }
    };

    socket.onerror = () => {
      if (active) {
        setSocketState("offline");
      }
    };

    socket.onmessage = (event) => {
      if (!active) {
        return;
      }

      try {
        const message = JSON.parse(event.data) as WsMessage;

        if (message.type === "tree_snapshot") {
          setNodes(dedupeNodes(message.nodes));
          setEdges(dedupeEdges(message.edges));
          return;
        }

        if (message.type === "node_added") {
          setNodes((prev) => dedupeNodes([...prev, message.node]));
          if (message.edge !== null) {
            const edge = message.edge;
            setEdges((prev) => dedupeEdges([...prev, edge]));
          }
          return;
        }

        if (message.type === "tree_reset") {
          setNodes([]);
          setEdges([]);
        }
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Failed to parse websocket payload");
      }
    };

    return () => {
      active = false;
      socket.close();
    };
  }, []);

  const nodeIndex = useMemo(() => {
    return new Map(nodes.map((node) => [node.id, node]));
  }, [nodes]);

  const hoveredNode = hoveredNodeId ? nodeIndex.get(hoveredNodeId) ?? null : null;

  const flowNodes = useMemo<Node[]>(() => {
    return nodes.map((node) => {
      const snippet = node.decodedContent.trim();
      return {
        id: node.id,
        data: {
          label: (
            <div className="node-card">
              <div className="node-id">{node.id}</div>
              <div className="node-contents">{snippet.length > 70 ? `${snippet.slice(0, 70)}...` : snippet || "(empty)"}</div>
            </div>
          )
        },
        position: { x: 0, y: 0 },
        draggable: false,
        selectable: true
      };
    });
  }, [nodes]);

  const flowEdges = useMemo<Edge[]>(() => {
    return edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: true
    }));
  }, [edges]);

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    return layoutElements(flowNodes, flowEdges);
  }, [flowEdges, flowNodes]);

  const onNodeEnter: NodeMouseHandler = (_event, node) => {
    setHoveredNodeId(node.id);
  };

  const onNodeLeave: NodeMouseHandler = () => {
    setHoveredNodeId(null);
  };

  return (
    <div className="app-shell">
      <div className="canvas-shell">
        <div className="toolbar">
          <div className="toolbar-title">MCTS Live Tree</div>
          <div className={`status status-${socketState}`}>{socketState}</div>
          <button
            type="button"
            onClick={() => {
              resetTree().catch((cause) => {
                setError(cause instanceof Error ? cause.message : "Failed to reset tree");
              });
            }}
          >
            Reset Tree
          </button>
        </div>

        <div className="canvas-wrap">
          <ReactFlow
            nodes={layoutedNodes}
            edges={layoutedEdges}
            fitView
            onNodeMouseEnter={onNodeEnter}
            onNodeMouseLeave={onNodeLeave}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={18} size={1} />
            <MiniMap pannable zoomable />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </div>

      <aside className="side-panel">
        <section>
          <h2>Hover State</h2>
          {hoveredNode ? (
            <>
              <p>
                <strong>Node:</strong> {hoveredNode.id}
              </p>
              <p>
                <strong>Parent:</strong> {hoveredNode.parentId ?? "(root)"}
              </p>
              <p>
                <strong>Cumulative detokenized text:</strong>
              </p>
              <pre>{hoveredNode.cumulativeText || "(empty)"}</pre>
              <p>
                <strong>Visits:</strong> {hoveredNode.visits}
              </p>
              <p>
                <strong>Value:</strong> {hoveredNode.value}
              </p>
              <p>
                <strong>Raw contents:</strong>
              </p>
              <pre>{JSON.stringify(hoveredNode.contents)}</pre>
            </>
          ) : (
            <p>Hover a node to inspect cumulative decoded state.</p>
          )}
        </section>

        <section>
          <h2>POST /api/nodes</h2>
          <p>Send each node as it is created during MCTS.</p>
          <pre>{`{
  "id": 17,
  "parentId": 4,
  "contents": [128000, 271, 9906],
  "visits": 12,
  "value": 8.5
}`}</pre>
          <p>
            <strong>cURL:</strong>
          </p>
          <pre>{`curl -X POST http://localhost:3001/api/nodes \\
  -H 'content-type: application/json' \\
  -d '{"id":0,"parentId":18446744073709551615,"contents":[128000,271],"visits":1,"value":0.0}'`}</pre>
        </section>

        <section>
          <h2>Tree Stats</h2>
          <p>
            <strong>Nodes:</strong> {nodes.length}
          </p>
          <p>
            <strong>Edges:</strong> {edges.length}
          </p>
        </section>

        {error ? (
          <section>
            <h2>Error</h2>
            <pre>{error}</pre>
          </section>
        ) : null}
      </aside>
    </div>
  );
}
