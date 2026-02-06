import { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  MiniMap,
  Node,
  NodeMouseHandler
} from "reactflow";
import { fetchTreeSnapshot, resetTree, resetWorkerTree } from "./api";
import { layoutElements } from "./layout";
import { TreeEdge, TreeNode, WorkerTree, WsMessage } from "./types";

type WorkerTreeState = {
  nodes: TreeNode[];
  edges: TreeEdge[];
};

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

function toWorkerMap(workers: WorkerTree[]): Record<string, WorkerTreeState> {
  const next: Record<string, WorkerTreeState> = {};
  for (const worker of workers) {
    next[worker.workerId] = {
      nodes: dedupeNodes(worker.nodes),
      edges: dedupeEdges(worker.edges)
    };
  }
  return next;
}

export default function App() {
  const [workerTrees, setWorkerTrees] = useState<Record<string, WorkerTreeState>>({});
  const [workerIds, setWorkerIds] = useState<string[]>([]);
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [socketState, setSocketState] = useState<"connecting" | "live" | "offline">("connecting");

  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let connectTimeout: ReturnType<typeof setTimeout> | null = null;
    let reconnectDelayMs = 1000;

    const clearReconnectTimer = () => {
      if (!reconnectTimer) {
        return;
      }
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    };

    const clearConnectTimeout = () => {
      if (!connectTimeout) {
        return;
      }
      clearTimeout(connectTimeout);
      connectTimeout = null;
    };

    const applySnapshot = (snapshot: { workers: WorkerTree[]; workerIds: string[] }) => {
      const nextWorkerTrees = toWorkerMap(snapshot.workers);
      const nextWorkerIds = snapshot.workerIds.length > 0 ? snapshot.workerIds : Object.keys(nextWorkerTrees).sort();

      setWorkerTrees(nextWorkerTrees);
      setWorkerIds(nextWorkerIds);
      setSelectedWorkerId((prev) => (prev && nextWorkerIds.includes(prev) ? prev : nextWorkerIds[0] ?? null));
    };

    fetchTreeSnapshot()
      .then((snapshot) => {
        if (!active) {
          return;
        }

        applySnapshot(snapshot);
      })
      .catch((cause) => {
        if (!active) {
          return;
        }
        setError(cause instanceof Error ? cause.message : "Failed to load initial tree");
      });

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsOverride = (import.meta.env.VITE_MCTS_WS_URL as string | undefined)?.trim();
    const defaultWsUrl =
      window.location.port === "5173"
        ? `${wsProtocol}://${window.location.hostname}:3001/ws`
        : `${wsProtocol}://${window.location.host}/ws`;
    const wsUrl = wsOverride || defaultWsUrl;

    const scheduleReconnect = () => {
      if (!active || reconnectTimer) {
        return;
      }

      setSocketState("connecting");
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, reconnectDelayMs);
      reconnectDelayMs = Math.min(reconnectDelayMs * 2, 15_000);
    };

    const connect = () => {
      if (!active) {
        return;
      }

      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
      }

      setSocketState("connecting");
      const ws = new WebSocket(wsUrl);
      socket = ws;

      clearConnectTimeout();
      connectTimeout = setTimeout(() => {
        if (socket === ws && ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
      }, 8_000);

      ws.onopen = () => {
        if (!active || socket !== ws) {
          return;
        }

        clearConnectTimeout();
        clearReconnectTimer();
        reconnectDelayMs = 1000;
        setSocketState("live");
      };

      ws.onclose = () => {
        if (!active || socket !== ws) {
          return;
        }

        clearConnectTimeout();
        socket = null;
        setSocketState("offline");
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!active || socket !== ws) {
          return;
        }

        ws.close();
      };

      ws.onmessage = (event) => {
        if (!active || socket !== ws) {
          return;
        }

        try {
          const message = JSON.parse(event.data) as WsMessage;

          if (message.type === "tree_snapshot") {
            applySnapshot(message);
            setSelectedNodeId(null);
            return;
          }

          if (message.type === "node_added") {
            setWorkerTrees((prev) => {
              const current = prev[message.workerId] ?? { nodes: [], edges: [] };
              const nextEdges = message.edge ? dedupeEdges([...current.edges, message.edge]) : current.edges;

              return {
                ...prev,
                [message.workerId]: {
                  nodes: dedupeNodes([...current.nodes, message.node]),
                  edges: nextEdges
                }
              };
            });

            setWorkerIds((prev) => {
              if (prev.includes(message.workerId)) {
                return prev;
              }
              return [...prev, message.workerId].sort();
            });

            setSelectedWorkerId((prev) => prev ?? message.workerId);
            return;
          }

          if (message.type === "tree_reset") {
            setWorkerTrees({});
            setWorkerIds([]);
            setSelectedWorkerId(null);
            setSelectedNodeId(null);
            return;
          }

          if (message.type === "worker_tree_reset") {
            setWorkerTrees((prev) => {
              if (!(message.workerId in prev)) {
                return prev;
              }

              const next = { ...prev };
              delete next[message.workerId];
              return next;
            });

            setWorkerIds((prev) => {
              const nextWorkerIds = prev.filter((workerId) => workerId !== message.workerId);
              setSelectedWorkerId((current) => {
                if (current && nextWorkerIds.includes(current)) {
                  return current;
                }
                return nextWorkerIds[0] ?? null;
              });
              return nextWorkerIds;
            });

            setSelectedNodeId(null);
          }
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Failed to parse websocket payload");
        }
      };
    };

    const handleOnline = () => {
      if (!active) {
        return;
      }

      reconnectDelayMs = 1000;
      clearReconnectTimer();
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
      }
      connect();
    };

    window.addEventListener("online", handleOnline);
    connect();

    return () => {
      active = false;
      window.removeEventListener("online", handleOnline);
      clearReconnectTimer();
      clearConnectTimeout();
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close();
      }
    };
  }, []);

  const activeTree = useMemo<WorkerTreeState>(() => {
    if (!selectedWorkerId) {
      return { nodes: [], edges: [] };
    }

    return workerTrees[selectedWorkerId] ?? { nodes: [], edges: [] };
  }, [selectedWorkerId, workerTrees]);

  const nodeIndex = useMemo(() => {
    return new Map(activeTree.nodes.map((node) => [node.id, node]));
  }, [activeTree.nodes]);

  useEffect(() => {
    if (selectedNodeId && !nodeIndex.has(selectedNodeId)) {
      setSelectedNodeId(null);
    }
  }, [nodeIndex, selectedNodeId]);

  const selectedNode = selectedNodeId ? nodeIndex.get(selectedNodeId) ?? null : null;

  const totalStats = useMemo(() => {
    let totalNodes = 0;
    let totalEdges = 0;

    for (const workerId of workerIds) {
      const tree = workerTrees[workerId];
      if (!tree) {
        continue;
      }
      totalNodes += tree.nodes.length;
      totalEdges += tree.edges.length;
    }

    return { totalNodes, totalEdges };
  }, [workerIds, workerTrees]);

  const flowNodes = useMemo<Node[]>(() => {
    return activeTree.nodes.map((node) => {
      const snippet = node.decodedState.trim();
      const selected = node.id === selectedNodeId;

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
        style: selected
          ? {
              border: "2px solid var(--accent)",
              boxShadow: "0 0 0 2px color-mix(in srgb, var(--accent) 20%, transparent)"
            }
          : undefined,
        draggable: false,
        selectable: true
      };
    });
  }, [activeTree.nodes, selectedNodeId]);

  const flowEdges = useMemo<Edge[]>(() => {
    return activeTree.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: true
    }));
  }, [activeTree.edges]);

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    return layoutElements(flowNodes, flowEdges);
  }, [flowEdges, flowNodes]);

  const onNodeClick: NodeMouseHandler = (_event, node) => {
    setSelectedNodeId(node.id);
  };

  return (
    <div className="app-shell">
      <div className="canvas-shell">
        <div className="toolbar">
          <div className="toolbar-title">MCTS Live Tree</div>

          <label className="worker-picker">
            Worker
            <select
              value={selectedWorkerId ?? ""}
              onChange={(event) => {
                const nextWorkerId = event.target.value || null;
                setSelectedWorkerId(nextWorkerId);
                setSelectedNodeId(null);
              }}
              disabled={workerIds.length === 0}
            >
              {workerIds.length === 0 ? <option value="">No workers</option> : null}
              {workerIds.map((workerId) => (
                <option key={workerId} value={workerId}>
                  {workerId}
                </option>
              ))}
            </select>
          </label>

          <div className={`status status-${socketState}`}>{socketState}</div>
          <button
            type="button"
            disabled={!selectedWorkerId}
            onClick={() => {
              if (!selectedWorkerId) {
                return;
              }

              resetWorkerTree(selectedWorkerId).catch((cause) => {
                setError(cause instanceof Error ? cause.message : "Failed to reset selected worker tree");
              });
            }}
          >
            Reset Worker
          </button>
          <button
            type="button"
            onClick={() => {
              resetTree().catch((cause) => {
                setError(cause instanceof Error ? cause.message : "Failed to reset tree");
              });
            }}
          >
            Reset All
          </button>
        </div>

        <div className="canvas-wrap">
          <ReactFlow
            nodes={layoutedNodes}
            edges={layoutedEdges}
            fitView
            onNodeClick={onNodeClick}
            onPaneClick={() => setSelectedNodeId(null)}
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
          <h2>Selected Node</h2>
          {selectedNode ? (
            <>
              <p>
                <strong>Worker:</strong> {selectedNode.workerId}
              </p>
              <p>
                <strong>Node:</strong> {selectedNode.id}
              </p>
              <p>
                <strong>Parent:</strong> {selectedNode.parentId ?? "(root)"}
              </p>
              <p>
                <strong>Decoded state:</strong>
              </p>
              <pre>{selectedNode.decodedState || "(empty)"}</pre>
              <p>
                <strong>Visits:</strong> {selectedNode.visits}
              </p>
              <p>
                <strong>Value:</strong> {selectedNode.value}
              </p>
              <p>
                <strong>Raw contents:</strong>
              </p>
              <pre>{JSON.stringify(selectedNode.contents)}</pre>
              <p>
                <strong>Created:</strong> {selectedNode.createdAt}
              </p>
            </>
          ) : (
            <p>{selectedWorkerId ? "Click a node to inspect its properties." : "Select a worker to view its tree."}</p>
          )}
        </section>

        <section>
          <h2>POST /api/nodes</h2>
          <p>Send each node as it is created during MCTS.</p>
          <pre>{`{
  "workerId": 0,
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
  -d '{"workerId":0,"id":0,"parentId":18446744073709551615,"contents":[128000,271],"visits":1,"value":0.0}'`}</pre>
        </section>

        <section>
          <h2>Tree Stats</h2>
          <p>
            <strong>Workers:</strong> {workerIds.length}
          </p>
          <p>
            <strong>Total Nodes:</strong> {totalStats.totalNodes}
          </p>
          <p>
            <strong>Total Edges:</strong> {totalStats.totalEdges}
          </p>
          {selectedWorkerId ? (
            <>
              <p>
                <strong>Selected Worker:</strong> {selectedWorkerId}
              </p>
              <p>
                <strong>Worker Nodes:</strong> {activeTree.nodes.length}
              </p>
              <p>
                <strong>Worker Edges:</strong> {activeTree.edges.length}
              </p>
            </>
          ) : null}
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
