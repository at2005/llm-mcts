import http from "node:http";
import cors from "cors";
import express from "express";
import { WebSocketServer } from "ws";
import { decodeTokens, loadLlamaTokenizer } from "./tokenizer";
import { TreeStore } from "./tree-store";
import { NodePayload, WsMessage } from "./types";

const PORT = Number(process.env.MCTS_UI_PORT ?? 3001);

function isIntegerArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => Number.isInteger(item));
}

function isNodeIdLike(value: unknown): value is string | number {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  return typeof value === "number" && Number.isInteger(value) && Number.isFinite(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeNodeId(value: string | number): string {
  return typeof value === "string" ? value.trim() : String(value);
}

function isRootParentId(value: string | number | null | undefined): boolean {
  if (value === null || value === undefined) {
    return true;
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return true;
    }

    if (!Number.isSafeInteger(value)) {
      // Rust usize::MAX on 64-bit will exceed JS safe integer range.
      return true;
    }

    return value < 0 || value === 4_294_967_295;
  }

  const normalized = value.trim();
  if (!normalized) {
    return true;
  }

  if (normalized === "-1" || normalized === "4294967295" || normalized === "18446744073709551615") {
    return true;
  }

  const maybeNumeric = Number(normalized);
  if (!Number.isNaN(maybeNumeric)) {
    return isRootParentId(maybeNumeric);
  }

  return false;
}

function isNodePayload(value: unknown): value is NodePayload {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<NodePayload>;
  const hasId = isNodeIdLike(candidate.id);
  const hasParent = candidate.parentId === null || isNodeIdLike(candidate.parentId);
  const hasContents = isIntegerArray(candidate.contents);
  const hasVisits = isFiniteNumber(candidate.visits);
  const hasValue = isFiniteNumber(candidate.value);

  return hasId && hasParent && hasContents && hasVisits && hasValue;
}

async function start() {
  const tokenizer = await loadLlamaTokenizer();
  const store = new TreeStore();

  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "1mb" }));

  const server = http.createServer(app);
  const wss = new WebSocketServer({ server, path: "/ws" });

  const broadcast = (payload: WsMessage): void => {
    const encoded = JSON.stringify(payload);

    for (const client of wss.clients) {
      if (client.readyState === 1) {
        client.send(encoded);
      }
    }
  };

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, now: new Date().toISOString() });
  });

  app.get("/api/tree", (_req, res) => {
    res.json(store.getSnapshot());
  });

  app.post("/api/nodes", (req, res) => {
    if (!isNodePayload(req.body)) {
      res.status(400).json({
        error: "Invalid payload. Required: { id, parentId, contents: integer[], visits: number, value: number }"
      });
      return;
    }

    const payload = req.body;
    const nodeId = normalizeNodeId(payload.id);

    if (store.getNode(nodeId)) {
      res.status(409).json({ error: `Node '${nodeId}' already exists.` });
      return;
    }

    const parentRaw = payload.parentId;
    const parentId = isRootParentId(parentRaw) ? null : normalizeNodeId(parentRaw as string | number);
    const parent = parentId ? store.getNode(parentId) : undefined;
    if (parentId && !parent) {
      res.status(404).json({ error: `Parent node '${parentId}' not found.` });
      return;
    }

    const decodedState = decodeTokens(tokenizer, payload.contents);
    const savedNode = store.addNode({ ...payload, id: nodeId, parentId }, decodedState);
    const edge = savedNode.parentId ? store.buildEdge(savedNode.parentId, savedNode.id) : null;

    broadcast({ type: "node_added", node: savedNode, edge });

    res.status(201).json({ node: savedNode, edge });
  });

  app.post("/api/reset", (_req, res) => {
    store.reset();
    broadcast({ type: "tree_reset" });
    res.status(204).send();
  });

  wss.on("connection", (socket) => {
    socket.send(
      JSON.stringify({
        type: "tree_snapshot",
        ...store.getSnapshot()
      } satisfies WsMessage)
    );
  });

  server.listen(PORT, () => {
    console.log(`MCTS UI server listening on http://localhost:${PORT}`);
  });
}

start().catch((error) => {
  console.error(error);
  process.exit(1);
});
