# MCTS Real-Time UI

React + Express app to visualize Monte Carlo Tree Search as your runner posts nodes.

## Features

- `POST /api/nodes` endpoint for ingesting nodes in real time
- WebSocket stream at `/ws` for live UI updates
- DAG rendering of nodes and edges
- Click a node to inspect its decoded state and properties
- `POST /api/reset` endpoint to clear the tree

## Requirements

- Bun 1.2+
- A Llama 3 `tokenizer.json` file
- A matching Llama 3 `tokenizer_config.json` file

Place tokenizer files at:

- `tokenizer/tokenizer.json`
- `tokenizer/tokenizer_config.json`

Or set:

```bash
export LLAMA3_TOKENIZER_PATH=/absolute/path/to/tokenizer.json
export LLAMA3_TOKENIZER_CONFIG_PATH=/absolute/path/to/tokenizer_config.json
```

## Run

```bash
bun install
bun run dev
```

- UI: `http://localhost:5173`
- API: `http://localhost:3001`

Optional client overrides (useful for SSH tunnels / non-proxied setups):

```bash
export VITE_MCTS_API_BASE=http://localhost:3001
export VITE_MCTS_WS_URL=ws://localhost:3001/ws
```

## API

### `POST /api/nodes`

Body:

```json
{
  "workerId": 0,
  "id": 12,
  "parentId": 3,
  "contents": [128000, 271, 9906],
  "visits": 9,
  "value": 0.81
}
```

Rules:

- `workerId` is required
- `id` must be unique
- `id` uniqueness is scoped per `workerId` (same node id can be reused across different workers)
- root can be represented with `parentId = usize::MAX` (for Rust runners)
- non-root `parentId` must already exist

### `GET /api/tree`
Returns worker-partitioned trees (`workers[]` plus `workerIds[]`).

### `POST /api/reset`
Clears all nodes.

### `POST /api/reset-tree`
Alias for resetting all worker trees.

### `POST /api/reset-tree/:workerId`
Resets only one worker tree.

### `POST /api/workers/:workerId/reset`
Alternate path for resetting one worker tree.

### `GET /api/health`
Health check endpoint.

## Minimal runner example

```bash
curl -X POST http://localhost:3001/api/nodes \
  -H 'content-type: application/json' \
  -d '{"workerId":0,"id":0,"parentId":18446744073709551615,"contents":[128000,271],"visits":1,"value":0.0}'

curl -X POST http://localhost:3001/api/nodes \
  -H 'content-type: application/json' \
  -d '{"workerId":0,"id":1,"parentId":0,"contents":[9906,374],"visits":3,"value":1.4}'

curl -X POST http://localhost:3001/api/nodes \
  -H 'content-type: application/json' \
  -d '{"workerId":1,"id":0,"parentId":18446744073709551615,"contents":[128000,271],"visits":1,"value":0.0}'

curl -X POST http://localhost:3001/api/reset-tree/1

curl -X POST http://localhost:3001/api/workers/0/reset
```
