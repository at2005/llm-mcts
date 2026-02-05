# MCTS Real-Time UI

React + Express app to visualize Monte Carlo Tree Search as your runner posts nodes.

## Features

- `POST /api/nodes` endpoint for ingesting nodes in real time
- WebSocket stream at `/ws` for live UI updates
- DAG rendering of nodes and edges
- Hover on a node to inspect cumulative state (detokenized with Llama 3 tokenizer)
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
  "id": 12,
  "parentId": 3,
  "contents": [128000, 271, 9906],
  "visits": 9,
  "value": 0.81
}
```

Rules:

- `id` must be unique
- root can be represented with `parentId = usize::MAX` (for Rust runners)
- non-root `parentId` must already exist

### `GET /api/tree`
Returns current nodes and edges.

### `POST /api/reset`
Clears all nodes.

### `GET /api/health`
Health check endpoint.

## Minimal runner example

```bash
curl -X POST http://localhost:3001/api/nodes \
  -H 'content-type: application/json' \
  -d '{"id":0,"parentId":18446744073709551615,"contents":[128000,271],"visits":1,"value":0.0}'

curl -X POST http://localhost:3001/api/nodes \
  -H 'content-type: application/json' \
  -d '{"id":1,"parentId":0,"contents":[9906,374],"visits":3,"value":1.4}'
```
