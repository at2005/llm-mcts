#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  echo "Caught signal, killing process group..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup INT TERM EXIT

./run_inference.sh &

(
  cd mcts-rust
  cargo run --release
) > logs/mcts-rust.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc-per-node 4 train.py