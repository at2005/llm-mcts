#!/bin/bash

export NUM_INFERENCE_GPUS=4
export NUM_TRAINING_GPUS=4

# get redis up 
redis-server --daemonize yes --port 6379 --bind 0.0.0.0

# run inference
uv run torchrun --nproc_per_node=$NUM_INFERENCE_GPUS inference.py &

uv run cargo run --release --bin mcts &

uv run torchrun --nproc_per_node=$NUM_TRAINING_GPUS train.py &