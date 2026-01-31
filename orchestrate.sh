#!/bin/bash
set -e

export NUM_INFERENCE_GPUS=4
export NUM_TRAINING_GPUS=4

INFERENCE_BASE_PORT=50051

wait_for_port() {
    local port=$1
    local max_attempts=120  # 10 minutes max
    local attempt=0

    echo "Waiting for port $port..."
    while ! nc -z localhost "$port" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "Timeout waiting for port $port"
            exit 1
        fi
        sleep 5
    done
    echo "Port $port is ready"
}

# get redis up
redis-server --daemonize yes --port 6379 --bind 0.0.0.0
wait_for_port 6379

# run inference servers (one per GPU, no torchrun)
INFERENCE_PIDS=()
for i in $(seq 0 $((NUM_INFERENCE_GPUS - 1))); do
    echo "Starting inference server for rank $i..."
    RANK=$i uv run python inference.py &
    INFERENCE_PIDS+=($!)
done

# wait for all inference servers to be ready
for i in $(seq 0 $((NUM_INFERENCE_GPUS - 1))); do
    wait_for_port $((INFERENCE_BASE_PORT + i))
done
echo "All inference servers ready"

echo "Starting mcts-rust..."
(cd mcts-rust && cargo run --release) &
MCTS_PID=$!
echo "mcts-rust started with PID $MCTS_PID"

echo "Starting training..."
uv run torchrun --nproc_per_node=$NUM_TRAINING_GPUS train.py &
TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"

# wait for all background processes
wait "${INFERENCE_PIDS[@]}" $MCTS_PID $TRAIN_PID
