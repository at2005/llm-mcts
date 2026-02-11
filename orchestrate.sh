#!/bin/bash
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
CONFIG="configs/config.json"
NUM_INFERENCE_GPUS=$(jq '.num_inference_gpus' "$CONFIG")
NUM_TRAINING_GPUS=$(jq '.num_training_gpus' "$CONFIG")
INFERENCE_BASE_PORT=$(jq '.inference_base_port' "$CONFIG")
REDIS_PORT=$(jq '.redis_port' "$CONFIG")

LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ── Tracking ────────────────────────────────────────────────────────
ALL_PIDS=()
INFERENCE_PIDS=()
SHUTTING_DOWN=0

# ── Cleanup & signal handling ───────────────────────────────────────
cleanup() {
    if [ "$SHUTTING_DOWN" -eq 1 ]; then return; fi
    SHUTTING_DOWN=1
    echo ""
    echo "=========================================="
    echo "  Shutting down all components..."
    echo "=========================================="

    # kill child processes in reverse order (training -> mcts -> inference -> redis)
    for pid in $(echo "${ALL_PIDS[@]}" | tac -s ' '); do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Sending SIGTERM to PID $pid..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # give processes time to exit gracefully
    echo "Waiting up to 15s for graceful shutdown..."
    local deadline=$((SECONDS + 15))
    local all_dead=0
    while [ $SECONDS -lt $deadline ] && [ $all_dead -eq 0 ]; do
        all_dead=1
        for pid in "${ALL_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_dead=0
                break
            fi
        done
        sleep 1
    done

    # force-kill any stragglers
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force-killing PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # stop redis
    redis-cli -p "$REDIS_PORT" shutdown nosave 2>/dev/null || true

    echo "Shutdown complete. Logs are in $LOG_DIR"
}

trap cleanup EXIT INT TERM

# ── Helpers ─────────────────────────────────────────────────────────
wait_for_port() {
    local port=$1
    local name=${2:-"service"}
    local max_attempts=120  # 10 minutes

    echo "[$name] Waiting for port $port..."
    local attempt=0
    while ! nc -z localhost "$port" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "[$name] TIMEOUT waiting for port $port after $((attempt * 5))s"
            exit 1
        fi
        sleep 5
    done
    echo "[$name] Port $port ready"
}

monitor_pid() {
    local pid=$1
    local name=$2
    # background monitor: if process dies unexpectedly, log and trigger shutdown
    (
        wait "$pid" 2>/dev/null
        exit_code=$?
        if [ "$SHUTTING_DOWN" -eq 0 ] && [ $exit_code -ne 0 ]; then
            echo ""
            echo "!!! [$name] CRASHED (PID $pid, exit code $exit_code) !!!"
            echo "!!! Check log: $LOG_DIR/${name}.log"
            echo "!!! Triggering shutdown..."
            kill -TERM $$ 2>/dev/null  # signal the main script
        fi
    ) &
}

# ── 1. Redis ────────────────────────────────────────────────────────
echo "Starting Redis on port $REDIS_PORT..."
redis-server --daemonize yes --port "$REDIS_PORT" --bind 0.0.0.0 \
    --logfile "$LOG_DIR/redis.log" --loglevel notice
wait_for_port "$REDIS_PORT" "redis"

# ── 2. Inference servers ────────────────────────────────────────────
for i in $(seq 0 $((NUM_INFERENCE_GPUS - 1))); do
    name="inference-$i"
    port=$((INFERENCE_BASE_PORT + i))
    echo "Starting $name on port $port..."
    RANK=$i uv run python inference.py \
        > "$LOG_DIR/${name}.log" 2>&1 &
    pid=$!
    INFERENCE_PIDS+=("$pid")
    ALL_PIDS+=("$pid")
    monitor_pid "$pid" "$name"
done

for i in $(seq 0 $((NUM_INFERENCE_GPUS - 1))); do
    wait_for_port $((INFERENCE_BASE_PORT + i)) "inference-$i"
done
echo "All $NUM_INFERENCE_GPUS inference servers ready"

# ── 3. MCTS (Rust) ─────────────────────────────────────────────────
echo "Starting mcts-rust..."
(cd mcts-rust && RUST_LOG=info cargo run --release) \
    > "$LOG_DIR/mcts.log" 2>&1 &
MCTS_PID=$!
ALL_PIDS+=("$MCTS_PID")
monitor_pid "$MCTS_PID" "mcts"
echo "mcts-rust started (PID $MCTS_PID)"

# ── 4. Training ────────────────────────────────────────────────────
echo "Starting training on $NUM_TRAINING_GPUS GPUs..."
uv run torchrun --nproc_per_node="$NUM_TRAINING_GPUS" train.py \
    > "$LOG_DIR/training.log" 2>&1 &
TRAIN_PID=$!
ALL_PIDS+=("$TRAIN_PID")
monitor_pid "$TRAIN_PID" "training"
echo "Training started (PID $TRAIN_PID)"

# ── Status ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  All components running"
echo "  Logs: $LOG_DIR/"
echo "    tail -f $LOG_DIR/inference-0.log"
echo "    tail -f $LOG_DIR/mcts.log"
echo "    tail -f $LOG_DIR/training.log"
echo "=========================================="
echo ""

# ── Wait ────────────────────────────────────────────────────────────
# Wait for any process to exit — cleanup trap handles the rest
wait -n "${ALL_PIDS[@]}" 2>/dev/null || true
