#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./reset.sh

cleanup() {
  echo
  echo "Ctrl-C caught -> stopping all ranks..."
  pids="$(jobs -pr)"
  if [ -n "$pids" ]; then
    # First try graceful termination for all background jobs/pipelines.
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 0.2
    # Then force kill any stragglers (e.g. subprocesses that ignore TERM).
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
  # Extra safety: terminate any direct child processes of this launcher.
  pkill -P $$ 2>/dev/null || true
  wait || true
  exit 130
}
trap cleanup INT TERM

WORLD_SIZE=6

for i in $(seq 2 7); do
  shard_rank=$((i - 2))
  log="logs/best_of_n_rank_${i}.log"
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$i" \
  RANK="$shard_rank" \
  WORLD_SIZE="$WORLD_SIZE" \
  SGLANG_PORT="$((30000 + i))" \
  uv run best_of_n.py 2>&1 \
    | tee "$log" \
    | sed -u "s/^/[RANK $i | SHARD $shard_rank] /" &
done

wait
