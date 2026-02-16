#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

cleanup() {
  echo
  echo "Ctrl-C caught -> stopping all ranks..."
  jobs -pr | xargs -r kill 2>/dev/null || true   # terminate all background jobs (pipelines too)
  wait || true
  exit 130
}
trap cleanup INT TERM

for i in $(seq 0 6); do
  log="logs/rank_${i}.log"
  PYTHONUNBUFFERED=1 RANK="$i" uv run inference.py 2>&1 \
    | tee "$log" \
    | sed -u "s/^/[RANK $i] /" &
done

wait

