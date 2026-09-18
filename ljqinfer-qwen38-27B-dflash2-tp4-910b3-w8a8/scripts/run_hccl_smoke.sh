#!/usr/bin/env bash
set -euo pipefail
rm -f /dev/shm/qtp4_smoke_root.bin /tmp/qtp4_rank*.log
DEVICE_BASE="${DEVICE_BASE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
pids=()
for rank in 0 1 2 3; do
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/smoke_hccl_tp4.py --rank "$rank" \
    --device-base "$DEVICE_BASE" \
    >"/tmp/qtp4_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
for rank in 0 1 2 3; do
  echo "===rank${rank}==="
  cat "/tmp/qtp4_rank${rank}.log"
done
exit "$rc"
