#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_tp4.sh"
LAYERS="${LAYERS:-64}"
ROOT="/dev/shm/qtp4_engine${LAYERS}_root.bin"
rm -f "$ROOT" /tmp/qtp4_engine${LAYERS}_rank*.log
DEVICE_BASE="${DEVICE_BASE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
pids=()
for rank in 0 1 2 3; do
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/smoke_tp4_engine.py \
    --rank "$rank" --device-base "$DEVICE_BASE" --layers "$LAYERS" \
    >"/tmp/qtp4_engine${LAYERS}_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
for rank in 0 1 2 3; do
  echo "===rank${rank}==="
  cat "/tmp/qtp4_engine${LAYERS}_rank${rank}.log"
done
exit "$rc"
