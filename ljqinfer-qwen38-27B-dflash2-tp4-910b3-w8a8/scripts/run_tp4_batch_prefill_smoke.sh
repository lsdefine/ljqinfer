#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_tp4.sh"
P="${PYTHON_BIN:-python3}"
export LJQ_HCCL_ROOT="${LJQ_HCCL_ROOT:-/dev/shm/qtp4_batch_prefill_root.bin}"
rm -f "$LJQ_HCCL_ROOT" /tmp/qtp4_batch_prefill_rank*.log
pids=()
for rank in 0 1 2 3; do
  PYTHONUNBUFFERED=1 "$P" scripts/smoke_tp4_batch_prefill.py --rank "$rank" \
    >"/tmp/qtp4_batch_prefill_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
for rank in 0 1 2 3; do
  echo "===rank${rank}==="
  cat "/tmp/qtp4_batch_prefill_rank${rank}.log"
done
exit "$rc"
