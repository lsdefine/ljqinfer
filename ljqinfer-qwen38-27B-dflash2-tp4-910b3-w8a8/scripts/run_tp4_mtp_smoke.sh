#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_tp4.sh"
P="${PYTHON_BIN:-python3}"; D="${DEVICE_BASE:-4}"
rm -f /dev/shm/qtp4_mtp_root.bin /tmp/qtp4_mtp_rank*.log
pids=(); for r in 0 1 2 3; do
 PYTHONUNBUFFERED=1 "$P" scripts/smoke_tp4_mtp.py --rank "$r" --device-base "$D" >"/tmp/qtp4_mtp_rank$r.log" 2>&1 & pids+=("$!")
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for r in 0 1 2 3; do echo "===rank$r==="; cat "/tmp/qtp4_mtp_rank$r.log"; done
exit "$rc"
