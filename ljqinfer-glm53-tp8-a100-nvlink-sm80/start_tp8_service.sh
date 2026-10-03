#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/mnt/data/kw/anaconda3/bin/python
exec 9>/tmp/ljqinfer_glm53_service.lock
flock -n 9 || { echo "GLM53 service launcher already running" >&2; exit 1; }
ENGINE_PID=""
HTTP_PID=""
cleanup() {
  trap - EXIT INT TERM
  [ -z "$HTTP_PID" ] || kill -TERM "$HTTP_PID" 2>/dev/null || true
  [ -z "$ENGINE_PID" ] || kill -TERM "$ENGINE_PID" 2>/dev/null || true
  wait || true
}
trap cleanup EXIT INT TERM
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 -m server.engine_server &
ENGINE_PID=$!
ready=0
for ((i=0; i<900; i++)); do
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "Engine exited" >&2; exit 1; }
  if "$PY" -c 'import requests; r=requests.get("http://127.0.0.1:62001/health",timeout=2); assert r.json()["status"]=="ok"' 2>/dev/null; then ready=1; break; fi
  sleep 2
done
[ "$ready" = 1 ] || { echo "Engine readiness timeout" >&2; exit 1; }
"$PY" -m server.server &
HTTP_PID=$!
wait -n "$ENGINE_PID" "$HTTP_PID"
