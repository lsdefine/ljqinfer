#!/usr/bin/env bash
# Two-stage serving stack for ljqinfer-dsv41f (TP8 on Atlas 800 A2).
#   ./scripts/serve.sh start|stop|status
# Stage 1: torchrun engine, 8 ranks; rank0 exposes the private RPC on 127.0.0.1:62001.
# Stage 2: protocol front-end (Anthropic /v1/messages + OpenAI /v1/chat/completions) on 0.0.0.0:8000.
# NOTE: the engine MUST be launched as a module (-m strategy.decode_worker).  Launching the
# file by path puts strategy/ itself on sys.path, so "import strategy.cold_kv" fails with
# "'strategy' is not a package".
set -uo pipefail
ROOT=/data/ljqinfer_dsv41f_tp8
LOGS=/data/logs
ENGINE_LOG=$LOGS/engine_serve.log
FRONT_LOG=$LOGS/api_front.log
ENGINE_URL=http://127.0.0.1:62001/health
FRONT_PORT=8000

start_engine() {
  mkdir -p "$LOGS"
  cd "$ROOT" || exit 1
  TASK_QUEUE_ENABLE=${LJQ_TQE:-1} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="$ROOT" \
    setsid nohup torchrun --nproc_per_node=8 --master_port=29571 \
    -m strategy.decode_worker > "$ENGINE_LOG" 2>&1 < /dev/null &
  echo "engine launching, log: $ENGINE_LOG"
  for _ in $(seq 1 60); do
    sleep 5
    grep -q ENGINE_READY "$ENGINE_LOG" && { echo "ENGINE_READY"; return 0; }
  done
  echo "engine failed to report ENGINE_READY; see $ENGINE_LOG"; return 1
}

start_front() {
  cd "$ROOT" || exit 1
  PYTHONPATH="$ROOT" setsid nohup python3 -m server.server > "$FRONT_LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 20); do
    sleep 2
    curl -sf -m 2 "http://127.0.0.1:$FRONT_PORT/health" > /dev/null && { echo "front ready on :$FRONT_PORT"; return 0; }
  done
  echo "front-end failed to answer /health; see $FRONT_LOG"; return 1
}

case "${1:-status}" in
  start) start_engine && start_front ;;
  stop)
    pkill -f '[s]trategy.decode_worker'
    pkill -f '[s]erver.server'
    echo stopped ;;
  status)
    echo -n "engine: "; curl -sf -m 2 "$ENGINE_URL" || echo down; echo
    echo -n "front : "; curl -sf -m 2 "http://127.0.0.1:$FRONT_PORT/health" || echo down; echo
    pgrep -fc '[s]trategy.decode_worker' | sed 's/^/engine ranks: /' ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
