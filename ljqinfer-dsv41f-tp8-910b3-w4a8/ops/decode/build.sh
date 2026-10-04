#!/bin/bash
# AscendC kernel build (910B3 = dav-c220). Usage: bash build.sh [<src.cpp> <out.so>] (no args: all decode libraries)
set -euo pipefail
A=${ASCEND_HOME_PATH:?source the CANN environment first}
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
# Explicit build-time convenience; never invoked by the runtime.
if [ "$#" -eq 0 ]; then
  ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
  for name in norm attention gemm window hc_project w4_prepare wo_b_cube draft_vocab_cube markov_cube draft_down_cube source_cube; do
    bash "$ROOT/build.sh" "$ROOT/$name.cpp" "$ROOT/libdecode_$name.so"
  done
  bash "$ROOT/build.sh" --tiling "$ROOT"
  bash "$ROOT/build.sh" --draft-vocab-tiling "$ROOT"
  bash "$ROOT/build.sh" --markov-tiling "$ROOT"
  bash "$ROOT/build.sh" --draft-down-tiling "$ROOT"
  exit 0
fi
# Host-only build step; runtime never compiles or generates tiling.
if [ "${1:-}" = "--tiling" ] || [ "${1:-}" = "--draft-vocab-tiling" ] || [ "${1:-}" = "--markov-tiling" ] || [ "${1:-}" = "--draft-down-tiling" ]; then
  TILING_SOURCE=wo_b_tiling.cpp
  PREFIX=
  ROWS="6 12 18 24"
  if [ "$1" = "--draft-vocab-tiling" ]; then
    TILING_SOURCE=draft_vocab_tiling.cpp
    PREFIX=draft_vocab_
  fi
  if [ "$1" = "--markov-tiling" ]; then
    TILING_SOURCE=markov_tiling.cpp
    PREFIX=markov_
    ROWS="1 2 3 4"
  fi
  if [ "$1" = "--draft-down-tiling" ]; then
    TILING_SOURCE=draft_down_tiling.cpp
    PREFIX=draft_down_
  fi
  OUT=${2:?tiling output directory required}
  mkdir -p "$OUT"
  OUT=$(cd -- "$OUT" && pwd)
  WORK=$(mktemp -d "$OUT/.wo_b_tiling.XXXXXX")
  trap 'rm -rf -- "$WORK"' EXIT
  export LD_LIBRARY_PATH="$A/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/usr/bin/g++ -std=c++17 -O2 "$ROOT/$TILING_SOURCE" -o "$WORK/host_tiling" -I${A}/aarch64-linux/asc/include/adv_api/matmul -I${A}/aarch64-linux/asc -I${A}/aarch64-linux/tikcpp/tikcfw -I${A}/include -L${A}/lib64 -ltiling_api -lplatform -ldl -lunified_dlog -lc_sec -lregister
  (cd -- "$WORK" && ./host_tiling)
  for m in $ROWS; do
    test "$(wc -c < "$WORK/${PREFIX}tiling_m$m.bin")" -eq 200
  done
  for m in $ROWS; do
    mv -- "$WORK/${PREFIX}tiling_m$m.bin" "$OUT/${PREFIX}tiling_m$m.bin"
  done
  exit 0
fi
SRC=${1:?source.cpp required}
OUT=${2:?output.so required}
ARCH=dav-c220
# Vector-only modules require an explicit AIV target with direct bisheng.
if [ "$(basename -- "$SRC")" = "norm.cpp" ] || [ "$(basename -- "$SRC")" = "attention.cpp" ] || [ "$(basename -- "$SRC")" = "hc_project.cpp" ] || [ "$(basename -- "$SRC")" = "w4_prepare.cpp" ]; then
  ARCH=dav-c220-vec
fi
if [[ "$(basename -- "$SRC")" == *_cube.cpp ]]; then
  ARCH=dav-c220-cube
fi
"$A/tools/ccec_compiler/bin/bisheng" -x cce --cce-aicore-arch="$ARCH" \
  -O2 -std=c++17 -shared -fPIC \
  -I "$A/aarch64-linux/asc" \
  -I $A/aarch64-linux/tikcpp/tikcfw \
  -I $A/aarch64-linux/tikcpp/tikcfw/impl \
  -I $A/aarch64-linux/tikcpp/tikcfw/interface \
  -I $A/aarch64-linux/ascendc/include/basic_api \
  -isystem $A/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0 -isystem $A/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu -isystem $A/tools/hcc/aarch64-target-linux-gnu/include -I $A/include \
  -o "$OUT" "$SRC" \
  -L $A/lib64 -lruntime -lascendcl
echo "built $OUT"
