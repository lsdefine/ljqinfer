#!/bin/bash
# AscendC kernel build (910B3 = dav-c220). Usage: bash build.sh <src.cpp> <out.so>
set -e
A=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}
SRC=${1:-poc_add.cpp}
OUT=${2:-libpocadd.so}
$A/tools/ccec_compiler/bin/bisheng -x cce --cce-aicore-arch=dav-c220 \
  -O2 -std=c++17 -shared -fPIC \
  -I $A/aarch64-linux/tikcpp/tikcfw \
  -I $A/aarch64-linux/tikcpp/tikcfw/impl \
  -I $A/aarch64-linux/tikcpp/tikcfw/interface \
  -I $A/aarch64-linux/ascendc/include/basic_api \
  -isystem $A/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0 -isystem $A/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu -isystem $A/tools/hcc/aarch64-target-linux-gnu/include -I $A/include \
  -o "$OUT" "$SRC" \
  -L $A/lib64 -lruntime -lascendcl
echo "built $OUT"
