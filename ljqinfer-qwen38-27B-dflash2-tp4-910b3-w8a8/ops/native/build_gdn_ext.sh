#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/ops/native/src"
OUT="${1:-$ROOT/ops/native/ljq_gdn_ext_v3.so}"
PYTHON_BIN="${PYTHON_BIN:-/data/apps/torch_npu_env/bin/python}"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.1}"
TORCH_ROOT="$($PYTHON_BIN - <<'PY'
import pathlib, torch
print(pathlib.Path(torch.__file__).resolve().parent)
PY
)"
NPU_ROOT="$($PYTHON_BIN - <<'PY'
import pathlib, torch_npu
print(pathlib.Path(torch_npu.__file__).resolve().parent)
PY
)"
PY_INCLUDE="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths()['include'])
PY
)"
BUILD="${BUILD_DIR:-$ROOT/build/gdn_ext}"
mkdir -p "$BUILD" "$(dirname "$OUT")"
CXX="${CXX:-g++}"
COMMON=(-DTORCH_EXTENSION_NAME=ljq_gdn_ext_v3 -DTORCH_API_INCLUDE_EXTENSION_H
  -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\"
  -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -D_GLIBCXX_USE_CXX11_ABI=1
  -I"$SRC" -I"$NPU_ROOT/include" -I"$CANN_HOME/aarch64-linux/include"
  -isystem "$TORCH_ROOT/include" -isystem "$TORCH_ROOT/include/torch/csrc/api/include"
  -isystem "$PY_INCLUDE" -fPIC -std=c++17 -O3)
"$CXX" "${COMMON[@]}" -c "$SRC/ljq_gdn_ext.cpp" -o "$BUILD/ljq_gdn_ext.o"
"$CXX" "${COMMON[@]}" -c "$SRC/aclnn_torch_adapter/NPUBridge.cpp" -o "$BUILD/NPUBridge.o"
"$CXX" "$BUILD/ljq_gdn_ext.o" "$BUILD/NPUBridge.o" -shared \
  -L"$NPU_ROOT/lib" -ltorch_npu -Wl,-rpath,"$NPU_ROOT/lib" \
  -L"$TORCH_ROOT/lib" -lc10 -ltorch_cpu -ltorch -ltorch_python -o "$OUT"
echo "$OUT"
