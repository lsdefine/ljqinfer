#!/usr/bin/env bash
# Source before starting Python. Keeps the CANN/custom-op provider deterministic.
_TP4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_TP4_CANN="${TP4_CANN_HOME:-/usr/local/Ascend/cann-9.0.1}"
if [[ ! -f "$_TP4_CANN/aarch64-linux/lib64/libhccl.so" ]]; then
  _TP4_CANN=/usr/local/Ascend/cann-9.0.1
fi
if [[ -f "$_TP4_CANN/set_env.sh" ]]; then
  # Clear any inherited toolkit identity before sourcing the sole CANN 9 provider.
  unset ASCEND_HOME_PATH ASCEND_OPP_PATH
  # CANN's script also supplies driver/runtime paths; exact paths are re-prepended below.
  source "$_TP4_CANN/set_env.sh" >/dev/null 2>&1 || true
fi
export ASCEND_HOME_PATH="$_TP4_CANN"
export ASCEND_CUSTOM_OPP_PATH="$_TP4_ROOT/vendor/gdn_custom_opp"
export LD_PRELOAD="$_TP4_ROOT/vendor/runtime/libstdc++.so.6.0.30${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHON_BIN="${PYTHON_BIN:-/data/apps/torch_npu_env/bin/python}"
_TP4_PYROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
_TP4_PYVER="$($PYTHON_BIN -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
_TP4_SITE="$_TP4_PYROOT/lib/$_TP4_PYVER/site-packages"
export LD_LIBRARY_PATH="$_TP4_ROOT/vendor/gdn_custom_opp/op_api/lib:$_TP4_ROOT/vendor/gdn_custom_opp/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$_TP4_CANN/aarch64-linux/lib64:$_TP4_SITE/torch_npu/lib:$_TP4_PYROOT/lib64/$_TP4_PYVER/site-packages/torch/lib:$_TP4_SITE/torch/lib:${LD_LIBRARY_PATH:-}"
unset _TP4_ROOT _TP4_CANN _TP4_PYROOT _TP4_PYVER _TP4_SITE
