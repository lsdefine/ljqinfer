#!/usr/bin/env bash
# Dedicated A800 TP4 runtime. Visibility is placement, never an algorithm switch.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PATH="/data/ljq/vllm-env/bin:$PATH"
export PYTHON_BIN=/data/ljq/vllm-env/bin/python
_TP4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$_TP4_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset _TP4_ROOT
