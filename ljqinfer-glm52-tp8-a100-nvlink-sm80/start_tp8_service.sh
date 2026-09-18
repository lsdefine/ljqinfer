#!/usr/bin/env bash
# GLM-5.2 原版 TP8 完美启动命令（00eb047 实际验证通过）
#
# !!! 不得乱加任何环境变量或启动参数 !!!
# 特别禁止添加/修改：
#   LD_PRELOAD
#   PYTORCH_CUDA_ALLOC_CONF
#   TORCH_CUDA_ARCH_LIST
#   CUDA_VISIBLE_DEVICES
#   LD_LIBRARY_PATH 的内容或顺序
#
# 错误环境可能让 /health 正常，但首个真实请求在 MONO CUDA Graph 捕获时
# 触发 illegal memory access，并毒化整个 CUDA context。
# 下面就是验证通过的完整启动方式，不要“优化”、包装或改写。

set -e

cd /mnt/data/kw/ljqinfer_tp8

unset LD_PRELOAD
unset PYTORCH_CUDA_ALLOC_CONF
unset TORCH_CUDA_ARCH_LIST
unset CUDA_VISIBLE_DEVICES

export PATH=/mnt/data/kw/.local/lib/python3.13/site-packages/cmake/data/bin:/usr/local/cuda/bin:/mnt/data/kw/anaconda3/bin:/mnt/data/kw/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/mnt/data/kw/.local/nccl/lib

nohup /mnt/data/kw/anaconda3/bin/python -u -m server.engine_server \
  > /tmp/ljqinfer_tp8_engine.log 2>&1 < /dev/null &

echo $! > /tmp/ljqinfer_tp8_engine.pid

nohup /mnt/data/kw/anaconda3/bin/python -u -m server.server \
  > /tmp/ljqinfer_tp8_api.log 2>&1 < /dev/null &

echo $! > /tmp/ljqinfer_tp8_api.pid
