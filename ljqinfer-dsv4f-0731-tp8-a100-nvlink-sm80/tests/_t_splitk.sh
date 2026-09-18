#!/bin/bash
cd /mnt/data/kw/ljqinfer_dsv4f_tp8
export CUDA_VISIBLE_DEVICES=0
# split-K is fixed at 8 in ops/sparse_attn_paged.cu (it is a numerics knob).
python tests/_t_splitk.py
echo DONE
