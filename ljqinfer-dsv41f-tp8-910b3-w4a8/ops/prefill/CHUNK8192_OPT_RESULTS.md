# 8192-token leaf optimization: 2026-10-06

Target: 8192/8000 = 1.024 seconds per encoder chunk. NOT demonstrated: these are synthetic leaf benchmarks without model/engine execution. Full operator coverage remains incomplete.

## Changes
- index_select: four-line large-input branch (rows>=2048, aligned keys<=4096), compute tile32 to128; preserves original 32-row communication sequence. Adds 8 MiB temporary dot for tested four-head shape. Wider keys fall back.
- hc_expand: one-line launch choice, 20 cores for T>=2048, otherwise original 24.

## TP8 measured results
Reference c66d283. Four warmups, 30 alternating pairs; median of per-round maximum wall time across eight ranks. Production PrefillParallel communicator. Synthetic q[8192,4,128], keys[4096,128], topk512; HC residual[8192,4,5120].

| Operator | Seed | Before ms | After ms | Speedup |
|---|---:|---:|---:|---:|
| index_select | 4201 | 93.767 | 82.380 | 1.138x |
| hc_expand | 4201 | 1.373 | 0.928 | 1.479x |
| index_select | 4202 | 89.216 | 84.618 | 1.054x |
| hc_expand | 4202 | 1.376 | 0.944 | 1.459x |

Single-device index speedup ~2.26-2.28x is NOT TP8 speedup. TP8 T1 index ratios are 0.976x/1.002x; strict zero regression is not established. All measured main tests had zero allocation retries; index adds 8 MiB peak allocated memory, not a full-model memory-pressure test.

## Validation
- Single rank: 24 PASS (T1/128/2047/2048/2049/8192, two seeds, two operators).
- TP8: 12 PASS per rank (T1/2048/8192, two seeds, two operators). Outputs bit-identical; post-communication index scores byte-identical, communication shapes/counts unchanged, explicit inputs unchanged.
- Extra single-rank tests: 8192x4097 wide-key fallback, 2049x4095 padded keys + candidate generation, 8192x4096 candidate mask: all PASS.
- Combining communication into 128-row blocks failed TP8 TopK equality; REJECTED and absent from production source.
- Frozen Python/C++ boundaries share underlying native libraries; not an independent mathematical oracle. Actual weights, CED integration, whole encoder and serving untested.

## Evidence and reproduction
scripts/test_prefill_chunk_ab.py is the reproducible main benchmark. In the configured A2 CANN/torch_npu environment set PYTHONPATH=.:scripts, PYTORCH_NPU_ALLOC_CONF=expandable_segments:True, TASK_QUEUE_ENABLE=1, OMP_NUM_THREADS=1, MAX_JOBS=4. Run python scripts/test_prefill_chunk_ab.py --output <fresh-single-dir>, then python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/test_prefill_chunk_ab.py --output <fresh-tp8-dir> --rows 1 2048 8192.
chunk8192_ab_results.json contains source hashes, all raw timings, memory data and extra tests/script. Original logs: A2-3 /data/prefill_leaf_opt/{chunk_single_v3,chunk_tp8,chunk_extra,chunk_comm_tp8}*.
Historical ~1.322s encoder chunk needs ~0.298s (22.6%) reduction to reach 1.024s. Do not infer chunk latency by adding/subtracting leaf timings. No services restarted; A2-1/A2-2 untouched.
