# Standalone TopK V2-style selection (A100)
Original four-kernel implementation inspired by SGLang PR26788 coarse-bin/exact-refinement design, not a port of SM90 cluster kernels. Decode/verify now uses V2 by default via ops.sparse_index_decode.select_decode; prefill remains independent.

## Contract
FP32 contiguous [rows,width] scores; int64 inclusive last-valid positions; int32 [rows,K] unsorted output. Same-device explicit workspace from ops.sparse_topk_v2.workspace; no invocation-time allocation. Short/empty rows pad -1. IEEE ordered-key descending selection; lowest ID resolves exact ties (+0 above -0). NaNs unsupported. Caller supplies non-aliasing buffers and positions without int64 overflow. Output permutation differs from old selector; integration must validate attention reductions.

Stages: 10-bit monotone FP16 coarse histogram, threshold/chunk-prefix plan, warp-ballot collect, exact FP32 boundary refinement. Full-N candidate storage avoids truncation/fallback, approximately 9 bytes per score in addition to scores/output. Narrow/flat distributions may place all N values in the boundary bin: correctness tested, latency not represented by normal-distribution benchmark.

## Reproduce
```sh
python -m torch.distributed.run --standalone --nproc_per_node=8 tests/test_topk_v2.py --bench --output /tmp/topkv2
compute-sanitizer --tool memcheck --error-exitcode 99 python tests/test_topk_v2.py --quick --output /tmp/topkv2_mem
compute-sanitizer --tool racecheck --error-exitcode 99 python tests/test_topk_v2.py --quick --output /tmp/topkv2_race
```

## Measurement
A100-SXM4-80GB, eight independent GPUs. Each rank passed 50 distribution/shape cases including changed graph inputs/positions, checked against independent full sort. Bench additionally compares sets with old decode selector (03a737b). Graph device events, 200 invocations/sample, ABBA x3, rank maximum before sample median. Synthetic normal scores; excludes scoring, TP collectives, page conversion, attention and full model. Q is rows PER GPU, not total TP8 queries.

| Rows/GPU | N | Old us | V2 us | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 8192 | 22.723 | 21.806 | 1.042x |
| 6 | 8192 | 23.311 | 22.866 | 1.019x |
| 8 | 8192 | 23.885 | 22.930 | 1.042x |
| 1 | 65536 | 29.693 | 25.692 | 1.156x |
| 6 | 65536 | 34.271 | 26.422 | 1.297x |
| 8 | 65536 | 36.536 | 27.525 | 1.327x |
| 1 | 81920 | 33.654 | 26.665 | 1.262x |
| 6 | 81920 | 37.496 | 28.416 | 1.320x |
| 8 | 81920 | 39.091 | 28.641 | 1.365x |
| 1 | 1048576 | 161.129 | 103.555 | 1.556x |
| 6 | 1048576 | 342.523 | 143.304 | 2.390x |
| 8 | 1048576 | 458.898 | 190.797 | 2.405x |

Retained build: tile4096. Tested tile1024 was faster at 8K but much slower at 1M; no runtime heuristic added. Retained 8K differences are small, not a robust large gain. Evidence: node09 /mnt/data2/kw/glm53_layer12_tp8/topkv2_standalone/{report.json,tile4k_full.rank*.json,tile4k_mem.log,race.log}.

## Decode integration (main worktree)
`select_decode` runs packed TP8 scoring -> V2 -> ID gather. Optional caller-owned `topk_workspace` uses min(ceil(T/8),query_tile) rows, reused across tiles.
All 8 ranks passed FP16/BF16 selection references, ties/empty rows, uneven TP ownership, workspace reuse and dynamic graph replay. Q=1/6/8 at 64K also passed selection -> gather -> sparse MLA and full-chain graph replay changing positions, context, page table, keys and queries. Selected sets agree with old decode; output order can differ. Initial attention maxabs=1.52587890625e-5; tolerance rtol=3e-3, atol=3e-4.
Reproduce: python -m torch.distributed.run --standalone --nproc_per_node=8 tests/test_decode_topk.py --bench --baseline-cu ops/sparse_index_decode.cu --output /tmp/topkv2_integrated
Evidence: /mnt/data2/kw/glm53_layer12_tp8/topkv2_standalone/integrated_final.{log,exit.json} and integrated_final.rank*.json on node09.
Q6 index: 118-119us vs old 119-122us; index+MLA: 170-172us vs old 168-170us. No demonstrated end-to-end gain. TP8 total Q6 owns one row/rank, not six. No whole-model speedup claim.
The operator-only results above do not establish full-model serving or end-to-end speedup. Model-layer binding is described below; Engine.load retains its integration guard.

## Model-layer binding
TransformerBlock(..., sparse=binding) now runs real attention projections -> paged MLA/index cache writes -> TP8 scoring and decode V2 (or separate prefill selector) -> sparse MLA -> output projection -> TP all-reduce/residual. load_indexer(root,layer,rank,device) reads TP-local indexer weights; configuration-marked shared layers return None.

Full layers use SparseAttentionBinding(tokens=T,capacity=N,parallel=parallel,index_weights=weights,index_pool=pool128,index_table=table). Shared layers use shared_from=producer instead of index weights/cache, sharing selected IDs but NOT MLA KV. The caller must run the configured producer before consumers each step with identical position/context buffers, and allocate one binding per concurrent graph. Optional pair_group=create_pair_group() enables paired prefill MLA; decode remains local. All ranks must create groups in identical order. This fixed-shape single-sequence interface is not a multi-request scheduler.

Validation uses real layer-10 indexer/attention and layer-12 shared attention weights, synthetic hidden states and historical KV. Eight ranks, Q=1/6/8, top-2048, capacities 4096 and 65536. Tests call TransformerBlock.attention against independent PyTorch complex-RoPE, explicit attention and global index-score references. Decode, prefill, shared IDs and changed-input decode CUDA graph replay pass. At 64K, paired prefill and independent new-index-cache checks also pass. Maximum full/shared residual-output relative L2: 7.1543e-5. This does not test intervening layer 11, block MoE, 78-layer logits or the service engine.

The 64K cache check exposed FP32 RoPE phase error; binding now computes frequencies/phases in FP64, trig outputs in FP32 and stores FP16. Original tolerance retained. Mathematical-reference agreement is not full-model checkpoint/HF parity or a performance result.

Reproduce: CAPACITY=65536 PAIR=1 REPORT=/tmp/layer_binding python -m torch.distributed.run --standalone --nproc_per_node=8 tests/test_glm53_sparse_binding.py

Evidence on node09: /mnt/data2/kw/glm53_layer12_tp8/topkv2_standalone/layer_binding_64k_fixed.{log,exit.json} and .rank0.json through .rank7.json (14 records each, ALL_PASS, exit 0). Earlier 4K evidence: layer_binding.* in the same directory. No layer-level latency improvement is claimed.
