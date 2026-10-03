# decode skew findings (2026-09-16)

## 基线 (HEAD 219007d)
short 9.61s / 32K 13.78s, 26.88ms/step.
核账(rank0,ms/step): AR 10.61 | MoE 3.59 | dense 3.9 | route 1.3 | hc 1.2 | sparse_attn 1.04

## 决定性证据: AR 的 10.6ms 不是通信, 是 rank0 单边等待
SPEC_PROF=50 采 8 个 rank 的 PROFJSON (10 步累计 ms):
- rank0 oneshot_ar = 132.8  -> 13.3 ms/step
- rank1..7 oneshot_ar = 24~32 -> 2.4~3.2 ms/step (逼近实测地板 2.7)
- 其余所有核在各 rank 完全一致: down 18.6 / gu 17.3 / cutlass 13.4 / route 9.0 / attn 7.4
- rank0 total 296 vs 其他 ~190

推论: one-shot AR 核时长 = 自身启动到全员数据到齐. rank0 最长 => rank0 每步**最先**到达 AR,
非 0 rank 晚到约 10ms, 即非 0 rank 的 GPU 每步有 8-10ms 空隙. 消除后 26.88 -> ~17ms.

## 已排除
- IPC/算法换 AR 核: 无效 (核本身已达地板 2.7ms)
- fused FP8 dense GEMV 接管 decode 投影: 退化 13% (A100 SM80 无 FP8 tensor core, CUDA core 标量 FMA 打不过 BF16 TC GEMM). 已回滚.
- V4 hc_mix_gemv / 融合 wq_a+wkv: 无收益

## 每步 host 循环 (strategy/decode_worker.py:174-205)
每 4 轮 dist.broadcast(gloo ctrl flag); step_g(graph replay); emitted.tolist() 阻塞 DtoH; rank0 另有 emit 回调.
所有 rank 跑同一 generate.

## 测量方法
bash _rs_ab.sh <port> [ENV...]  (ENV 必须分开传, 不能用引号并成一串, 否则 env 把它当单个赋值 -> rank0 崩)
轮询 worker.log 出现 'READY worker' (30-60s) -> python t32.py <port>
SPEC_PROF=50 出 PROFJSON; 再加 SPEC_TRACE=1 导出 /tmp/serve_384k/trace_r<rk>.json

## 2026-09-16 修正: AR 不是瓶颈, 之前 "AR 占 39.5%" 是单边等待假象

chrome trace (SPEC_PROF=50 SPEC_TRACE=1, 8 rank 全导出) 逐核对比:
- 除 oneshot_ar 外, **所有核在 8 个 rank 上时长完全一致**(差 <1%)
- oneshot_ar: r0=132ms r1=106ms r2..r7=25~32ms (10 步累计)
- 每个 rank 的 GPU idle 都是 23~26ms/10步 (2.4ms/step), gap 分布也一致
- rank2 span=231ms/10步=23.1ms/step, 与生产实测步时 22.5~24ms **完全吻合**
=> AR 真实成本 = 2.49ms/step (rank2 值, 即纯传输); rank0/1 的多出来的 8~10ms 是
   它们先到 AR 后的自旋等待, 不进入 wall time. 优化 AR 核 / 消 skew 无收益.

## 干净账本 (rank2, 20.74ms busy + 2.4ms idle = 23.1ms/step)
| 项 | ms/step | 次/step |
|---|---|---|
| oneshot_ar | 2.49 | 89 |
| moe_decode_down_fp4 | 1.87 | 40 |
| moe_decode_gu_fp4 | 1.73 | 40 |
| cutlass_80_tensorop_s16816gemm_bf16_64x64 | 1.38 | 188 |
| _route_gemv | 0.91 | 43 |
| sparse_attn_decode | 0.75 | 38 |
| nccl AllGather_RING_LL | 0.69 | 25 |
| _hc_gemv | 0.66 | 86 |
| ampere_bf16_s16816gemm 64x64 sliced | 0.64 | 44.5 |
| unrolled_elementwise | 0.63 | 57 |
| splitKreduce | 0.56 | 235 |
| _hc_gates | 0.53 | 86 |
| ampere_s16816gemm 64x64 tn x2 | 1.00 | 90 |
| fp4_gemv_dec3 | 0.45 | 6 |
| wmma_tensorop_bf16 | 0.41 | 56 |
| _route_topk | 0.40 | 43 |
| rms_norm_f32w | 0.39 | 101 |
| indexSelect | 0.38 | 73.5 |
| vectorized_elementwise x2 | 0.68 | 342 |
| radix sort onesweep | 0.32 | 34 |
| _collapse_norm | 0.32 | 87 |
| sparse_attn_combine | 0.30 | 38 |
| memcpy32_post | 0.29 | 162 |
| rope_inplace | 0.28 | 149.5 |

簇汇总: dense GEMM 家族(cutlass+ampere+wmma+splitK) ~4.33ms/380 launch |
MoE 3.60 | AR 2.49 | route 1.31 | hc 1.19 | attn 1.05 | norm 0.71 |
纯搬运/杂项(elementwise+indexSelect+memcpy32+rope+expand+index_ew) ~2.5

## 下一刀候选 (按可砍量)
1. 杂项搬运核 ~2.5ms/step, 上千次 launch: 融进相邻核, 纯净收益
2. GPU idle 2.4ms/step: 图外 eager 段
3. dense GEMM 4.33ms 拆成 380 次 launch (M<=8): 合并同形状投影, 减少 splitK/尾效应
4. AR 2.49 / MoE 3.60 / attn 1.05: 已近硬件下限, 不动
