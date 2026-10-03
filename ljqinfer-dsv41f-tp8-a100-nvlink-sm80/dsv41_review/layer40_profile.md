# 40 层逐层耗时分析 (decode, 8-rank trace, 2026-09-17)

## 方法
- 数据: `SPEC_TRACE=1` 采 8-rank torch profiler trace, 取 rank0 (`/tmp/serve_384k/trace_r0.json`, 23655 个 GPU 事件, span 321 ms)。
- 切层锚点: `sparse_attn_decode_kernel` 全程恰 400 次, 且 ts 大间隔(4.5~6.4 ms)严格出现在 `i % 40 == 39`,
  故 **40 层 x 10 个 verify step** 成立, 第 i 次 attention 对应 `step = i//40, layer = i%40`。
- 单层区间定义为 `[attn_i, attn_{i+1})` (相位对齐到 attention 起点, 覆盖完整一层)。
- 口径校验: 40 层中位数之和 = **18.69 ms**, 与引擎实测步时 17.3 ms 吻合(profiler 自身有开销)。

## 结果: 三类层
| 类别 | 层 | 中位 wall | 核数 | 说明 |
|---|---|---|---|---|
| 基线层 | 31 层(L2..L6,L8..L12,L14..L18,L20..L22,L24..L26,...) | **288~303 us** | 36 | 纯主干: attn + MoE gu/down + AR, 极稳(max-min<20us) |
| 尖峰-小 | L0 | 370 us | 38 | 多 1 次 AR(+56us) + engram_gate + 1 个 128x64 GEMM |
| 尖峰-中 | L23/L27/L31/L35 (每 4 层一次) | **401 us** | 56 | 多 1 次 nccl AllGather(+25us) + `_score` + `_cand_b1/b2` (MTP 候选) |
| 尖峰-大 | L7 / L13 / L1 | 517 / 603 / 750 us | 102~105 | 多 ~18 个 vectorized + 10 个 elementwise + 6 个 index + 6 unrolled + 3 cutlass + 1 AllGather |
| 尖峰-最大 | L19 | **875 us** | 167 | 多 13 个 `cub::DeviceRadixSort`(94us) + 33 vectorized + 16 elementwise + 2 AllGather + mbtopk |
| step 尾段 | L39 + tail | **5118 us** | 553 | 见下 |

## step 尾段 (占 27.4%, 最大单块)
wall 5117.6 us / busy 3551.7 us / 553 核。构成:
- `oneshot_ar` x9 = 755 us (单次 ~84us, 是层内 AR 的 2.4 倍)
- cutlass GEMM x16 = 352 us; `ampere_bf16 64x64` x6 = 248 us; `128x64` x1 = 108 us (lm_head/logits)
- `ncclDevKernel_AllGather` x7 = 235 us
- **碎核: vectorized x92=175us, unrolled x42=149us, memcpy32_post x69=139us, elementwise x25=106us, index x23+x20=161us, CatArray x19=49us** -> 合计 ~780 us / 270 个核
- drafter 的 MoE: gu x4 + down<3> x3 + down<6> x1 = 218 us
- **纯空窗 GAP 872 us** (发生在 busy 3406us 处, 之后紧跟 `Memcpy HtoD (Pinned->Device)`) + 189/102/76/48 us 四个小 GAP

## 归因 (每 step)
1. **主干 40 层的真实成本 ~11.7 ms** (31 个基线层 x 293us 外推)。这部分是 attn + MoE + AR, 已接近带宽/算力上限。
2. **torch eager 碎核税 ~2.6 ms**: 尖峰层多出的 elementwise/index/cub 共 ~1.83 ms + 尾段碎核 ~0.78 ms。
   这些核全部来自 **host 侧 torch 编排**(engram 取数、MTP 候选、topk/排序), 不是模型计算。
3. **engram AllGather ~0.3 ms**: 每 ~6 层一次 nccl AllGather(24~51us), 串在主干关键路径上。
4. **host 同步空窗 ~0.87 ms**: 尾段一次 872us 的 GPU 全空, 紧跟 pinned HtoD, 即等 host 准备下一 step 输入。
5. 尾段 AR 单次 84us 明显劣于层内 35us, 说明尾段 AR 的 skew/尺寸有问题。

## 可下刀的三处(按盘子排序)
1. **尾段 872us 空窗 + 270 个碎核**(~1.65 ms): 把 sampling/候选/KV 写回/下一 step 输入准备 fuse 成少数几个核, 并把 HtoD 预取提前到上一 step。
2. **L19 的 cub::DeviceRadixSort + mbtopk (~540 us)**: 用自写 fused topk 替换 torch topk。
3. **L1/L7/L13 的 ~40 个 eager 小核 (~0.77 ms)**: engram 取数路径 fused 化, 顺带消掉串行 AllGather。
