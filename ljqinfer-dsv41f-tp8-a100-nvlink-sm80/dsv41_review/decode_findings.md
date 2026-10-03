# V4.1 decode 提速 — 实测账本 (node09:/mnt/data/kw/ljqinfer_dsv41f_tp8)

口径: **B1Q6**, TP8, REPLAY_MS median, 每刀均验 LOGITS_EQUAL maxdiff 0。

## 战绩 37.87 -> 20.596 ms (-45.6%)
| 刀 | commit | 内容 | 步时 |
|---|---|---|---|
| 1-8 | c70e2ae | 批量结构整改 | 22.631 |
| 10 | 6193bd3 | NCCL_ALGO=Tree (init 前设置) | 21.983 |
| 11 | 55e99e9 | residual 常量 cast 缓存 | 21.174 |
| 12 | 296aa6a | swiglu 走 triton 单核 (8核->1核) | 20.652 |
| 13 | 8830d53 | hc_mix 常量 scale/base 缓存 (160次/步) | 20.596 |
| 14a | 46ff94f | collapse_norm pre 缓存 (81次/步) | 20.675 (**收益0**) |
| 15 | 待验 | v4k.py bmm 转置权重缓存 (k16) | ? |

## 真实 kernel 时间榜 (刀14 后, 21.74ms/replay, 4398 核)
| kernel | ms | 次数 |
|---|---|---|
| ncclDevKernel_AllReduce TREE | **2.914 (13.4%)** | 83 |
| moe_decode_down_fp4 | 1.863 | 40 |
| moe_decode_gu_fp4 | 1.698 | 40 |
| **unrolled_elementwise direct_copy** | **1.288** | **250** |
| cutlass s16816gemm | 1.25 | 167 |
| **cublasLt splitKreduce** | **0.762** | **299** |

## 关键认知
1. **graph replay 中 meta op (view/t/slice/empty) 零成本** —— 只有真 kernel 算数。探针计数高 != 有收益 (刀13/14 砍掉 241 次 cast 但都是 [H] 小向量，收益为 0)。
2. **290 次 mm 是必要 GEMM**：projection_workspace 的 bf16 路径先 dequant 再走 tensor core，与 V4 (load 期 dequant) 同款。A/B 实测 `V41_BF16_DENSE=0` 强制 fused 单核 = **22.964ms，反而慢 2.4ms**。此路已封。
3. 250 次 direct_copy (1.288ms) 元凶 = `ops/decode/v4k.py` 的 `torch.bmm(x.transpose(0,1), weight.transpose(1,2))`，bmm 读不了转置视图，每步物化常量权重副本。

## 已证伪 (勿重试)
刀4 select() 融核 | peer_ar | 自写 GEMV | splitK | MoE 6x | NVLS |
刀14b engram_gate 接 CUDA 核 (等价性 MAXDIFF 0 但 decode 根本不调 residual 版，prefill_block.py:151 早已走核)

## 操作铁则
1. 禁 pkill -f (会自杀)，清卡用 nvidia-smi 取 PID
2. setsid + disown 后台，**禁 sleep 等待**：发车即返回，下轮 grep 日志
3. ssh 内联 setsid 会挂起 → 写 .sh 文件再 `setsid bash`
4. 改码禁 str.replace 盲改；用行号 + assert 锚点；避开 @装饰器
5. 每次换 master_port (已用到 29766)；跑完清卡

## 刀15-18 实验判决 (均已回滚, 基线保持 20.596)
| 实验 | 假设 | 实测 | 判决 |
|---|---|---|---|
| k16 刀15 | v4k.py:106 bmm 转置权重缓存 | 20.646 | 收益0 + 多占数GB显存 → 回滚 |
| k17 | NCCL_PROTO=LL + MIN/MAX_NCHANNELS=4 | 21.223 | 慢0.63 → env 调优已到顶 |
| k18 | moe down 核 kRows 4→8 | 20.14 但 **maxdiff nan** | 假收益, 回滚 |

### MoE 核带宽账 (B1Q6, nsel=36)
- gu 核: 读 53MB/层, 理论 35us vs 实测 42us = **83% 带宽, 已近顶**
- down 核: 读 26.5MB/层, 理论 17us vs 实测 46us = 38%
- **但 down 核的 38% 不可改**: kRows=4 硬绑 32-lane 布局 (rsel=lane>>3 → 8 lane/行),
  acc[6][4] 的 4 是 uint4 word 数不是行数; 且 shuffle 树 d=4,2,1 刻意复刻标量归约顺序
  以保 bit 对齐 (核注释第 130-134 行明示)。动布局 = 放弃 bit-exact。

## 20.6ms 的构成结论 (剩余全是硬成本)
AllReduce 2.9 (通信下限, env 到顶) + MoE 3.56 (带宽/bit-exact 双约束) +
GEMM 2.0 (必要计算) + 其余长尾。**再降需结构性改动**:
序列并行(ReduceScatter+AllGather 减半通信) / 通信-计算 overlap / 放弃 bit-exact 的低精度通信。

## FP8 dense GEMV 重做 (v10) 判决 — 2026-09-16
口径: CUDA Graph 内重放计时 (decode 就是图重放), A100 80G, M=6, 真实 decode shape。
| shape (M6) | cuBLAS bf16 mm | 仓内 v9 fp8 | 新 v10 fp8 (最优 cfg) |
|---|---|---|---|
| K5120 N5120 | 36.2us | 121.8 | **32.5** |
| K1280 N4608 | 8.5 | 32.7 | 12.5 |
| K5120 N1280 | 10.4 | 33.8 | 11.9 |
| K5120 N640 | 10.2 | 18.0 | **7.9** |
| K4096 N1024 | 8.8 | 27.2 | 9.7 |
| K5120 N128 | 6.7 | 9.1 | **5.0** |
| K1024 N640 | 6.0 | 5.4 | **4.0** |
| 合计 | 86.8 | 353 | 83.5 |

结论: v10 比仓内 v9 快 **3-4x**, 但对 cuBLAS 只有 **+4%** -> 全轮收益 <0.1ms, **不接入**。

拿到的硬事实 (后人勿重蹈):
1. **cuBLAS bf16 mm 在 M=6 下已达 1.45TB/s (A100 峰值 ~1.55)**, 不是 tensor-core 浪费,
   "M=6 用 64x64 tile 所以低效" 的推断错误。手写 bf16 GEMV 最高只到 1.0TB/s, 必输。
2. 图外计时会给每个 mm 加约 17us 启动地板, 用它比较必然得出错误结论 -> 必须图内计时。
3. v9 慢的根因: 每 lane 每步只取 4B 权重 (ILP=1), 延迟暴露。
4. 写 fp8 GEMV 的三个坑, 依次让 kernel 慢 25x / 4x / 3x:
   - kernel 内对局部数组取地址或做指针算术 -> 数组掉进 local memory;
   - E4M3 解包放在 M 循环内 -> 转换量 x MT;
   - **__nv_cvt_fp8x2_to_halfraw2 在 sm_80 是软件模拟**, 必须手写位运算:
     `bits = ((u & 0x007F007F) << 7) | ((u & 0x00800080) << 8)` 再 `__hmul2(*(half2*)&bits, 256.f)`,
     次正规也精确; E8M0 scale 同理用 `__int_as_float(e << 23)` 代替 exp2f。
5. 剩余可能收益只在"取消 BF16 权重 bank"(省数十 GB 显存)时才值得, 步时本身无收益。


## FP8 权重直读 (第三条路) — 实测否决, 本条永久关闭该方向

背景: dense 路径为每个 FP8 权重常驻一份 BF16 反量化副本, cuBLAS 在 M=6 下
跑到 1.45 TB/s, 但每个权重元素读 2 字节. 前两次尝试 (v9/v10 fp8_dense_gemv.cu)
都是手写 SIMT GEMV, 放弃了 tensor core, 必输. 本轮试的是第三条路:
保留 tl.dot 走 tensor core, 在寄存器里用位运算把 E4M3 解包成 BF16,
scale 是 2 的幂所以 BF16 乘法精确 (bits=((u&0x7F)<<4)|((u&0x80)<<8), 再乘 2^(e-7)).
正确性没问题 (maxrel <= 3.5e-3, 与 BF16 同级), 但性能:

| M | 7 个 decode shape 合计 | 对 cuBLAS |
|---|---|---|
| 6  | 159.9 us (cuBLAS 91.2) | x0.57 |
| 48 | 327.6 us (cuBLAS 95.7) | x0.29 |

分离探针 (probe_bw.py, shape 5120x5120):
* P1 只读 FP8 字节不解包不乘: 35.12 us = **746 GB/s**
* P2 同样 tiling 但权重已是 BF16 直接 tl.dot: 34.71 us = **1510 GB/s**
* cuBLAS BF16: 39.59 us

结论: 读一半的字节并没有换来一半的时间. A100 是 sm_80, 没有 FP8 硬件,
E4M3->BF16 必须走 ALU, 逐元素解包的指令开销把字节红利吃干净; FP8 路径
的实际带宽上限 746 GB/s 对比 BF16 的 1510 GB/s, 折合收益只有 ~11%,
而解包和 dot 的开销远大于这 11%. 三条路 (SIMT GEMV x2, tensor-core 解包 x1)
全部输给 cuBLAS, dense GEMM 的 4.06 ms/step 可以认为已经到顶.
在 Hopper 及以后 (原生 FP8 mma) 这个结论会反过来, 届时再开.

副产品 (正面): P2 说明 **Triton 手写 BF16 GEMM 在大 shape 上比 cuBLAS 快 12%**
(34.71 vs 39.59 us), 小 shape 落后只是因为探针没开 split-K, grid 只有几个 block.
真要再榨 dense, 方向是 Triton + split-K 重写而不是换权重精度, 但盘子只有 ~0.3 ms.

## B8Q6 展望 (M=48) — 固定税会被摊薄, 干活占比自然逼近 80%

cuBLAS dense 7 shape 合计: M=6 时 91.2 us, M=48 时 95.7 us, **只涨 5%**.
权重带宽受限的部分 (dense / MoE / mHC) 的代价几乎与 M 无关, AR 和小核税
也是每步固定. 随 batch 线性增长的只有 attention (KV 不能跨序列复用).
新 _score 核实测 T=48 仅 135 us/call (旧核 2746 us, 旧核在 b8q6 下单算子
就要 ~23 ms/step, 会直接堵死这条路线).


## 100% 覆盖的 decode 核账 (659abe4, 32K, q=6, TP8 rank-slowest)

TOTAL 17.044 ms/step, **110 种 kernel, 1943 次启动/step** (平均每 8.8us 一次).
口径: torch profiler self-time, 10 步平均. 与 decode_ms_per_step 17.30 吻合.

| 类别 | ms/step | 占比 | 说明 |
|---|---|---|---|
| A 真干活 (算/搬权重与KV) | 9.67 | 56.7% | MoE 3.86 + dense GEMM 3.62 + attn 1.46 + hc_gemv 0.66 + engram 0.07 |
| B 通信 | 2.91 | 17.1% | oneshot_ar 2.417(89次) + AllGather 0.491(16次) |
| C 税 (既非算也非必需搬运) | 4.46 | 26.2% | 见下 |

税的构成 (无单项大头, 全是碎片):

| 项 | ms | 次/step |
|---|---|---|
| _hc_gates_collapse | 0.686 | 80 |
| splitKreduce (cuBLAS split-K 归约) | 0.667 | 167 |
| route 族 (topk/gemv/part/join) | 0.627 | 95 |
| torch eager 散核 (elementwise/indexSelect/reduce/sort/cat) | ~0.72 | ~250 |
| rope_inplace + memcpy32_post | 0.558 | 298 |
| rms 族 (rms_norm_f32w/rms_split2/collapse_norm) | 0.208 | 68 |
| _expand | 0.211 | 80 |
| topk_select_post | 0.207 | 4 |
| 量化 roundtrip (_fp8_rt/_fp4_rt) | 0.127 | 63 |
| candidate 族 (_cand_b1/b2/hist/hist2/emit) | 0.125 | 20 |
| _attn_ids + _swiglu_packed | 0.164 | 80 |
| memcpy/memset 族 | 0.076 | ~30 |

结论: 没有"丢失的大头". 26% 的税由 ~86 种小核摊出, 最大单项也只有 0.69ms.
这是碎片化本身的成本, 只能靠融合逐项回收, 没有一刀砍掉 3ms 的地方.

### 由此账本纠正的一个判断: splitKreduce 是 cuBLAS 的隐藏税

p3.py 只比了 GEMM 本体 (cuBLAS 94.0us vs Triton+splitK 96.9us, 每层),
当时判 Triton 没赢. 但 cuBLAS 的 split-K 要额外起 splitKreduce 归约核,
账上是 0.667ms/step = 16.7us/层, 而 Triton 的 split-K 用 kernel 内 atomic
归约, 不需要第二个核. 真实对比是 110.7us vs 96.9us, **Triton 反而快 13.8us/层
= 0.55ms/step (3.2%)**. 这是目前 dense 路径唯一还剩的确定性收益.


## 否决: _hc_gates_collapse 的 "降寄存器 + 分块两遍" 改写 (2026-09-17)

**动机**: 核账里 `_hc_gates_collapse` 是单项最大税 (0.686 ms/step, 80 次, 8.6us/次),
而它的 grid 只有 `(rows,)` = 6 个 block, 在 108 SM 上看似严重并行不足;
kernel 内又用 `BD = next_power_of_2(D) = 8192` 的整块寄存器 tile, 且对 H=4
逐个做标量 `tl.load(PPRE + row*H + i)`, 形成 4 次串行依赖。

**改法**: 一次性向量 load `PPRE[BH]`, 用 `[BH, BD]` 2D load 取 X,
`BD` 降到 1024 分块循环, 因 RMS 需要全 D 归约故拆成两遍 (pass1 累 sumsq, pass2 store)。

**实测 (真实服务 A/B, 同一 runab.sh 流程, 32K 文档)**:

| | MARK short | MARK doc32k | decode_ms_per_step (32k) |
|---|---|---|---|
| baseline | 8.36 s | 10.57 s | 17.99 |
| 改写后 | 9.38 s | 12.37 s | 19.63 |

**慢 12~17%, 已回滚。**

**为什么输**: 改动没有触及真正的约束 —— block 数仍是 6。RMS 必须在全 D 上归约,
所以 collapse 无法沿 D 切块并行, 这是算法约束不是写法问题。而分块两遍带来了
X 的二次读取和循环开销, 纯粹是净增成本。原来的 `BD=8192` 单遍大 tile 反而是对的:
一次读完、无重复访存, 寄存器压力由编译器自行 spill 到 L1, 代价小于多读一遍。

**结论**: 这 0.686 ms 里的大部分不是"写法差", 是 6 个 block 的启动+访存延迟下限。
要真正回收它, 只能改结构 —— 把 collapse 融进它的消费者(下一个 projection GEMM),
或让 RMS 用两阶段跨 block 归约换取并行度(会再加一个核, 得不偿失)。
维持现状。

**附带确认**: `tests/test_decode_layer.py` 与 `tests/test_decode_model.py` 在
baseline 下同样是红的 (前者 `DenseLinear` stub 缺 `fused`, 后者玩具尺寸 K=64
不满足 `ksplit*bk` 整除), 属预先存在的失败, 不能用来验证 decode 改动。
