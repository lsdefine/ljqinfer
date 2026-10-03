
## 死路径清除 / prefill算子替换 —— 引擎内实测裁决 (B1Q6, TP8, node09)

基线 REPLAY_MS = 20.596 (commit f482390)

### 1. linprobe 探针：decode 实际走哪条 linear 分支
引擎内 hook PrefillLinear.__call__ + ProjectionWorkspace.__call__，按 (name,dtype,branch,M,K,N) 计数：

| 分支 | 调用数 | 说明 |
|---|---|---|
| ProjectionWorkspace bf16预反量化+torch.mm | 8120 | 主 decode 图内，全部 M=6 |
| ProjectionWorkspace fused FP8 GEMV | **0** | 融合核从未命中 |
| ProjectionWorkspace mod.projection(N256参考) | 290 | 全 M=19，属 drafter，不在 replay 图内 |
| packed_linear (FP8/FP4打包路径) | **0** | decode 全程零调用 |
| torch F.linear (bf16小权重) | 少量 | weights_proj(N=4)/wk/wgate |

结论：decode 热路径没有在跑 prefill 打包 GEMM；packed_linear 与 WorkspaceRouted.gemm/gemm_device
在 decode 侧是死路径（后者另经 moegemm no-op 消融确认：20.554 vs 20.596，无影响）。

### 2. 反直觉裁决：fused FP8 GEMV 比 bf16 预反量化**慢**
V41_BF16_DENSE=0 强制走融合核：

| 配置 | REPLAY_MS | LOGITS_EQUAL |
|---|---|---|
| bf16预反量化+tensor core mm（现状） | **20.596** | True |
| fused FP8 dense GEMV | 23.0 | True |

M=6 太瘦，FP8 GEMV 时间花在 ALU 逐字节解包而非搬运带宽；预反量化一次让 tensor core 干活反而赢 2.4ms。
"带宽减半"的直觉在此形状下不成立。proj 占比 13.7%(2.823ms) 是真实计算量，不是 prefill 残留的水分。

### 3. drafter 的 prefill grouped_linear -> decode 版 (v4k.grouped_linear_bf16)
model/dspark.py 的 wo_a 分组 GEMM 原调用 ops.prefill.gemm.grouped_linear（逐 group 循环 + .float()），
换成 ops/decode/v4k.py 早已备好的 strided-batched bf16 版：

| | DRAFT_MS median |
|---|---|
| 改前 (prefill grouped_linear) | 15.22 |
| 改后 (v4k.grouped_linear_bf16) | 15.30 |

噪声内无差别(max 波动 14.2~21.3)。保留该改动的理由是结构性的：decode 侧不再 import prefill 算子；
性能上不可主张收益。tests/test_dspark_{layer,block,drafter,attend}: 16 passed / 3 failed，
3 项失败为改动前既有(CPU 上 Triton MoE 无法运行)，改前改后完全一致。

### 结论
"删死路径 + prefill 算子换 decode 专用"这一批已无性能可拿：热路径早就是 decode 专用实现，
剩余 prefill 引用要么零调用，要么在非热点的 drafter 上。
下一步收益只能从 Amdahl 账本里 51.2% 的碎片/长尾(每层58核、7250次launch)拿，
即算子融合减少 launch，而非替换现有 GEMM 实现。
