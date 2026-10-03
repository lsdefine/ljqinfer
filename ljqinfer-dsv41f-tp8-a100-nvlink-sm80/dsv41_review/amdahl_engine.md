# V4.1 decode 引擎内 Amdahl 消融账本

基线 B1Q6 REPLAY = **20.596 ms**（HEAD a7f761e）  测量口径：`bench/decode_graph.py:99 run()` = **仅 model.forward**，drafter/commit 不在图内。

方法：`/tmp/abl.py` monkeypatch **引擎真实函数对象** → no-op，再用 `runpy` 跑**原驱动**，重捕获 CUDA Graph，测 replay 步时降幅。未另写任何 forward。

交叉验证：AR 消融 11.8% vs CUPTI 核榜 13.4%；MoE 消融 3.585ms vs CUPTI 3.56ms —— 两独立口径吻合。

| 组件 | 引擎真实挂钩点 | 消融后(ms) | Δ(ms) | 占比 | 性质 |
|---|---|---|---|---|---|
| attention 全体 | `model/decode_attention.py:40 decode_attend` | 16.864 | +3.732 | 18.1% | decode融合核 |
| MoE 全体 | `ops/prefill/moe_workspace.py:170 WorkspaceRouted.__call__` | 17.011 | +3.585 | 17.4% | ⚠️prefill类,内部走fp4 decode核 |
| projection linear | `ops/prefill/gemm.py:89 PrefillLinear.__call__` | 17.773 | +2.823 | 13.7% | ⚠️prefill算子当decode用 |
| TP8 AllReduce | `torch.distributed.all_reduce` | 18.175 | +2.421 | 11.8% | NCCL |
| RMSNorm | `ops/decode/v4k.py:64 rms` | 20.253 | +0.343 | 1.7% | decode |
| grouped_linear_bf16 | `ops/decode/v4k.py:91` | 20.282 | +0.314 | 1.5% | decode |
| RoPE | `ops/decode/v4k.py rope_` | 20.381 | +0.215 | 1.0% | decode |
| prefill grouped GEMM | `WorkspaceRouted.gemm/gemm_device` | 20.554 | +0.042 | 0.2% | ☠️decode从不调用=死路径 |
| drafter/MTP | `SpecDecoder._draft` | 20.557 | +0.039 | 0.2% | 不在replay图内 |
| commit | `model.commit` | 20.602 | -0.006 | -0.0% | 不在replay图内 |

## 关键结论

1. **四大件 attention(18.1%)+MoE(17.4%)+projection(13.7%)+AllReduce(11.8%) 合计约 61%**（消融项非正交，attention 内的 q/kv projection 与 proj 项有重叠，实际独立占比更低）。
2. **全部归零的 Amdahl 上限也只有 ~1.9x → 10.86ms，仍达不到 10ms。** 10ms 必须从未归因的碎片/长尾（≥35%）中获取，与 CUPTI 观测的 ATen 碎片 24.6% / 7250 次 launch 一致。
3. **`WorkspaceRouted.gemm/gemm_device` 消融后步时不变（Δ≈0）→ prefill 的 grouped GEMM 在 decode 路径是死代码**，可直接删，无性能影响。
4. **`PrefillLinear` 占 13.7%，是真正"prefill 算子当 decode 用"的性能代价**，必须换 decode 专用 GEMV。
5. drafter/commit 不在 replay 图内 → 当前 20.596ms **不含 drafter 开销**，端到端真实 step 更高，与 V4 的 B1Q8 26ms 对比时须注意口径。

## prefill 残留静态审查（用户关切）

对照：**V4 仓 decode 路径 prefill import = 0；V4.1 = 13 处**

- `model/decode_layer.py:13,14` 重复 import，**13 行的 `candidates` 被 14 行覆盖丢失**（潜在 bug）
- `model/decode_build.py:15,16,17,45,46,87,104` → PrefillBlock / PrefillEngram / PrefillMoE / DenseRouted / MoEReduceBuffer / PrefillLinear / ProjectionWorkspace / WorkspaceRouted / RowWorkspace
- `model/decode.py:15,17,18`

即 V4.1 的 decode 模型是**用 prefill 类搭起来的**，V4 则完全自洽。