# 解耦叶算子优化：首批已验证结果

本轮落地五处轻量实现优化：RMS、HC门控、Engram门控、MoE分发和激活补零。
**工作流已验证，但“每个模块化算子都优化一档”尚未完成。** 不把未改入口的计时噪声当收益。

## 方法与复跑

- 基线：`57f8f5f6a2a88ac6bfbe1e8d4b3e9a8d3f2a49ad`，脚本直接从Git取冻结Python源码。
- A2-3 / 单NPU / 合成权重；长度1、127、513、2048；4轮预热、30组交替成对计时。
- 52项PASS：输出数值与CPU字节视图一致，重复执行一致，检查显式只读输入；压缩carry也参与对照。
- native库两侧共享，故这是组合改动回归，不是原生kernel独立数学真值。LocalComm为world=1，不覆盖真实HCCL。
- 同目录`leaf_ab_results.json`含源码SHA256、每轮wall/event样本与峰值额外分配；峰值并非全部下降。

```bash
PYTHONPATH=. PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1 \
  /data/apps/ascend-cann9/bin/python scripts/test_prefill_leaf_ab.py \
  --reference 57f8f5f --output /tmp/prefill_leaf_ab.json
```

## 已落地入口收益

以下为基线中位墙钟/候选中位墙钟；大于1表示加速。各入口不可相加推算整模TPS。

| 入口/场景 | T1 | T127 | T513 | T2048 |
|---|---:|---:|---:|---:|
| hc_prepare | 1.105x | 1.085x | 1.094x | 1.073x |
| dispatch_quant | 1.165x | 1.091x | 1.117x | 1.115x |
| activation_quant_320 | 1.169x | 1.088x | 1.183x | 1.139x |
| rms_torch.bfloat16 | 1.057x | 1.066x | 1.071x | 1.063x |
| rms_torch.float32 | 1.076x | 1.069x | 1.088x | 1.064x |
| engram_apply_world1 | 1.101x | 1.102x | 1.126x | 1.030x |
| attention_prepare | 1.036x | 1.051x | 1.068x | 1.043x |
| source_append_r1 | 1.046x | 1.030x | 1.056x | 1.039x |
| source_append_r2 | 0.998x | 1.044x | 1.019x | 1.031x |

## 未完成与拒绝候选

- hc_finish：未改。20核调度在T2048曾测得1.376x，其余长度接近噪声，尚未落地。
- index_select：候选大输入1.054–1.078x，但T1约0.959x，暂不采用。
- combine：Python候选及24核调度大输入退化，拒绝；保留原实现。
- grouped_linear：GMM tuning从{0,1}改{0,0}不满足逐位一致门禁，拒绝；保留原实现。不能由此断言候选数学错误。
- sparse_attention、attention_finish、路由/shared分支：本轮未取得独立优化收益；sparse_attention和GMM不在52项主台架内。
- activation无padding路径未变；其计时差异不计作优化。
- source_append收益来自共用RMS；压缩native kernel未改。
- 未跑8卡真实通信、真实模型权重、CED共享调用或端到端；不宣称整模提速或达到8k TPS。

## 资产与运行状态

- 原始候选、冻结源码、拒绝候选日志：A2-3 `/data/prefill_leaf_opt/`。
- 本轮只改ops/prefill实现、文档与独立台架；未改引擎/model/server。
- 服务保持停止供开发使用；未自动恢复，未修改A2-1/A2-2。
