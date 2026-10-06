# DeepSeek-V4.1-Flash · TP8 · Ascend 910B3 · W4A8

## 功能

- B1～B4 动态批处理、DSpark Q6 推测解码。
- 引擎容量配置 1M token，API 长度上限 512K；NPU KV 池 2M token。
- 主存前缀缓存 50 GiB，环境变量 `LJQINFER_COLD_BYTES` 可调。
- 新 prefill 超过 128K 时，等待当前 decode 批次完成后上车。
- OpenAI `/v1/chat/completions`、Anthropic `/v1/messages`，支持普通和流式输出。
- 图片使用 base64 Data URL；含图请求跳过主存前缀缓存。
- 模型名 `deepseek-v41-flash`，默认 API key：`devkey`。

## 运行

环境：Linux/aarch64、8×910B3、CANN、PyTorch/torch_npu，准备模型权重与算子构建产物。

源部署路径 `/data/ljqinfer_dsv41f_tp8`，模型路径 `/data/models/DeepSeek-V4.1-Flash`。
使用 `/data/apps/ascend-cann9/bin/python` 所在虚拟环境，按 `model/prefill_build.py`、`ops/decode/build.sh` 准备构建。

```bash
LJQ_TQE=1 bash scripts/serve.sh start
bash scripts/serve.sh status
```

API 端口 `8000`，引擎 RPC `127.0.0.1:62001`；日志位于 `/data/logs/`。

## 本轮优化与验证

- 前20层 eager prefill 按算子语义模块化；索引评分与 Top512 使用 CANN LightningIndexer。
- A2-3 TP8，131,072 tokens / 16×8192 chunk：2轮预热、4轮稳态中位 **15.053秒 / 8,707 tok/s**；八卡零分配重试。不含 CED、decode、TTFT。
- 8000服务实测131,029输入token：模型prefill **15.064秒 / 8,698 tok/s**，TTFT **15.987秒**。
- 5分钟服务观察59次主动请求全部成功；16项索引边界回归、6项完整生成语义用例通过。此结论不等于与旧实现logits逐位一致。
- 部署必须使用本机 CANN 9.0.1 + torch/torch_npu 2.10 环境；同步源码同时更新匹配的 AscendC 产物（本轮 `ops/kernels/libhc.so` 已变化）。主机JIT桥接扩展由新源码在本机重建；不拷贝权重/GMM缓存。

## A2-1 / A2-2 同版部署验收

两机部署同一来源提交 `84ecd2c`，141项源码与41项运行产物逐文件SHA256一致；旧实例完整保留，可回滚。各持续约308秒、66次请求全部成功，包含8k/36k/128k检索以及B1/B2/B4混合流式/非流式请求；各27次health全部200，八个rank与前端持续存活，未见异常堆栈、OOM或HTTP 5xx。

| 实例 | 约36k模型prefill | 约128k模型prefill | 约128k吞吐 |
|---|---:|---:|---:|
| A2-1 | 4.131秒 | 15.002秒 | 8,734 tok/s |
| A2-2 | 4.142秒 | 15.072秒 | 8,693 tok/s |

以上为单次真实API性能观测，不等同于重复独占基准或长期稳定性保证。归档仅保留脱敏API_KEY，其他来源源码逐字节一致；不包含运行.so和模型缓存。

## 来源

源提交 `84ecd2cc2b9e8f310c7d58f9754ce3c350dbeb29`；归档默认 key 调整为 `devkey`。
速度见[主 README](../README.md)，469 条开发提交消息见 [GIT_HISTORY.md](GIT_HISTORY.md)。
