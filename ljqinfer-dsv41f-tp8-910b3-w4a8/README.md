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
LJQ_TQE=0 bash scripts/serve.sh start
bash scripts/serve.sh status
```

API 端口 `8000`，引擎 RPC `127.0.0.1:62001`；日志位于 `/data/logs/`。

## 来源

源提交 `43b40e136e6073706465ab9b8f6aa162c5e50f24`；归档默认 key 调整为 `devkey`。
速度见[主 README](../README.md)，452 条开发提交消息见 [GIT_HISTORY.md](GIT_HISTORY.md)。
