# DSV41F 导入说明

从 node09 正式仓 `/mnt/data/kw/ljqinfer_dsv41f_tp8` 导入。
源提交：`50a5e23b3f9a8b7f07c178ec6c43c94f4166e124`。

DeepSeek-V4.1-Flash，8×A100 NVLink，EP8/TP8，FP8 / FP4 混合权重，DSpark 推测解码；含原生 TP8 图像输入。原始 `README.md` 和开发文档原样保留，其中阶段性“未集成”说明应结合后续提交阅读；图像能力与验证边界见 `IMAGE_SUPPORT.md`。

## 旧仓源码依赖（已核验，不重复打包）

`ops/decode/v4k.py` 硬编码使用 `/mnt/data/kw/ljqinfer_dsv4f_tp8/ops`。
其旧 docstring 虽提及 `DSV4_OPS_DIR`，**当前代码没有读取该环境变量**。
所借用文件为：

- `dsv4_wgemm.cu`、`prefill_moe_cutlass_gemm.cu`、`sparse_attn_paged.cu`
- `paged_io.cu`、`compressor_tail.cu`、`index_score.cu`、`peer_ar_ipc.cu`
- `prefill_moe_cutlass_gemm.h`
- `third_party/cutlass/include/` 与 `third_party/cutlass/LICENSE`

上述 **811 个文件**与本汇总仓 `../ljqinfer-dsv4f-0731-tp8-a100-nvlink-sm80/ops/` 逐个 SHA256 完全一致。旧仓源提交为 `9e19d0471c3ffc4f8d0eea78422e29aba080642d`；清单见 `V4_DEPENDENCIES.json`（路径相对于旧仓 `ops/`）。

部署时保留原路径，或将该旧仓路径链接至汇总仓已有 DSV4F-0731 实例；也可在部署副本中明确调整 `v4k.py` 的 `_DIR`。构建前创建对应 `ops/.build` 目录。
**仅复制本实例目录而不提供旧仓依赖，无法完整构建 decode。** 为保持正式源提交字节一致，本次未改写路径或旧 docstring。

## 部署前提

- 原模型默认 `/mnt/data/kw/models/DeepSeek-V4.1-Flash`；权重和 tokenizer 另行准备。
- Python 依赖见 `pyproject.toml`，首次构建需 CUDA toolkit，环境须匹配 A100 / PyTorch。
- TP8 worker 为 `strategy.decode_worker`（torchrun），HTTP 入口为 `server.server`；默认 RPC 62001 / HTTP 8000。不要与 GLM53 或现有 worker 争用 GPU / 端口。
- 图像样例与限制见 `IMAGE_SUPPORT.md`；外网部署前配置鉴权。

## 归档与验证边界

- 导入固定提交全部 **1001 个已追踪文件**，源码字节及可执行位保持原样，无嵌套 `.git`。
- `GIT_HISTORY.md` 导出 **262 条完整提交消息**；`IMPORT_MANIFEST.json` 记录源提交及逐文件 SHA256 / Git 模式。
- 不含模型权重、编译产物、运行日志及未追踪实验目录；历史外部实验材料不属于本次快照。
- 本次完成源码字节、Python 语法及上述旧仓依赖核验；未重编 CUDA、运行 GPU 推理或复测速度，未改动 node09 服务。
- 历史性能数字不视为当前快照的重新验收；这是机器专用源码快照，不是免配置安装包。
