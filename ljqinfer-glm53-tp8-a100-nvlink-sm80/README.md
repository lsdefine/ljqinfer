# GLM53 · TP8 · A100 NVLink / sm80

源仓：node09 `/mnt/data/kw/ljqinfer_glm53_tp8`；源提交：`ed9501e507cf75712e6729b460ac173fdf47839e`。

## 运行与构建

- GLM-5.3，8×A100 NVLink，TP8；DFlash2，Q8 verify（anchor + 7 drafts）。
- 原 FP8 模型 `/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8`、routed MoE G64 INT4 缓存 `/mnt/data2/kw/glm53_int4_tp8`、draft `/mnt/data2/kw/GLM-5.3-DFlash2` 均为外部依赖；不是全模型 INT4，原模型不可删除。
- 在匹配的 Linux / CUDA / PyTorch / Triton 环境运行 `python -m ops.build_selected`；算子产物和 `ops/selected_ops.lock.json` 在部署机生成，其余 JIT 构建按源码执行。
- 启动入口 `start_tp8_service.sh`：默认 Python `/mnt/data/kw/anaconda3/bin/python`，engine RPC 62001；部署前核对路径、依赖、GPU、端口和 API 鉴权。
- 默认采样 `temperature=0.5, top_p=0.95`；显式 `temperature=0` 保持贪心。采样路径限制回复内再次生成 assistant 角色头，这是结构约束，不是未约束的原始采样分布。

## 最小归档

- 108 个运行/构建源码与必要资源，保持源文件字节和 Git 模式；另保留本说明、`IMPORT_MANIFEST.json` 溯源清单及 `GIT_HISTORY.md` 完整开发消息。
- 不收录测试、基准脚本、报告、实验文档、模型权重、缓存、编译产物和运行日志；旧材料仍可从 Git 历史追溯。
- 本次同步检查文件哈希、Python 语法及已选算子的构建源完整性；不在 Windows 重跑 GPU 验收，不改动线上服务。
