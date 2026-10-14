# GLM53 · TP8 · A100 NVLink / sm80

从 node09 正式仓 `/mnt/data/kw/ljqinfer_glm53_tp8` 导入。
源提交：`07c90519d7e804ff1ba50f8eae381c3b170dccec`。

## 实现与部署前提

- GLM-5.3，8×A100 NVLink / TP8；DFlash2，Q8 verify（anchor + 7 drafts）。
- 原始 FP8 模型加 routed MoE **G64 INT4** 缓存；不是全模型 INT4。gate/up/down 转换说明见 `model/INT4_CONVERSION.md`。
- 原模型默认 `/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8`，INT4 缓存默认 `/mnt/data2/kw/glm53_int4_tp8`，draft 默认 `/mnt/data2/kw/GLM-5.3-DFlash2`。原 FP8 模型仍是必需依赖，不能因完成转换而删除。
- 在匹配的 Linux / CUDA / PyTorch / Triton 环境中，从实例根目录运行 `python -m ops.build_selected`，生成算子产物及 `ops/selected_ops.lock.json`。其余 JIT 算子仍按原实现构建。
- 原启动入口 `start_tp8_service.sh` 默认 Python 为 `/mnt/data/kw/anaconda3/bin/python`，engine RPC 为 62001；启动前检查机器路径、Python 依赖、GPU 和端口占用。HTTP / API 配置以 `server/` 为准，外网部署前配置鉴权。

## 归档与验证边界

- 导入固定提交全部 **173 个已追踪文件**，源码字节及可执行位保持原样，无嵌套 `.git`。
- `GIT_HISTORY.md` 导出 **74 条完整提交消息**；`IMPORT_MANIFEST.json` 记录源提交及逐文件 SHA256 / Git 模式。
- 不含模型权重、转换缓存、编译产物、运行日志及未追踪实验材料；历史文档可能引用源机器上的外部材料。
- 本次验证源码字节、Python 语法、selected operator 构建源文件完整性；未重编 CUDA、运行 GPU 推理或复测性能，未改动 node09 服务。
- 保留机器专用路径；这是稳定开发仓的源码快照，不是免配置安装包。历史性能与数值验收的范围以各条原始记录为准。

## 本次同步更新

- 同步至 `07c9051`：包括前缀缓存修复、逐请求温度与 thinking 控制、批量逆 CDF / CUDA 采样、长输出自动留证。
- 源机实测 B1 完整 decode 步时：贪心 44.89ms、T=1 45.52ms（同题短样本，非全负载保证）；真实 8192-token 输出已验证自动落盘。详细条件与限制见 `tests/SAMPLING_INVERSE_CDF.md`。
- 长输出日志留在源机，不纳入代码归档；未宣称偶发超长 thinking 重复问题已修复。
