# ljqinfer

**ljqinfer 是一个随着模型、硬件和环境自我变换的推理引擎。**

由 agent 针对模型与硬件生成专用引擎，再继承已有实例与演化历史，持续优化速度。**Powered by [GA (GenericAgent)](https://github.com/lsdefine/GenericAgent)** · [MIT License](LICENSE)。

适配具体模型与硬件组合，包括 **A100 / sm80 稀疏注意力**、**Ascend 910B3 / W8A8 + DFlash2**。[设计理念](PRINCIPLES.md)

## 速度

### Prefill 吞吐与 Decode 步时

| 实例 | Prefill（tok/s） | Decode（ms/step，B1） |
|---|---:|---:|
| GLM52 · 8×A100 | — | **43.170** @12K，Q6 |
| DSV4F-0731 · 8×A100 | **8,018** @12K；6,969 @160K | **28.531** @51K；33.2 @160K |
| Qwen38-27B-DFlash2 · 4×A800 · BF16 | **5,696** @12K | **23.738**，Q8 |
| Qwen38-27B-DFlash2 · 4×910B3 · W8A8 | **7,031** @12K；4,781 @262K | **28.276–28.301**，Q8 |
| **DSV41F · 8×A100** | **10,665** @36K | **15.875**（模型 host 步时，900-token生成测试） |

### 输出吞吐与平均 accept

| 实例 | 输出长度（tokens/请求） | 平均 draft accept | 实测（tok/s） | 按基准步时估算（tok/s） |
|---|---:|---:|---:|---:|
| GLM52 | 595–1,203 | 1.063 | **41.84** | **47.80** |
| DSV4F-0731 | 633–1,785 | 2.566 | **125.48** | **124.97** |
| Qwen · A800 · BF16 | 625–906 | 3.345 | **159.81** | **183.05** |
| Qwen · 910B3 · W8A8 | 936–1,488 | 4.270 | **160.15** | **186.22** |
| **DSV41F · 短测，5次** | 64 | 4.727 | **373.69–375.03** | **360.77** |
| DSV41F · 长生成，1次 | 519 | 2.065 | **191.09** | **193.08** |

**估算 TPS = 1000 / 步时(ms) × (1 + 平均 draft accept)**。accept 受 prompt 影响，按总接受 draft 数 / 总步数计算；前四项各取5条长输出日志，实测为总输出 / 总 Decode 秒。估算依次采用 43.170 / 28.531 / 23.738 / 28.301 / 15.875 ms 基准步时。

## 实例与量化

| 实例 | 硬件 / 并行 | 模型精度 | 推测解码 |
|---|---|---|---|
| [GLM52](ljqinfer-glm52-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | Huihui GLM-5.2 abliterated GGUF，**UD-Q3_K_M 混合量化** | 原生 MTP，Q6 verify |
| [DSV4F-0731](ljqinfer-dsv4f-0731-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | DeepSeek-V4-Flash-0731，**FP8 / FP4 混合权重** | MTP / DSpark |
| [qwen38-27B-dflash2 · BF16](ljqinfer-qwen38-27B-dflash2-tp4-a800-bf16/) | 4×A800 PCIe · TP4 | Huihui Qwen3.8-27B abliterated，target **BF16** | DFlash2，Q8 verify |
| [qwen38-27B-dflash2 · W8A8](ljqinfer-qwen38-27B-dflash2-tp4-910b3-w8a8/) | 4×Ascend 910B3 · TP4 | Qwen3.8-27B，target **W8A8** | DFlash2，Q8 verify |
| [GLM53](ljqinfer-glm53-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | GLM-5.3，原始 FP8 模型 + routed MoE **G64 INT4** 缓存 | DFlash2，Q8 verify |
| [DSV41F](ljqinfer-dsv41f-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · EP8/TP8 · sm80 | DeepSeek-V4.1-Flash，**FP8 / FP4 混合权重**；原生图像输入 | DSpark |

GLM53 与 DSV41F 已收录 node09 正式源码快照。[GLM53 部署说明](ljqinfer-glm53-tp8-a100-nvlink-sm80/README.md) · [DSV41F 导入与旧仓依赖](ljqinfer-dsv41f-tp8-a100-nvlink-sm80/IMPORT.md) · [DSV41F 图像能力](ljqinfer-dsv41f-tp8-a100-nvlink-sm80/IMAGE_SUPPORT.md)。上方速度表保留原历史测试口径；本次源码导入未复测性能，GLM53 未增列缺乏统一口径的速度数字。

各实例的 `GIT_HISTORY.md` 保留完整开发消息，包括优化思路、实验结果与速度记录。
