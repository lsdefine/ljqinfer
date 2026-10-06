# ljqinfer

**ljqinfer 是一个随着模型、硬件和环境自我变换的推理引擎。**

由 agent 针对模型与硬件生成专用引擎，再继承已有实例与演化历史，持续优化速度。**Powered by [GA (GenericAgent)](https://github.com/lsdefine/GenericAgent)** · [MIT License](LICENSE)。

适配具体模型与硬件组合，包括 **A100 / sm80 稀疏注意力**、**Ascend 910B3 / W8A8、W4A8**。[设计理念](PRINCIPLES.md)

## 速度

以下为各版本实测数据，测试上下文与计时方式列于表中。

### Prefill 吞吐与 Decode 步时

| 实例 | Prefill（tok/s） | Decode（ms/step，B1） |
|---|---:|---:|
| GLM52 · 8×A100 | — | **43.170** @12K，Q6 |
| DSV4F-0731 · 8×A100 | **8,018** @12K；6,969 @160K | **28.531** @51K；33.2 @160K |
| Qwen38-27B-DFlash2 · 4×A800 · BF16 | **5,696** @12K | **23.738**，Q8 |
| Qwen38-27B-DFlash2 · 4×910B3 · W8A8 | **7,031** @12K；4,781 @262K | **28.276–28.301**，Q8 |
| GLM53 · 8×A100 | **5,583** @12K（78 层整模） | **43.616**（12-case 完整 round）；**45.805** @32K（模型 host，Q8） |
| DSV41F · 8×A100 | **约 12,500** @21,212 / 26,012 tokens（后续 worker 记录）；10,665 @36K（旧测） | **15.950**（后续 B1 固定批宽测试）；15.875（旧模型 host 测试） |
| [DSV41F · 8×910B3 · W4A8](ljqinfer-dsv41f-tp8-910b3-w4a8/) | **8,707** @128K，前20层 eager、零缓存；API模型计时 **8,698** | **32.50–32.93** @约7–8K（模型均值）；**33.65–34.73**（完整均值） |

### 输出吞吐与平均 accept

| 实例 | 输出长度（tokens/请求） | 平均 draft accept | 实测（tok/s） | 按基准步时估算（tok/s） |
|---|---:|---:|---:|---:|
| GLM53 · 整模 12-case | — | — | **69.76**（每 case TPS 算术均值） | — |
| GLM53 · 32K 数数短测 | 256 | 7.000 | **171.79**（冷）/ **173.01**（热） | — |
| GLM52 | 595–1,203 | 1.063 | **41.84** | **47.80** |
| DSV4F-0731 | 633–1,785 | 2.566 | **125.48** | **124.97** |
| Qwen · A800 · BF16 | 625–906 | 3.345 | **159.81** | **183.05** |
| Qwen · 910B3 · W8A8 | 936–1,488 | 4.270 | **160.15** | **186.22** |
| **DSV41F · 短测，5次** | 64 | 4.727 | **373.69–375.03** | **360.77** |
| DSV41F · 长生成，1次 | 519 | 2.065 | **191.09** | **193.08** |

**估算 TPS = 1000 / 步时(ms) × (1 + 平均 draft accept)**。accept 受 prompt 影响，按总接受 draft 数 / 总步数计算；GLM52 / DSV4F-0731 / 两个 Qwen 项各取5条长输出日志，实测为总输出 / 总 Decode 秒。估算依次采用 43.170 / 28.531 / 23.738 / 28.301 / 15.875 ms 基准步时。

GLM53 32K 数数测试每步输出 8 token，完整步时 46.617 ms；12-case 吞吐取算术均值。

### 后续服务吞吐记录

| 实例 / 记录 | Decode 吞吐 | 口径 |
|---|---:|---|
| GLM53 · 同任务热缓存 HTTP，C1 / C2 / C4 / C8 | 每请求 **58.19 / 50.56 / 27.47 / 27.28 tok/s** | 256 token/请求；SSE 客户端计时排除 TTFT，历史版本 `b2bb4a0` |
| 同上，端到端总吞吐 | **54.02 / 93.88 / 102.95 / 103.33 tok/s** | 包含排队和 prefill；C8 为 8 路客户端并发 |
| DSV41F · 41 条线上请求观察 | **165–344 tok/s**；128K 热缓存 **218.9 tok/s** | 源提交 `298a7ad`；每步输出 2.77–5.76 token（含 anchor），41 条请求范围 |

## 实例与量化

| 实例 | 硬件 / 并行 | 模型精度 | 推测解码 |
|---|---|---|---|
| [GLM52](ljqinfer-glm52-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | Huihui GLM-5.2 abliterated GGUF，**UD-Q3_K_M 混合量化** | 原生 MTP，Q6 verify |
| [DSV4F-0731](ljqinfer-dsv4f-0731-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | DeepSeek-V4-Flash-0731，**FP8 / FP4 混合权重** | MTP / DSpark |
| [qwen38-27B-dflash2 · BF16](ljqinfer-qwen38-27B-dflash2-tp4-a800-bf16/) | 4×A800 PCIe · TP4 | Huihui Qwen3.8-27B abliterated，target **BF16** | DFlash2，Q8 verify |
| [qwen38-27B-dflash2 · W8A8](ljqinfer-qwen38-27B-dflash2-tp4-910b3-w8a8/) | 4×Ascend 910B3 · TP4 | Qwen3.8-27B，target **W8A8** | DFlash2，Q8 verify |
| [GLM53](ljqinfer-glm53-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 · sm80 | GLM-5.3，原始 FP8 模型 + routed MoE **G64 INT4** 缓存 | DFlash2，Q8 verify |
| [DSV41F](ljqinfer-dsv41f-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · EP8/TP8 · sm80 | DeepSeek-V4.1-Flash，**FP8 / FP4 混合权重**；原生图像输入 | DSpark |
| [DSV41F · 910B3](ljqinfer-dsv41f-tp8-910b3-w4a8/) | 8×Ascend 910B3 · TP8 | DeepSeek-V4.1-Flash，**W4A8**；图片输入 | DSpark，Q6 verify |

各实例的 `GIT_HISTORY.md` 保留完整开发消息，包括优化思路、实验结果与速度记录。
