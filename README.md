# ljqinfer

**Specialized performance. Evolving capabilities.**

An agent-driven inference engine project for DeepSeek, GLM, and Qwen on NVIDIA A100/A800 and Ascend 910B3. Starting from proven implementations and their development history, the agent adapts and optimizes kernels, memory management, and execution for each model, machine, and workload.

**The adaptation process is general-purpose. The resulting engine is specialized.**

Powered by [GA (GenericAgent)](https://github.com/lsdefine/GenericAgent) · [MIT License](LICENSE)

[Performance](#performance) · [How it works](#how-it-works) · [Capabilities](#capabilities) · [Models & deployment](#models--deployment) · [Design principles](PRINCIPLES.md)

## Performance

**5 models · 7 specialized implementations · NVIDIA GPUs and Ascend NPUs**

B1 and B4 are batch sizes 1 and 4. **Decode TPS shows average single-stream generation performance.** Step time captures execution efficiency and batching behavior; TPS captures the resulting generation experience.

| Model | Hardware | Prefill<br>token/s | B1<br>ms/step | B4<br>ms/step | Decode<br>token/s |
|:---|:---|---:|---:|---:|---:|
| GLM-5.2 | 8×A100 | — | 43.17 | 110.12 | **42** |
| DeepSeek-V4-Flash | 8×A100 | 8,018 | 25.80 | 45.01 | **125** |
| Qwen3.8-27B | 4×A800 | 5,696 | 23.74 | 40.43 | **160** |
| Qwen3.8-27B | 4×910B3 | 7,031 | 28.29 | 45.09 | **160** |
| GLM-5.3 | 8×A100 | 5,583 | 43.73 | 96.00 | **70** |
| DeepSeek-V4.1-Flash | 8×A100 | 12,500 | 16.14 | 29.93 | **191** |
| DeepSeek-V4.1-Flash | 8×910B3 | 8,707 | 35.08 | 75.03 | **114** |

Step timings: full decode rounds for GLM-5.2/5.3; decode graph replay for DeepSeek-V4; rank-based timings for Qwen; fixed-width decode for V4.1 on A100; request-log wall time for V4.1 on Ascend. Context lengths, acceptance statistics, and service results are included below.

<details>
<summary><strong>Step timing and aggregation</strong></summary>

- **GLM-5.2 · A100:** 12K context, Q6 full decode round including verify, sampling, draft, commit, and TP synchronization. Development entry: `checkpoint: full MTP decode below 45 ms`.
- **DeepSeek-V4-Flash · A100:** decode graph replay. Development entry: `amdahl (stop-service, MAXSEQ=65536)`.
- **Qwen3.8-27B · A800:** slowest-rank median step time, from the `down_layout` optimization.
- **Qwen3.8-27B · 910B3:** mean of three runs. B1: 28.301 / 28.276 / 28.298 ms; B4: 45.139 / 45.024 / 45.112 ms.
- **GLM-5.3 · A100:** full round, mean rank-max time. Development entry: `Batch GLM53 attention projections and DFlash compute`.
- **DeepSeek-V4.1-Flash · A100:** fixed-width decode A/B measurement. Development entry: `A/B, fixed-width batches`.
- **DeepSeek-V4.1-Flash · 910B3:** mean request-log `wall_ms_per_step`, using 33,403 requests that stayed at B1 and 2,130 that stayed at B4. A2-1 and A2-3 service logs were read on 2026-10-27; requests were selected by `decode_batch_steps`. Single-stream TPS is the arithmetic mean across requests: **113.5446 token/s**, rounded to **114**.

The first six timing records are in the corresponding instance's `GIT_HISTORY.md`. Their overview Decode TPS values are rounded from the long-generation or full-model results below. Prefill values use representative tests for each instance; GLM-5.2 has a blank Prefill entry.

</details>

<details>
<summary><strong>Context-length benchmarks, draft acceptance, and service throughput</strong></summary>

### Prefill throughput and B1 decode timing

| Instance | Prefill (token/s) | B1 decode (ms/step) |
|:---|:---|:---|
| GLM52 · 8×A100 | — | **43.170** @12K, Q6 |
| DSV4F-0731 · 8×A100 | **8,018** @12K; 6,969 @160K | **28.531** @51K; 33.2 @160K |
| Qwen38-27B-DFlash2 · 4×A800 · BF16 | **5,696** @12K | **23.738**, Q8 |
| Qwen38-27B-DFlash2 · 4×910B3 · W8A8 | **7,031** @12K; 4,781 @262K | **28.276–28.301**, Q8 |
| GLM53 · 8×A100 | **5,583** @12K, full 78-layer model | **43.616**, 12-case full round; **45.805** @32K, model host, Q8 |
| DSV41F · 8×A100 | **~12,500** @21,212 / 26,012 tokens, later worker records; 10,665 @36K, earlier test | **15.950**, later B1 fixed-width test; 15.875, earlier model-host test |
| DSV41F · 8×910B3 · W4A8 | **8,707** @128K, cold cache; service-log Prefill **8,698** | **32.50–32.93** @~7–8K, model mean; **33.65–34.73**, full-round mean |

### Output throughput and draft acceptance

| Instance / workload | Output tokens/request | Mean accepted drafts/step | Measured token/s | Estimated token/s |
|:---|---:|---:|---:|---:|
| GLM53 · full-model, 12 cases | — | — | **69.76** | — |
| GLM53 · 32K counting test | 256 | 7.000 | **171.79** cold / **173.01** warm | — |
| GLM52 | 595–1,203 | 1.063 | **41.84** | **47.80** |
| DSV4F-0731 | 633–1,785 | 2.566 | **125.48** | **124.97** |
| Qwen · A800 · BF16 | 625–906 | 3.345 | **159.81** | **183.05** |
| Qwen · 910B3 · W8A8 | 936–1,488 | 4.270 | **160.15** | **186.22** |
| DSV41F · short generation, 5 runs | 64 | 4.727 | **373.69–375.03** | **360.77** |
| DSV41F · long generation, 1 run | 519 | 2.065 | **191.09** | **193.08** |

**Estimated TPS = 1000 / step time (ms) × (1 + mean accepted drafts per step).** Acceptance varies with the prompt and is computed as total accepted drafts divided by total steps. GLM52, DSV4F-0731, and each Qwen instance use 5 long-output logs; measured throughput is total output tokens divided by total decode seconds. Estimates use baseline step times of 43.170 / 28.531 / 23.738 / 28.301 / 15.875 ms, respectively.

The GLM53 32K counting test produces 8 tokens per step with a full-round time of 46.617 ms. The 12-case result is the arithmetic mean of per-case TPS.

### Service throughput

| Instance / workload | Throughput (token/s) | Measurement |
|:---|:---|:---|
| GLM53 · warm-cache HTTP, C1 / C2 / C4 / C8 | **58.19 / 50.56 / 27.47 / 27.28** per request | 256 tokens/request; SSE client decode timing after TTFT; revision `b2bb4a0` |
| Same workload · aggregate end-to-end | **54.02 / 93.88 / 102.95 / 103.33** | Includes queueing and Prefill; C8 = 8 concurrent clients |
| DSV41F · 41 production requests | **165–344**; **218.9** with a warm 128K cache | Revision `298a7ad`; 2.77–5.76 tokens/step including the anchor; range across 41 requests |

</details>

## How it works

Specialized performance depends on the model, hardware, and workload. Quantization, offloading, caching, speculative decoding, and scheduling interact across the execution stack. As requirements change, the implementation needs continuous profiling, modification, and validation.

**People set the goals. The agent adapts the implementation.**

1. **Understand the target.** Inspect model structure, accelerator and host memory, compute capability, transfer bandwidth, and interconnects. Establish latency, throughput, and context-length goals.
2. **Build on experience.** Start from a related implementation and use its development history to guide the next changes.
3. **Optimize jointly.** Adapt quantization, memory offloading, multi-level caches, parallelism, and batching. Rewrite kernels, data layouts, communication, and scheduling where needed.
4. **Measure and iterate.** Validate correctness and end-to-end gains, integrate the effective changes, and profile the next bottleneck.

**Adaptability comes from the agent's ability to keep transforming the engine.** The process carries forward as hardware and objectives evolve.

## Capabilities

### Model- and hardware-specific acceleration

- **Compose acceleration methods.** GLM-5.3 and Qwen3.8 integrate DFlash2 with coordinated draft, verify, commit, and state recovery. DeepSeek implementations use MTP / DSpark.
- **Put existing GPUs to work.** SM80 implementations for A100/A800 specialize low-bit computation, sparse attention, and communication for newer models.
- **Target heterogeneous hardware.** Ascend 910B3 implementations combine W8A8 / W4A8 computation, AscendC kernels, and multi-device execution with device-specific Prefill and Decode paths.

### DRAM caching and speculative decoding, designed together

Host-memory caches preserve reusable computation state. Storage, transfer, and recovery are specialized for each model and machine.

- **Prefix reuse with DFlash2.** The Qwen Ascend implementation combines DRAM prefix caching with speculative decoding to accelerate both history recovery and continued generation.
- **One host copy of shared KV across eight ranks.** GLM-5.3 restores shared KV through pipelined broadcast while retaining rank-local draft shards. Cache payload drops from **768 to 117 KiB/token**, an **84.8% reduction**; 64K cache recovery improves from **~0.69 s to 0.13 s**.
- **Model-specific state recovery.** DeepSeek-V4.1 combines global KV storage, short-tail replay, and shared prefixes to restore local-window and compressed state and reuse conversation branches.

The agent jointly tunes memory offloading, multi-level caching, quantization, and execution against the actual hardware and workload.

## Models & deployment

Each linked implementation includes its build and launch instructions.

| Implementation | Hardware / parallelism | Weights / quantization | Speculative decoding |
|:---|:---|:---|:---|
| [GLM-5.2](ljqinfer-glm52-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 | Huihui abliterated GGUF · UD-Q3_K_M mixed quantization | Native MTP · Q6 |
| [DeepSeek-V4-Flash-0731](ljqinfer-dsv4f-0731-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 | FP8 / FP4 mixed weights | MTP / DSpark |
| [Qwen3.8-27B · A800](ljqinfer-qwen38-27B-dflash2-tp4-a800-bf16/) | 4×A800 PCIe · TP4 | Huihui abliterated · BF16 | DFlash2 · Q8 |
| [Qwen3.8-27B · 910B3](ljqinfer-qwen38-27B-dflash2-tp4-910b3-w8a8/) | 4×Ascend 910B3 · TP4 | W8A8 | DFlash2 · Q8 |
| [GLM-5.3](ljqinfer-glm53-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · TP8 | FP8 + routed MoE G64 INT4 cache | DFlash2 · Q8 |
| [DeepSeek-V4.1-Flash · A100](ljqinfer-dsv41f-tp8-a100-nvlink-sm80/) | 8×A100 NVLink · EP8/TP8 | FP8 / FP4 · image input | DSpark |
| [DeepSeek-V4.1-Flash · 910B3](ljqinfer-dsv41f-tp8-910b3-w4a8/) | 8×Ascend 910B3 · TP8 | W4A8 · image input | DSpark · Q6 |

**DeepSeek-V4.1 on Ascend 910B / 910B3:** [Deployment guide](ljqinfer-dsv41f-tp8-910b3-w4a8/README.md) · [Launch script](ljqinfer-dsv41f-tp8-910b3-w4a8/scripts/serve.sh). Features include B1–B4 dynamic batching, a 512K API context limit, and OpenAI / Anthropic APIs with streaming and non-streaming output.

## Documentation

- [Design principles](PRINCIPLES.md): architecture, state management, and the continuous optimization workflow.
- Instance `README.md` files: model dependencies, build environments, and launch instructions.
- Instance `GIT_HISTORY.md` files: development decisions, experiments, performance results, and implementation history.
