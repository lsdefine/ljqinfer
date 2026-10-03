# 性能来源与计时口径

README 的新增指标来自 node09 原始提交消息、JSON 和日志；本次未重跑 GPU。各项可能早于导入 HEAD，不表示每个后续提交均复测。

## GLM53

- **Prefill 5,583.08 tok/s**：`6f55a13`，12,288 tokens、history=0、78 层真实权重、eager、2 warmups + 5 samples，median wall 2.200936s，不含 HTTP。
  原始 `/mnt/data/kw/ljqinfer_glm53_tp8/reports/engine_history_integration.json` 的计时字段见 [PERFORMANCE_EVIDENCE.json](PERFORMANCE_EVIDENCE.json)。
- **完整 round 43.616257ms；每 case TPS 均值 69.756461**：`114dd0d`，12 prompts（含 code/thinking）、fresh paired suite、同步边界，不含 prefill / HTTP。TPS 为算术均值，不是加权吞吐；见原始 [DFLASH_FUSION_INTEGRATION.md](ljqinfer-glm53-tp8-a100-nvlink-sm80/DFLASH_FUSION_INTEGRATION.md)。
- **32K 冷缓存 host 45.805444ms / wall 46.617417ms / decode 171.794 tok/s**：`6985fcd` 与 `split_mla_service.log`。
  输入 32,770、输出 256 tokens、32 steps，accepted=224、proposed=224，即每步 7 draft 全接受。数数任务、reasoning off，不能代表普通生成。
  热缓存命中 32,769 tokens：host 45.501879ms、wall 46.272945ms、decode 173.007 tok/s。
  基线 host 83.253380ms、wall 85.539291ms、decode 93.586 tok/s，缓存命中 36 tokens；两次缓存状态不同，不将总耗时差作为等条件端到端收益。
  源目录 `/mnt/data2/kw/glm53_int4_tp8/service_audit/prefill_capacity/`，文件 `step_timing_service.log`、`split_mla_service.log`、`decode_split_{baseline,candidate,validation}.json`。
  [PERFORMANCE_EVIDENCE.json](PERFORMANCE_EVIDENCE.json) 仅保存 benchmark metrics，不收录请求正文、客户端 IP 或请求 ID。

### 历史 HTTP 黑盒报告

来自 `/mnt/data2/kw/glm53_int4_tp8/service_audit/blackbox_current/REPORT.md`，标题标注 `b2bb4a0`，早于 split-MLA 测试。同机 loopback HTTP，不包含公网延迟。
单请求生成速度估计为 `(completion_tokens-1)/(最后内容事件-首个内容事件)`，SSE 批量发送有小偏差。端到端总吞吐含排队和 prefill。以下为原报告摘录：

## 相同任务热缓存对照：每请求256输出token
| 客户端并发 | 单请求生成tok/s均值（排除TTFT） | TTFT范围s | 端到端总tok/s | 最慢完成s |
|---|---|---|---|---|
|1|58.19|0.344–0.344|54.02|4.739|
|2|50.56|0.377–0.378|93.88|5.453|
|4|27.47|0.606–0.612|102.95|9.947|
|8|27.28|0.497–10.480|103.33|19.817|

4→8客户端并发的总吞吐几乎不增长，8请求观察到两组完成，不能将8当成引擎同时B8。
不同任务512输出测试：热轮单流65.49tok/s，4并发每请求26.86–35.73tok/s，端到端总101.07tok/s；8并发首字最长23.38s，总103.22tok/s。首轮较慢记录完整保留，未当异常删除。

报告另记录部分标准选项不支持、输出格式问题及此前 OOM；性能样本不构成完整功能认证，也不用于断言后续导入版本仍有相同问题。

## DSV41F

- **Prefill 约 12,500 tok/s**：`6fc3897`，worker chunk 改为 12,288；21,212 tokens / 1.714s，26,012 tokens / 2.050s。与旧 36K 测试不同，不能直接计算跨测试收益。
- **B1 step 15.95ms**：`25c0642`，A/B；同记录 B2/B3/B4 为 20.67/25.55/29.32ms。该提交未明确上下文长度及完整计时边界，不称为 GPU-only 或 HTTP 时延。
- **线上 165–344 tok/s；128K 热缓存 218.9 tok/s**：`298a7ad`，记录 41 条 live requests 的计时一致性及流量范围。2.77–5.76 是含 anchor 的每步输出量，不是 draft accepted 数；范围不是均值，不与其他提交步时拼算。
- 原有 36K / 900-token / 64-token / 519-token 测试保留并标注旧测；本次未复核其他 case 的既有历史数据。无图像性能承诺。

## 原始提交消息

以下保留完整消息及限制条件，亦见各实例 GIT_HISTORY.md。

### glm53 · 6f55a13

```text
6f55a13ff05fc8d5f098cbece74bebcbde0d947b
2026-10-02 07:36:00 +0000
Integrate SM80 prefill optimizations; full-model 2.39826 -> 2.20094 s

Register softmax/max reduction, PAGE64 specialization and exact fused RMS.
TP8 Q / paired KV token-owned buffers shared across sequential layers.
Projection specialization restricted to the 12288-token first chunk.

Same-path Engine.prefill, 78 real layers, 2 warmups + 5 samples/version:
92045ca 2.398256 s / 5123.72 tok/s -> 2.200936 s / 5583.08 tok/s.
Latency -8.23%, throughput +8.97%, saved 197.32 ms.
All 8 ranks: logits and six feature tensors bitwise equal for 12K,
subsequent33, verify8, reset/first17; 48 RMS cases pass.
Rank0 allocated memory +170.75 MiB. No HTTP restart or constant changes.
Full-model 2-second target remains unmet.
Evidence: reports/engine_history_integration.json; service_audit engine_history_*
```

### glm53 · 114dd0d

```text
114dd0d0261e6d3119869df3f3aab376b51c465a
2026-10-02 11:53:47 +0000
Integrate bitwise DFlash fusion: engine round 44.10->43.62 ms

Share six-layer RoPE frequencies in draft and KV append, preserve BF16
product rounding and original RMS, fuse greedy path selection.
Fresh node09 TP8 12-case suite: 44.103706->43.616257 ms (-1.105%).
All 12 improved; all 8 ranks output hashes, steps, acceptance histograms
and cold/hot/reset cache outputs match baseline e6ab07c.
58 leaf checks and all-rank n=1..8 KV append graph regression passed.
Isolated draft 3.595->3.214 ms is separate from engine round timing.
HTTP not restarted, 40ms unmet, long-context/concurrency not certified.
See DFLASH_FUSION_INTEGRATION.md for evidence and scope.
```

### glm53 · 6985fcd

```text
6985fcd545fe9478835121f9ee158219a175d67e
2026-10-03 05:26:03 +0000
Reuse split MLA for Q8 decode: 32K API host step 83.3 to 45.8ms; validate FP32 oracle and B4 smoke
```

### dsv41f · 6fc3897

```text
6fc389744130986d696bf583dd6c46755f5c991a
2026-09-20 06:11:29 +0000
prefill: chunk at the engine's ceiling, not a fifth of it

ModelExecution already accepts up to 12288 tokens per prefill chunk and
defaults to exactly that.  The worker passed CHUNK = 2048 and overrode
it, so every prompt was cut into six times as many pieces as the engine
was willing to take.

Each piece costs about 0.118s of setup that has nothing to do with its
length -- a 24-token prompt pays it in full.  A 10686-token prefill was
six pieces, so 0.71s of the 1.55s it took was setup repeated five times
over.  Measured at the ceiling: 21212 tokens in 1.714s and 26012 in
2.050s, both about 12.5k tok/s against 6.9k before, which is the rate
the model layer computes at.

The reported chunk count was also measuring the wrong thing.  It divided
the whole prompt, cache hits included, so a request that prefilled 2613
tokens reported seven chunks; it now divides what was actually computed.

Host memory per rank is unchanged.  Device use goes from 65.0 to 71.9
GiB of 80 for the wider workspace.  27/27 api_suite passes.
```

### dsv41f · 25c0642

```text
25c064293cd3ffe820b4a31bb26760fb86a97cf5
2026-09-20 04:50:56 +0000
decode/index: stop -inf filling the index score buffer at pool capacity

scores() allocated a fresh [t, max_rows] fp32 tensor per layer per step and
filled it with -inf, where max_rows is the index pool capacity (~1M rows on
the 1M profile) rather than anything derived from the live sequence.  That
fill is a capacity-sized write no consumer ever observes:

  * topk_select_post_kernel clamps its scan to NL = (pos+1)/ratio and, as
    its own comment states, skips the [NL, N) tail entirely -- both the
    float4 body and the scalar remainder bound on NL, never N.
  * cand_blocks._bkey only admits blocks below `newest`, whose row range
    lies inside [0, lens); _score writes every row below that bound.

So the tail carried -inf purely for the benefit of readers that clamp
themselves.  Reuse a persistent per-(t, max_rows) buffer instead, which
also pins the address across graph replays.

ab.sh: b=1 16.14->15.95ms  b=2 21.02->20.67  b=3 25.85->25.55
       b=4 29.93->29.32 (thr 216.1->219.1 tok/s), split b=4 29.55->29.10
Combined with the preceding mask removal, versus the run before both:
       b=1 -2.1%  b=2 -2.8%  b=3 -3.4%  b=4 -4.2%
MTP accept rates unchanged across all seven cells (1.12/1.34/1.41/1.27/
0.58/1.19/1.11); tests/api_suite.py 27/27 with determinism_repeat identical.
```

### dsv41f · 298a7ad

```text
298a7ad79d8cba4f11d1a6922f6ec33d67deb72d
2026-09-20 10:18:34 +0000
decode: report decode_tps, drop the second copy of the step clock

engine_decode_seconds and wall_ms_per_step were the same measurement
written twice: across 41 live requests output/engine_decode_seconds and
output/(steps*wall_ms_per_step) agreed to 1.0000 on every row.  Neither
of them was the number anyone actually reads, so every caller divided it
back out by hand -- and tests/log_report.py already banded a decode_tps
column that the engine never emitted.

Emit decode_tps from the lane's own produced count and drop
engine_decode_seconds.  remote_strategy forwards the new key;
tests/api_audit rebuilds the decode seconds it needs from
decode_steps * wall_ms_per_step, which is the same quantity it used
before.  server/service.py already computed a service-side decode_tps
and now has the engine-side value override it, which is the tighter
of the two.

Observed range on live traffic: 165-344 tok/s, set almost entirely by
MTP acceptance (2.77-5.76 tokens/step) and nearly flat in context
length -- 128k warm decodes at 218.9 tok/s against 236.8 at 4k.
```
