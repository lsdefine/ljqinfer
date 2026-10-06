# Eager prefill 算子入口（DSv4.1 / A2 / TP8）

前20层由 `model/prefill_block.py` 编排。Past分页、slot、历史token、CPU哈希和预取仍由model层管理；设备计算入口如下。

| 入口 | 文件 | 契约 |
|---|---|---|
| hc_prepare / hc_finish | hc.py | BF16[T,4,5120]残差；FP32门控。incoming_pre用于当前子层，返回next_pre用于下一子层 |
| attention_prepare | attention_units.py | 绑定线性权重与norm，返回q/kv及可选index query/weights；无Past访问 |
| source_append | attention_units.py | 显式压缩carry，原位更新；返回压缩KV、index keys及位置 |
| index_select | attention_units.py | 候选打分选择；共享层复用selection由编排负责 |
| sparse_attention | attention_units.py | 显式local/bank/selection/position/meta；内部保留原pack/correction |
| attention_finish | attention_units.py | wo_a/wo_b后FP32 TP sum，再转换目标dtype |
| engram_apply | engram.py | 接收已准备设备行；workspace由构建层分配，输出借用gather缓冲，到下次复用前有效 |
| PrefillMoE | moe.py | 保留shared+routed组合及原舍入；路由算法在residual.route |
| dispatch_quant | moe_units.py | BF16[T,5120]、INT64[T,6]→INT8专家序行、FP32 scale、INT64 counts/order；稳定排序 |
| grouped_linear | moe_units.py | 显式packed W4权重/scale/bias、INT8输入及counts，输出BF16 |
| activation_quant | moe_units.py | BF16[6T,576]及FP32路由权重→量化288列，可补零到padded_dim |
| combine | moe_units.py | BF16[6T,5120]及order→FP32[T,5120]；调用者执行TP归约 |

计算沿用当前NPU stream；通信沿用传入comm及原顺序。除压缩carry和Engram workspace外，入口输入只读；eager临时张量由PyTorch分配器管理。index_select/sparse_attention直接复用attention.py实现，优化时保持CED调用兼容。

## 开发与验收

- 修改目标算子及对应叶kernel，保持shape/dtype、布局、舍入、状态读写和通信顺序。单算子优化先跑叶台架，再串行做TP8集成。
- 单卡台架：`PYTHONPATH=. python scripts/test_prefill_units.py --device 0`。覆盖HC prepare/finish、dispatch、activation及combine；输出逐位比较、NPU event时间和峰值额外显存。GMM及attention/Engram当前通过整模真实权重对照验收。
- TP8对照：先停原后端，再运行 `PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 scripts/test_prefill_encoder.py`。默认完成后退出；`PREFILL_ACCEPT_SERVE=1`通过后继续提供后端服务。参考源码从Git提交aef455f读取，与候选共用一套权重。

2026-10-06：8 ranks，输入分块[1]、[127,2,129]、[513]，逐层残差/门控及KV、index、carry、window、tail共3040项逐位一致。单卡4种长度全部通过。

性能复验：Engine加载前启用expandable_segments，消除共存decode图时的显存分配重试；HC/Attention静态参数预绑定，Engram固定权重乘积与FP32转换移到构建阶段。相同输入8192 tokens、4轮预热后30组成对交替测试，取8 ranks同步墙钟最大值：旧版中位1.319995s（6206 tps），模块化版1.322267s（6195 tps），差+0.17%；均值差+0.31%，仍不能证明严格零退化。两版全部68次运行、8 ranks分配重试均为0。此口径仅前20层encoder，不含CED与缓存。原始记录：A2-3 `/data/prefill_mod_perf_final/rank*.json`；默认分配器慢轮证据在 `/data/prefill_mod_perf/`。

后端原生聊天模板生成及4次无缓存8192-token请求返回200；热态encoder约1.36s，总prefill约1.49s。台架PASS表示数值正确性，不是性能零退化门禁；8k tps尚未达到。

## 解耦叶算子优化（2026-10-06）

本次只做单卡合成张量台架，不加载引擎，不等同于上述历史TP8验收。
可复跑脚本、冻结基线、逐项收益和未完成边界见 [LEAF_OPT_RESULTS.md](LEAF_OPT_RESULTS.md)。

8192-token leaf and TP8 results: [CHUNK8192_OPT_RESULTS.md](CHUNK8192_OPT_RESULTS.md). Full chunk 1.024s target is not yet validated.

## CPU Engram gather

`host_engram.gather(weight, scale, ids)` is a synchronous CPU-only operator returning owned BF16 rows. Hash/history remain in model; no engine dependency in the operator. AArch64 NEON + OpenMP, external Torch extension build cache. Independent test and current TP8 measurements: [HOST_ENGRAM_RESULTS.md](HOST_ENGRAM_RESULTS.md).
