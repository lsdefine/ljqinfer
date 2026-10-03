# DSpark (MTP) Drafter 设计与已核实事实  (2026-09-13, 调研自官方 inference/model.py + 论文 §2.4.3)

来源真值: /mnt/data/kw/models/DeepSeek-V4.1-Flash/inference/model.py (官方参考实现) 行 1021-1160, 1270-1285;
配置真值: inference/config.json == model/v41_config.json (已含全部 dspark_* 字段)

## 1. 参数
n_mtp_layers=3, dspark_block_size=5, dspark_noise_token_id=128799,
dspark_target_layer_ids=[37,38,39], dspark_markov_rank=256,
dspark_n_routed_experts=128, dspark_n_activated_experts=3, hc_mult=4, dim=5120, vocab=129280, window=128
=> B1Q6 = 1 个已确认 token + 5 个草稿 = 每步 verify 6 个位置。

## 2. 权重清单 (真 checkpoint index.json 核实)
每层 mtp.N (N=0,1,2) 与主干层同构:
  attn.{attn_sink[64], q_norm[1280], kv_norm[512], wq_a[1280,5120], wq_b[32768,1280],
       wkv[512,5120], wo_a[8192,4096], wo_b[5120,8192]}  (无 indexer, 无 compressor)
  ffn.{gate.weight/bias/bias_vl, experts.0..127.w1/w2/w3, shared_experts.w1/w2/w3}
  hc_attn_{base[24],fn[24,20480],scale[3]}, hc_ffn_{...}, attn_norm, ffn_norm
额外: mtp.0.main_norm[5120], mtp.0.main_proj[5120,15360](=3 个 target 层拼接),
      mtp.2.norm[5120], mtp.2.markov_head.embed[129280,256], mtp.2.markov_head.head[129280,256],
      mtp.2.confidence_head.proj[1,5376] (=5120+256)
weights.placement 已支持: markov_head=TP0, main_proj/main_norm/norm/confidence_head=replicated,
dspark 专家数 128 已在 placement 内区分 (group=='mtp')。

## 3. 权威前向语义
主干: 对 layer in [37,38,39] 收集**该层 attention 的输入** h.mean(dim=2) (hc 维求均),
      main_hidden = cat(..., dim=-1) -> [T, 15360]。仓内 PrefillModel/DecodeModel 已输出 main_hidden。
mtp[0].forward_embed(main_hidden, input_ids):
  main_x = main_norm(main_proj(main_hidden))
  draft_ids = [上一步已确认 token, noise_token x (block_size-1)]  长度 5
  x = embed(draft_ids); x = x.unsqueeze(2).repeat(1,1,hc_mult,1); pre_mix = identity
每层 DSparkAttention (compress_ratio==0, ring=128):
  main_kv = kv_norm(wkv(main_x)) + rope(pos=start_pos) + fp8 roundtrip
  环形缓存**每步只写 main_kv**:  window_kv_cache[:, start_pos % 128] = main_kv
  草稿 q/kv 用 pos = start_pos+1 .. start_pos+5
  bank = cat(window_kv_cache 全 128 行, 草稿 kv 5 行)  -> 133 行
  ix = get_dspark_topk_idxs = cat(arange(min(128, start_pos+1)), 128+arange(5))
     ** 所有 5 个草稿 query 共享同一可见集, 非因果 (草稿彼此可见, semi-AR) **
  sparse_attn(q, bank, ix, sink, scale) -> o -> wo_a(分组 einsum) -> wo_b
  prefill(start_pos==0) 只播种窗口 KV: seqlen<=128 直接写; 否则 cutoff=seqlen%128 环形 split 写最后 128 行
mtp[-1].forward_head(x, pre_mix, input_ids):
  x = hc_pre(x, pre_mix); logits = head(norm(x))  -> [b, 5, V]
  for i in range(5): (bias, emb) = markov_head(out_ids[:,i]); logits[:,i] += bias;
                     out_ids[:,i+1] = Gumbel-max sample(logits[:,i])
  confidence = confidence_head(cat(x, stack(embs), dim=-1).float())  -> 每位置接受概率

## 4. 实现决策 (本仓)
- 复用 ops/prefill/sparse_attn.attend: bank=133 行, ix=topk_idxs, 无 compressed/index 分支。
- 复用 PrefillBlock 的 mHC (mixes/collapse_norm/expand) 与 MoE, 但 MoE 前缀需参数化为 mtp.N.ffn
  且专家数走 dspark_n_routed_experts=128 / top3。
- past 已预留: default_layer_views() 给 layer 40/41/42 建 mode='dspark' view;
  WindowPast ring=128 已按 N_ATTENTION_LAYERS=43 分配。prefill_attention.attend 显式拒绝 dspark 层(独立路径)。

## 5. 施工步骤
1) model/dspark_attention.py: 环形窗口 + 草稿 attention (对拍朴素 torch 参考)
2) model/dspark.py: 3 层 block + forward_embed/forward_head + markov + confidence
3) tests/test_dspark_*.py: 与官方语义直译参考对拍 (小尺寸随机权重)
4) 真权重 B1Q6 端到端: draft < 20ms 检查点测速
