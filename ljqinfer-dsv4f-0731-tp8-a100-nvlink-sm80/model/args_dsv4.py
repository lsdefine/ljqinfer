# -*- coding: utf-8 -*-
"""Real DeepSeek-V4-Flash-0731 ModelArgs, mapped from
/mnt/data/kw/models/DeepSeek-V4-Flash-0731/config.json (HF keys -> arch.ModelArgs)."""
from .arch import ModelArgs

# 43 main layers + 3 MTP blocks
_COMPRESS_RATIOS = (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0)


POOL_TOKENS = 4 * 1024 * 1024   # 4M-token shared paged-KV budget, ~30.1 GiB/rank


def make_args(max_batch_size: int = 4, max_seq_len: int = 393216,
              pool_tokens: int = POOL_TOKENS) -> ModelArgs:
    # 384k = 393216. Production default context window (was 1048576).
    # Decode step time still grows mildly with the CAPTURED max_seq_len
    # (26.0ms @32k / 26.5 @128k / 28.2 @512k / 28.8 @1M) even after the sparse_attn
    # keff-truncation fix flattened the operator itself (F_sparse_attn = 1.07-1.08ms
    # at EVERY length). The residual is diffuse, not attributable to any single
    # kernel, so we cap the captured graph at the context we actually serve.
    return ModelArgs(
        temperature=0.0,  # greedy draft/head sampling: rank-consistent + deterministic (Gumbel per-rank noise diverged pos_t)
        max_batch_size=max_batch_size,
        pool_tokens=pool_tokens,
        max_seq_len=max_seq_len,
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype="fp4",
        scale_dtype="fp8",
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=2048,          # moe_intermediate_size
        n_layers=43,                 # num_hidden_layers
        n_hash_layers=3,             # num_hash_layers
        n_mtp_layers=3,              # num_nextn_predict_layers (mtp.0..2 in manifest, config.N_MTP)
        n_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,       # num_experts_per_tok
        score_func="sqrtsoftplus",   # scoring_func
        route_scale=1.5,             # routed_scaling_factor
        swiglu_limit=10.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,            # qk_rope_head_dim
        norm_eps=1e-6,               # rms_norm_eps
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,             # sliding_window
        compress_ratios=_COMPRESS_RATIOS,
        compress_rope_theta=160000.0,
        original_seq_len=65536,      # rope_scaling.original_max_position_embeddings
        rope_theta=10000.0,
        rope_factor=16,              # rope_scaling.factor
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(40, 41, 42),
        dspark_markov_rank=256,
    )
