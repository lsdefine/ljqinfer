# -*- coding: utf-8 -*-
"""A/B: fused MoE ffn vs original oracle path, per-layer rel diff.
nohup python tc_moe_ab.py > /tmp/tc_moe_ab.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, torch

# ---- torch oracle for MoE path (moved out of model/arch.py; arch keeps only weight containers) ----
from typing import Optional, Tuple
import torch.nn.functional as F
import torch.distributed as dist
from model.arch import linear, world_size


def gate_fwd(g, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = linear(x.float(), g.weight.float())
    if g.score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif g.score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    original_scores = scores
    # Bias shifts scores for expert selection (topk) but does not affect routing weights.
    if g.bias is not None:
        scores = scores + g.bias
    if g.hash:
        indices = g.tid2eid[input_ids]
    else:
        indices = scores.topk(g.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if g.score_func != "softmax":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= g.route_scale
    return weights, indices


def expert_fwd(e, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    dtype = x.dtype
    gate = e.w1(x).float()
    up = e.w3(x).float()
    if e.swiglu_limit > 0:
        up = torch.clamp(up, min=-e.swiglu_limit, max=e.swiglu_limit)
        gate = torch.clamp(gate, max=e.swiglu_limit)
    x = F.silu(gate) * up
    if weights is not None:
        x = weights * x
    return e.w2(x.to(dtype))


def moe_fwd(m, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shape = x.size()
    x = x.view(-1, m.dim)
    weights, indices = gate_fwd(m.gate, x, input_ids.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=m.n_routed_experts).tolist()
    for i in range(m.experts_start_idx, m.experts_end_idx):
        if counts[i] == 0:
            continue
        expert = m.experts[i]
        idx, top = torch.where(indices == i)
        y[idx] += expert_fwd(expert, x[idx], weights[idx, top, None])
    y += expert_fwd(m.shared_experts, x)
    if world_size > 1:
        dist.all_reduce(y)
    return y.type_as(x).view(shape)


t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind
    from model.args_dsv4 import make_args

    W = wcache.load('tp8')
    log('weights loaded')
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    bind(model, W)
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0')
    torch.set_default_device('cuda:0')
    log('bound')

    ids = json.load(open('/tmp/c1_ids.json'))[:32]
    tok = torch.tensor([ids], device='cuda:0')
    torch.manual_seed(0)

    for li in [0, 3, 20]:
        blk = model.layers[li]
        x = torch.randn(1, 32, args.hc_mult, args.dim,
                        device='cuda:0', dtype=torch.bfloat16) * 0.5
        # original oracle path
        residual = x
        h, post, comb = blk.hc_pre(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
        hn = blk.ffn_norm(h)
        h = moe_fwd(blk.ffn, hn, tok)
        ref = blk.hc_post(h, residual, post, comb)
        # fused path
        out = blk.fused_ffn(x, tok)
        d = (out.float() - ref.float())
        rel = d.norm() / ref.float().norm()
        log(f'L{li} rel={rel.item():.6f} max_abs={d.abs().max().item():.6f} '
            f'ref_norm={ref.float().norm().item():.3f}')

        # routing ids A/B for score-based layers
        if li >= 3:
            from ops import moe_rank_fused_prefill_fp4
            hf = hn.reshape(32, -1)
            gw_ref, gid_ref = gate_fwd(blk.ffn.gate, hf)
            m = blk.ffn
            gw, gb, t2e = m.fused_gate
            r = 0
            W = gw[r] if isinstance(gw, list) else gw
            ids_pre = W.new_empty((0,), dtype=torch.long)
            rbias = gb[r] if isinstance(gb, list) else gb
            (w1w, w1s), (w3w, w3s), (w2w, w2s) = m.fused_bank
            s1, s3, s2 = m.fused_shared  # bf16, dequantized at bind time
            g = (lambda t: t[r]) if isinstance(gw, list) else (lambda t: t)
            xf = x.reshape(32, args.hc_mult, args.dim).to(W.device)
            out5 = moe_rank_fused_prefill_fp4(
                xf, blk.hc_ffn_fn.to(W.device), blk.hc_ffn_scale.to(W.device),
                blk.hc_ffn_base.to(W.device), g(m.fused_norm), W, rbias, ids_pre,
                g(w1w), g(w1s), g(w3w), g(w3s), g(w2w), g(w2s),
                g(s1), g(s3), g(s2), blk.norm_eps)
            ids_k = out5[4]
            same = (ids_k.to(gid_ref.device).sort(-1)[0] == gid_ref.sort(-1)[0]).float().mean()
            log(f'L{li} ids_match={same.item():.4f} ids_k_shape={tuple(ids_k.shape)} '
                f'ref_shape={tuple(gid_ref.shape)}')
            log(f'L{li} ids_k[0]={ids_k[0].tolist()} min={ids_k.min().item()} max={ids_k.max().item()}')
            log(f'L{li} gid_ref[0]={gid_ref[0].tolist()} min={gid_ref.min().item()} max={gid_ref.max().item()}')
            log(f'L{li} gw_ref[0]={[round(v,4) for v in gw_ref[0].float().tolist()]}')
    log('AB DONE')
except Exception:
    import traceback; traceback.print_exc()
    log('AB FAILED')
