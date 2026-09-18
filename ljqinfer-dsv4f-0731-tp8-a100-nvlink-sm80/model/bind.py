# -*- coding: utf-8 -*-
"""Bind the weights.load_tp8() device tree onto arch.Transformer (loop-TP8, world_size=1).

Layout rules (verified against tp8 manifest, 2026-09-01):
  replica  -> bind shard[0] (lives on cuda:0): wq_a, wkv, all norms, compressor(ape/wkv/wgate/norm),
              gate(weight/bias/tid2eid), hc_* params, final norm
  tp dim0  -> ColumnParallel shard list: wq_b, wo_a(fp8 -> bf16 once at load, pre-cat [g,r,d]), indexer.wq_b, weights_proj(bf16),
              embed, head, expert/shared w1+w3;  attn_sink is a tp0 param -> cat to full [n_heads]
  tp dim1  -> RowParallel shard list: wo_b, expert/shared w2
  routed experts: fp4 (int8-packed [out, in//2] + E8M0 scale), stacked banks [256, ...] per rank
  shared experts / big linears: fp8 E4M3 + scale (attached as tensor attr .scale)
"""
import torch


def _r(t):
    """Leaf accessor: dist mode gives bare tensors, loop mode gives shard lists."""
    return t if torch.is_tensor(t) else t[0]


def _q(pair, replica=False):
    """QuantPair -> tensor (replica) or shard list, with .scale attached when present."""
    if torch.is_tensor(pair.weight):  # dist mode: already this rank's shard
        w = pair.weight
        if pair.scale is not None:
            w.scale = pair.scale
        return w
    ws = list(pair.weight)
    ss = list(pair.scale) if pair.scale is not None else [None] * len(ws)
    out = []
    for w, s in zip(ws, ss):
        if s is not None:
            w.scale = s
        out.append(w)
    return out[0] if replica else out


def _fp8lin(pair, replica=False):
    """fp8 Linear weight -> bf16 once at load (e8m0 scale => exact), tagged for ops.fp8_linear.
    bf16 pairs (no scale) pass through untouched."""
    from ops import dequant_fp8_bf16
    def one(w, s):
        if w.dtype != torch.float8_e4m3fn:
            return w
        d = dequant_fp8_bf16(w, s)
        d.qdq_block = 128
        return d
    if torch.is_tensor(pair.weight):
        return one(pair.weight, pair.scale)
    ss = list(pair.scale) if pair.scale is not None else [None] * len(pair.weight)
    out = [one(w, s) for w, s in zip(pair.weight, ss)]
    return out[0] if replica else out


def _p(param, t):
    """Rebind nn.Parameter storage; cast to the dtype the module declared."""
    param.data = t if t.dtype == param.dtype else t.to(param.dtype)


def _norm(mod, sh):
    mod.weight.data = _r(sh).float()


def _plain_lin(lin, sh):
    """Plain (replicated) Linear; honor fp32 modules (compressor), keep fp8/bf16 as-is."""
    w = _r(sh)
    lin.weight = w.float() if lin.dtype == torch.float32 and w.dtype != torch.float32 else w


def _compressor(c, CW):
    _p(c.ape, _r(CW.ape))
    _norm(c.norm, CW.norm)
    _plain_lin(c.wkv, CW.wkv)
    _plain_lin(c.wgate, CW.wgate)


def _bank(experts, B):
    """Routed experts: per-rank stacked banks [n_experts, out_shard, in(_packed)]."""
    if torch.is_tensor(B.w1.weight):  # dist mode: bank is this rank's [n_exp, out_shard, in]
        for name in ('w1', 'w2', 'w3'):
            pair = getattr(B, name)
            for eid in range(len(experts)):
                w = pair.weight[eid]
                if pair.scale is not None:
                    w.scale = pair.scale[eid]
                getattr(experts[eid], name).weight = w
        return
    for name in ('w1', 'w2', 'w3'):
        pair = getattr(B, name)
        ss = list(pair.scale) if pair.scale is not None else [None] * len(pair.weight)
        for eid in range(len(experts)):
            lst = []
            for w_r, s_r in zip(pair.weight, ss):
                w = w_r[eid]
                if s_r is not None:
                    w.scale = s_r[eid]
                lst.append(w)
            getattr(experts[eid], name).weight = lst


def _expert(e, EW):
    e.w1.weight = _q(EW.w1)
    e.w2.weight = _q(EW.w2)
    e.w3.weight = _q(EW.w3)


def bind(model, W):
    dist_mode = torch.is_tensor(W.embed)
    if dist_mode:  # replicate the full embedding table on every rank (1.06 GiB): decode embed without all_reduce
        import torch.distributed as dist
        parts = [torch.empty_like(W.embed) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, W.embed.contiguous())
        model.embed.weight = torch.cat(parts, dim=0)
    else:
        model.embed.weight = list(W.embed)
    model.head.weight = W.head if dist_mode else list(W.head)
    _norm(model.norm, W.norm)
    _p(model.hc_head_fn, _r(W.hc_head_fn))
    _p(model.hc_head_scale, _r(W.hc_head_scale))
    _p(model.hc_head_base, _r(W.hc_head_base))

    assert len(model.layers) == len(W.layers), (len(model.layers), len(W.layers))
    assert len(model.mtp) == len(W.mtp), (len(model.mtp), len(W.mtp))
    pairs = list(zip(model.layers, W.layers)) + [(blk, M.block) for blk, M in zip(model.mtp, W.mtp)]
    for blk, L in pairs:
        _norm(blk.attn_norm, L.attn_norm)
        _norm(blk.ffn_norm, L.ffn_norm)
        for n in ('hc_attn_fn', 'hc_attn_scale', 'hc_attn_base',
                  'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base'):
            _p(getattr(blk, n), _r(getattr(L, n)))

        a, AW = blk.attn, L.attn
        if dist_mode:
            _p(a.attn_sink, AW.attn_sink)  # this rank's head slice
        else:
            dev = AW.attn_sink[0].device
            _p(a.attn_sink, torch.cat([t.to(dev) for t in AW.attn_sink]))
        a.wq_a.weight = _fp8lin(AW.wq_a, replica=True)
        _norm(a.q_norm, AW.q_norm)
        a.wq_b.weight = _fp8lin(AW.wq_b)
        a.wkv.weight = _fp8lin(AW.wkv, replica=True)
        _norm(a.kv_norm, AW.kv_norm)
        ws = _q(AW.wo_a)
        ws = ws if isinstance(ws, list) else [ws]
        from ops import dequant_fp8_bf16
        dev = a.attn_sink.device
        def _wo(w):
            if w.dtype == torch.float8_e4m3fn:
                return dequant_fp8_bf16(w, w.scale).to(dev)  # e8m0 scale => exact, once at load
            if w.dtype == torch.bfloat16:
                return w.to(dev)
            raise RuntimeError(f"wo_a: unexpected shard dtype {w.dtype}")
        a.wo_a.weight = torch.cat([_wo(w) for w in ws], 0).view(a.n_local_groups, a.o_lora_rank, -1).contiguous()
        a.wo_b.weight = _fp8lin(AW.wo_b)
        if AW.compressor is not None:
            _compressor(a.compressor, AW.compressor)
        if AW.indexer is not None:
            assert a.indexer is not None, f'layer {L.idx}: ckpt has indexer, arch does not'
            _compressor(a.indexer.compressor, AW.indexer.compressor)
            a.indexer.wq_b.weight = _fp8lin(AW.indexer.wq_b)
            wp = AW.indexer.weights_proj
            a.indexer.weights_proj.weight = wp if dist_mode else list(wp)

        m, MW = blk.ffn, L.ffn
        _p(m.gate.weight, _r(MW.gate.weight))
        if MW.gate.bias is not None and m.gate.bias is not None:
            _p(m.gate.bias, _r(MW.gate.bias))
        if MW.gate.tid2eid is not None:
            _p(m.gate.tid2eid, _r(MW.gate.tid2eid))
        _bank(m.experts, MW.experts)
        _expert(m.shared_experts, MW.shared_experts)

        # fused MoE prefill tables: raw per-rank views (loop: lists, dist: this rank's tensors)
        def _l(t):
            return t if dist_mode or t is None else list(t)
        def _ws(pair):
            return _l(pair.weight), _l(pair.scale)
        def _wsd(pair):  # shared experts: fp8 -> bf16 once at load (kernel takes bf16, no cache)
            from ops import dequant_fp8_bf16
            if dist_mode:
                return dequant_fp8_bf16(pair.weight, pair.scale)
            return [dequant_fp8_bf16(w, s) for w, s in zip(pair.weight, pair.scale)]
        m.fused_norm = _l(L.ffn_norm)
        m.fused_hc = tuple(_l(getattr(L, n)) for n in ('hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base'))
        m.fused_gate = (_l(MW.gate.weight), _l(MW.gate.bias), _l(MW.gate.tid2eid))
        m.fused_bank = tuple(_ws(getattr(MW.experts, n)) for n in ('w1', 'w3', 'w2'))
        m.fused_shared = tuple(_wsd(getattr(MW.shared_experts, n)) for n in ('w1', 'w3', 'w2'))
    # DSpark stages: block weights bound in the loop above; stage-specific extras here.
    for blk, M in zip(model.mtp, W.mtp):
        if M.main_proj is not None:
            blk.main_proj.weight = _fp8lin(M.main_proj, replica=True)
            _norm(blk.main_norm, M.main_norm)
        if M.norm is not None:
            _norm(blk.norm, M.norm)
            _p(blk.hc_head_fn, _r(M.hc_head_fn))
            _p(blk.hc_head_scale, _r(M.hc_head_scale))
            _p(blk.hc_head_base, _r(M.hc_head_base))
            blk.markov_head.markov_w1.weight = M.markov_w1 if dist_mode else list(M.markov_w1)
            blk.markov_head.markov_w2.weight = M.markov_w2 if dist_mode else list(M.markov_w2)
            _plain_lin(blk.confidence_head.proj, M.confidence_proj)
        blk.embed = model.embed
        blk.head = model.head


def check_bound(model):
    """Walk modules; report Linear-likes whose weight is still None."""
    bad = [n for n, mod in model.named_modules()
           if hasattr(mod, 'weight') and mod.weight is None]
    return bad
