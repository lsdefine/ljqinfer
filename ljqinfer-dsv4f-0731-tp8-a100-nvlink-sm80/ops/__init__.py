import torch
"""ops: compiled CUDA kernels (hot paths only)."""

_m = None


def _mod():
    global _m
    if _m is None:
        from ops.build import load_wgemm
        _m = load_wgemm()
    return _m


def dequant_fp8_bf16(weight, scale):
    """weight [N,K] fp8 e4m3 + ue8m0 block scale -> bf16 [N,K]. Pure; call once at load."""
    return _mod().dequant_fp8_bf16(weight, scale)


def fp8_linear(x, weight):
    """fp8 Linear leaf. x [..., K] bf16; weight bf16 [N,K] pre-dequantized from fp8+e8m0
    (lossless) and tagged weight.qdq_block == 128. Activation is qdq'd per 128-block
    (pow2 scale, e4m3 sat) then bf16 GEMM with fp32 accumulate. Pure, no cache."""
    x2 = x.reshape(-1, x.shape[-1])
    return _mod().fp8_linear(x2.contiguous(), weight).reshape(*x.shape[:-1], weight.shape[0])


def rope_inplace(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place RoPE on x[..., :] (bf16 view [B,S,rd] / [B,S,H,rd]); freqs_cis complex64 [S, rd/2]. Returns x."""
    return _mod().rope_inplace(x, freqs_cis, inverse)


def wo_a_grouped(o, w):
    """o [..., G, D] bf16, w [G, R, D] bf16 (pre-cat at bind) -> [..., G, R] bf16. Pure."""
    G, Rr, D = w.shape
    y = _mod().wo_a_grouped(o.reshape(-1, G, D).contiguous(), w)
    return y.view(*o.shape[:-1], Rr)


def hc_pre_norm(x, hc_fn, hc_scale, hc_base, norm_w, eps):
    """Fused hc_pre + rmsnorm. x [..., hc, d] bf16 -> (x_normed [..., d] bf16, post [..., hc] f32,
    comb [..., hc, hc] f32). eps is shared by the rsqrt and the sinkhorn (norm_eps == hc_eps). Pure."""
    hc, d = x.shape[-2], x.shape[-1]
    lead = x.shape[:-2]
    post, comb, xn = _mod().hc_fused_pre(x.reshape(-1, hc, d).contiguous(), hc_fn, hc_scale, hc_base,
                                         norm_w, eps, False)
    return xn.view(*lead, d), post.view(*lead, hc), comb.view(*lead, hc, hc)


def hc_post(branch, residual, post, comb):
    """y[j] = post[j]*branch + sum_i comb[i,j]*residual[i]  (arch.hc_post semantics: comb^T @ residual).
    branch [..., d] bf16, residual [..., hc, d] bf16, post [..., hc], comb [..., hc, hc] -> [..., hc, d] bf16. Pure.
    The kernel contracts comb's LAST index, so comb is transposed here."""
    hc, d = residual.shape[-2], residual.shape[-1]
    y = _mod().hc_post_fused_bf16(branch.reshape(-1, d).contiguous(),
                                  residual.reshape(-1, hc, d).contiguous(),
                                  post.reshape(-1, hc).float().contiguous(),
                                  comb.reshape(-1, hc, hc).float().contiguous(), True)
    return y.view(residual.shape)


def rms_norm(x, weight, eps):
    """x [..., D] bf16, weight [D] fp32 -> bf16. Pure. Leading dims flattened (view)."""
    d = x.shape[-1]
    return _mod().rms_norm(x.reshape(-1, d), weight, eps).view(x.shape)


def bf16_gemm(x, weight):
    """x [M,K] bf16 @ weight [N,K] bf16 .T -> [M,N] bf16. Pure, no cache."""
    return _mod().bf16_gemm(x, weight)


def scatter3_rows(res, kvr, scr, rows, x, kv, sc):
    """One launch for res[rows]=x, kvr[rows]=kv, scr[rows]=sc (same row map)."""
    return _mod().scatter3_rows(res, kvr, scr, rows, x, kv, sc)


def add2_bf16_f32(a, b, y):
    """y[f32] = float(a[bf16]) + float(b[bf16]); bit-identical to a.float()+b.float()."""
    return _mod().add2_bf16_f32(a, b, y)


def moe_rank_fused_fp4(*args):
    """Fused per-rank MoE decode (no D2H sync; CUDA-graph safe). Same outputs as prefill variant."""
    return _mod().moe_rank_fused_fp4(*args)


def moe_rank_fused_prefill_fp4(*args):
    """Fused per-rank MoE prefill: hc-mix norm + router + routed fp4 experts
    + shared fp8 expert. Returns (post, comb, y, shared_out, ids)."""
    return _mod().moe_rank_fused_prefill_fp4(*args)


# ---------------------------------------------------------------- attention leaves
def sparse_attn_flat(q, kv, sink, topk_idxs, scale, kv_compress=None):
    """Drop-in for kernels_torch.sparse_attn on the broken TC kernel.
    q:[b,s,h,512] bf16; kv:[b,n,512] bf16 (main); kv_compress:[b,c,512] or None;
    topk_idxs:[b,s,K] (-1 pad; ids >= n address kv_compress[id-n]); sink:[h].
    Non-paged: one page per sequence (page=n), page_table=arange(b)."""
    import torch
    b, s, h, d = q.shape
    assert d == 512, (q.shape,)
    n = kv.shape[1]
    pool = kv.contiguous()                                   # [b, n, 512] -> page = n
    table = torch.arange(b, device=q.device, dtype=torch.int64).view(b, 1)
    if kv_compress is None or kv_compress.shape[1] == 0:
        cpool, ctable = pool, table                          # never addressed (ids < n)
    else:
        cpool, ctable = kv_compress.contiguous(), table
    idxs = topk_idxs.long().contiguous()
    sink = sink.float().contiguous()
    if h <= 8:                                               # kernel: warp w owns head w (8 heads max)
        out = _mod().sparse_attn_paged(q.contiguous(), pool, table, cpool, ctable,
                                       sink, idxs, n, float(scale), None)
        return out.view(b, s, h, d)
    # single-process oracle (h == 64): chunk over 8-head groups
    outs = [_mod().sparse_attn_paged(q[:, :, i:i + 8].contiguous(), pool, table, cpool,
                                     ctable, sink[i:i + 8], idxs, n, float(scale), None).view(b, s, -1, d)
            for i in range(0, h, 8)]
    return torch.cat(outs, dim=2)


def sparse_attn_ring_paged(q, ring_kv, cpool, ctable, sink, idxs, scale, keff=None):
    """Decode variant of sparse_attn_flat reading the past in place (no gather).
    q:[1,s,h,512] bf16; ring_kv:[ring,512] (one page, ids < ring address ring row id);
    cpool:[pages,cpage,512] + ctable:[n_pages] int64 page table (ids >= ring address
    compressed row id-ring); idxs:[1,s,K] int64 (-1 pad); sink:[h]."""
    import torch
    b, s, h, d = q.shape
    assert d == 512 and b == 1, (q.shape,)
    pool = ring_kv.unsqueeze(0)                              # [1, ring, 512] -> page = ring
    table = torch.zeros(1, dtype=torch.int64, device=q.device)
    total = ring_kv.shape[0]
    sink = sink.float().contiguous()
    idxs = idxs.contiguous()
    if h <= 8:
        out = _mod().sparse_attn_paged(q.contiguous(), pool, table, cpool, ctable,
                                       sink, idxs, total, float(scale), keff)
        return out.view(b, s, h, d)
    outs = [_mod().sparse_attn_paged(q[:, :, i:i + 8].contiguous(), pool, table, cpool,
                                     ctable, sink[i:i + 8], idxs, total, float(scale), keff).view(b, s, -1, d)
            for i in range(0, h, 8)]
    return torch.cat(outs, dim=2)


def sparse_attn_pool_paged(q, pool, page_table, cpool, ctable, sink, idxs, scale, keff=None):
    """A8 batched-row form of sparse_attn_ring_paged: ONE launch for B rows.
    q:[1,B*Q,h,512] bf16 (row b = tokens [b*Q,(b+1)*Q)); pool:[n_slots,ring,512] the
    whole main_kv pool (page = ring); page_table:[B,1] int64 = ring page (slot) of
    each row; cpool:[pages,cpage,512] + ctable:[B,n_pages] int64 per-row compressed
    page tables; idxs:[1,B*Q,K] int64 (-1 pad; ids < ring -> ring row, ids >= ring
    -> compressed row id-ring).  The kernel maps token -> batch = token // Q and
    indexes page_table / ctable per batch, so each token computes exactly what the
    per-row launch computed (same values).  B == 1 is the same launch as
    sparse_attn_ring_paged with pool = main_kv[slot:slot+1]."""
    b, s, h, d = q.shape
    assert d == 512 and b == 1 and pool.dim() == 3 and page_table.dim() == 2 and ctable.dim() == 2, \
        (q.shape, pool.shape, page_table.shape, ctable.shape)
    assert s % page_table.shape[0] == 0, (s, page_table.shape)
    total = pool.shape[1]                                     # ring rows per page
    sink = sink.float().contiguous()
    idxs = idxs.contiguous()
    if h <= 8:
        out = _mod().sparse_attn_paged(q.contiguous(), pool, page_table, cpool, ctable,
                                       sink, idxs, total, float(scale), keff)
        return out.view(b, s, h, d)
    outs = [_mod().sparse_attn_paged(q[:, :, i:i + 8].contiguous(), pool, page_table, cpool,
                                     ctable, sink[i:i + 8], idxs, total, float(scale), keff).view(b, s, -1, d)
            for i in range(0, h, 8)]
    return torch.cat(outs, dim=2)


def index_topk(q, kv, weights, ratio, start_pos, offset, index_topk):
    """Indexer tail on broken leaves. q:[b,s,h,d] (fp4-sim'd, rotated), kv:[b,n,d],
    weights:[b,s,h] (scaled). Returns int32 [b,s,K] compressed ids (+offset, -1 masked),
    matching arch.Indexer.forward semantics (mask only bites when start_pos == 0).
    Streams over query rows so the [b,rows,h,n] bf16 score tile stays under
    INDEX_TOPK_BYTES (default 1 GiB); a full 12k chunk at 1M context is ~50 GB."""
    import os
    b, s, h = q.shape[:3]
    n = kv.shape[1]
    budget = _INDEX_TOPK_BYTES
    rows = max(1, min(s, budget // max(1, b * h * n * 2)))
    if rows < s and _INDEX_TOPK_ROWSHARD:
        import torch.distributed as dist
        W = dist.get_world_size() if dist.is_initialized() else 1
        if W > 1 and rows >= W:
            rows -= rows % W   # keep every full segment W-divisible so the row-shard path applies
    if rows >= s:
        return _index_topk_rows(q, kv, weights, ratio, start_pos, offset, index_topk)
    outs = [_index_topk_rows(q[:, r0:r0 + rows], kv, weights[:, r0:r0 + rows], ratio,
                             start_pos + r0, offset, index_topk)
            for r0 in range(0, s, rows)]
    return torch.cat(outs, dim=1)


# --- indexer knobs -------------------------------------------------------------
# Both of these change NUMERICS (reduction / chunking order), so they are module
# constants, never environment variables: production has exactly one code path.
# Tests that want the reference path assign to these names explicitly, e.g.
#     import ops; ops._INDEX_TOPK_MODE = "ref"
_INDEX_TOPK_MODE  = "kernel"   # kernel | kernel_bf16 | ref_reduce | ref_topk | ref
_INDEX_TOPK_BYTES = 1 << 30    # score-tile budget for prefill row chunking
# Row-sharded reduce: reduce_scatter the fp32 score over query rows, run top-k on the
# local 1/W rows, all_gather the int32 ids. Halves the NCCL bytes vs all_reduce and
# removes the Wx redundant top-k/sort; fp32 reduction order differs from all_reduce
# (~1e-6 rel), so it sits behind the same numerics gate as the kernel mode.
_INDEX_TOPK_ROWSHARD = True

def _index_topk_rows(q, kv, weights, ratio, start_pos, offset, index_topk):
    import torch.distributed as dist
    import os
    mode = _INDEX_TOPK_MODE
    b, s = q.shape[:2]
    score = torch.einsum("bshd,btd->bsht", q, kv)                       # bf16 [b,s,h,n]
    if mode in ("ref_reduce", "ref"):
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(s // ratio, device=q.device).repeat(s, 1) >= torch.arange(1, s + 1, device=q.device).unsqueeze(1) // ratio
            score += torch.where(mask, float("-inf"), 0)
    else:
        score = _mod().index_score_reduce(score, weights, ratio, start_pos)  # f32 [b,s,n]
        if mode == "kernel_bf16":
            score = score.to(torch.bfloat16)   # match ref's bf16 rounding before all_reduce
    W = dist.get_world_size() if dist.is_initialized() else 1
    k = min(index_topk, kv.shape[1])
    if (W > 1 and _INDEX_TOPK_ROWSHARD and mode == "kernel" and b == 1):
        # Row-sharded path: reduce_scatter rows across TP ranks, local top-K, all_gather ids.
        # Rows padded with zero score to a multiple of W (pad rows are discarded).
        s_pad = (s + W - 1) // W * W
        if s_pad != s:
            score = torch.cat([score, score.new_zeros(b, s_pad - s, score.shape[-1])], dim=1)
        rows = s_pad // W
        rk = dist.get_rank()
        local = torch.empty(b, rows, score.shape[-1], dtype=score.dtype, device=score.device)
        dist.reduce_scatter_tensor(local, score.contiguous())
        idx = _mod().topk_select_post(local.float(), k, ratio, offset, start_pos + rk * rows)
        valid = idx >= 0
        sc = torch.gather(local.float(), -1, (idx - offset).clamp_min(0).long())
        sc = torch.where(valid, sc, torch.full_like(sc, float("-inf")))
        order = sc.argsort(dim=-1, descending=True, stable=True)
        idx = torch.gather(idx, -1, order).contiguous()
        out = torch.empty(W, rows, k, dtype=idx.dtype, device=idx.device)
        dist.all_gather_into_tensor(out, idx.view(rows, k))
        return out.view(1, s_pad, k)[:, :s]
    if W > 1:
        dist.all_reduce(score)
    if mode in ("ref_topk", "ref"):
        idx = score.topk(k, dim=-1)[1]
        if start_pos == 0:
            mask = idx >= torch.arange(1, s + 1, device=q.device).unsqueeze(1) // ratio
            return torch.where(mask, -1, idx + offset)
        return idx + offset
    idx = _mod().topk_select_post(score.float(), k, ratio, offset, start_pos)
    if mode != "kernel_unsorted":
        # radix top-k is unordered; sparse_attn accumulation order follows idx, so
        # reorder by score desc (masked -1 last) to mirror torch.topk ordering.
        valid = idx >= 0
        sc = torch.gather(score.float(), -1, (idx - offset).clamp_min(0).long())
        sc = torch.where(valid, sc, torch.full_like(sc, float("-inf")))
        order = sc.argsort(dim=-1, descending=True, stable=True)
        idx = torch.gather(idx, -1, order)
    return idx


def attn_decode_fused_r0(x, freq, tok, ring_kv, m, window: int):
    """Fused base-layer decode attention (broken attn_rank_sparse_decode_fp8 idea, new-past ring ABI).
    x:[T,dim] bf16 (after hc_pre_norm), freq:[T,32] c64, tok:[T] int64 device, ring_kv:[ring,KD] (written).
    Returns wo_b partial [T,dim] bf16 -- caller does the TP all_reduce (RowParallelLinear semantics)."""
    return _mod().attn_decode_fused_r0(x.contiguous(), freq.contiguous(), tok, ring_kv,
                                       m.wq_a.weight, m.q_norm.weight, m.wq_b.weight,
                                       m.wkv.weight, m.kv_norm.weight,
                                       m.wo_a.weight, m.wo_b.weight, m.attn_sink,
                                       m.n_local_heads, m.n_local_groups, int(window),
                                       float(m.softmax_scale), float(m.eps))
