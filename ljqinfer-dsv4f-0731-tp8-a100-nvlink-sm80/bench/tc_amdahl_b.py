# -*- coding: utf-8 -*-
"""Amdahl breakdown of the batched decode step across B = 1 / 2 / 4.

Two orthogonal views, both measured (never estimated):
  (1) LAYER-CLASS view  -- record_function tags on OPS_G[0/4/128] (the three attention
      variants = the three layer classes) plus mtp.* for the MTP tail.
  (2) OPERATOR view     -- CUPTI per-kernel aggregation of the captured CUDA graph replay,
      bucketed by kernel name (nccl / moe fp4 / dense gemm / indexer / sparse attn / ...).

Model+operator layer only: we drive model.step_g_batch directly, no model_api / scheduler /
service involved.  Rows use DIFFERENT prompt lengths (the real multi-sequence case).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, sys, json, time, traceback
from collections import defaultdict
import torch
import torch.distributed as dist

RANK = int(os.environ.get('RANK', '0'))
t0 = time.time()


def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)


N_PROF = int(os.environ.get('N_PROF', '3'))
BLIST = [int(x) for x in os.environ.get('BLIST', '1,2,4').split(',')]
Q = 8
BMAX = max(BLIST)
os.environ['LJQ_DECODE_G'] = '1'

# tags: three layer classes + per-module operators
TAGS = {'L.base', 'L.r4', 'L.r128',
        'attn.q_proj', 'attn.kv_proj', 'attn.o_proj', 'attn.compressor', 'attn.indexer',
        'attn.write_ckv', 'attn.all_rows', 'attn.sparse_core',
        'hc.pre_norm', 'hc.post', 'moe.fused_ffn', 'hc.head', 'head.logits', 'embed',
        'mtp.layer', 'mtp.head', 'mtp.embed', 'T.step'}


def bucket(name: str) -> str:
    n = name.lower()
    if 'nccl' in n:
        return 'A_nccl'
    if 'fp4' in n or 'moe' in n or 'fused_ffn' in n or 'grouped' in n:
        return 'B_moe_fp4'
    if 'gemm' in n or 'cutlass' in n or 'sgemm' in n or 'gemv' in n or 'skinny' in n or 's16816' in n or 'wgrad' in n:
        return 'C_dense_gemm'
    if 'index' in n:
        return 'E_indexer'
    if 'sparse' in n or 'paged' in n or 'flash' in n or 'attn' in n:
        return 'F_sparse_attn'
    if 'hc_fused' in n or 'rmsnorm' in n or 'norm' in n:
        return 'G_hc_norm'
    if n.startswith('void at::') or 'elementwise' in n or 'vectorized' in n or 'reduce_kernel' in n \
       or 'catarray' in n or 'copy' in n or 'fill' in n or 'index_elementwise' in n or 'unrolled' in n:
        return 'D_torch_tail'
    return 'H_other'


try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()
    ids = json.load(open('/tmp/c1_ids.json'))
    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args(max_batch_size=2 * BMAX, max_seq_len=int(os.environ.get('MAXSEQ', '1048576')))
    model = Transformer(args)
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to(dev)
    torch.set_default_device(dev)
    for m in model.modules():
        if hasattr(m, 'temperature'):
            m.temperature = 0.0
    dist.barrier()
    from ops import peer_ar_rows
    peer_ar_rows.prewarm(peer_ar_rows.nmax_for_pool(model.pool, 2 * BMAX * Q))
    log(f'model ready, BLIST={BLIST} N_PROF={N_PROF} peer_ar_rows={peer_ar_rows.ready()}')

    # ---- prefill BMAX rows with DIFFERENT lengths (real multi-sequence case) ----
    lens = [347, 336, 325, 314, 303, 292, 281, 270][:BMAX]
    ROOM = 4 * (N_PROF + 8) * Q + 128

    def prefill(slot, L):
        toks = torch.tensor(ids[:L], dtype=torch.long, device=dev).unsqueeze(0)
        model.pool.ensure(slot, L + ROOM)
        _, lg, _ = model(toks, start_pos=0, full_logits=True, slot=slot)
        return int(lg[0, -1].float().argmax())

    firsts = [prefill(b, lens[b]) for b in range(BMAX)]
    log(f'prefilled {BMAX} rows lens={lens}')

    # ---- record_function wrappers (layer-class + per-module operator view) ----
    from torch.profiler import record_function, profile, ProfilerActivity
    import ops.attn_decode_g as G, ops, model.arch as AR
    G.set_idx_rs(os.environ.get('IDX_RS','0')=='1'); G.set_idx_bf16(os.environ.get('IDX_BF16','0')=='1')
    if RANK==0: log(f'[AB] IDX_RS={G._IDX_RS} IDX_BF16={G._IDX_BF16}')

    def wrap(mod, name, tag):
        fn = getattr(mod, name)
        def w(*a, **k):
            with record_function(tag):
                return fn(*a, **k)
        setattr(mod, name, w)

    for n, t in [('_q', 'attn.q_proj'), ('_kv', 'attn.kv_proj'), ('_o', 'attn.o_proj'),
                 ('_compress_g', 'attn.compressor'), ('_indexer_g', 'attn.indexer'),
                 ('_write_ckv_g', 'attn.write_ckv'), ('_all_rows_g', 'attn.all_rows'),
                 ('sparse_attn_flat', 'attn.sparse_core')]:
        if hasattr(G, n):
            wrap(G, n, t)
    for r, t in [(0, 'L.base'), (4, 'L.r4'), (128, 'L.r128')]:
        fn = G.OPS_G[r]
        def mk(fn, t):
            def w(*a, **k):
                with record_function(t):
                    return fn(*a, **k)
            return w
        G.OPS_G[r] = mk(fn, t)
    wrap(ops, 'hc_pre_norm', 'hc.pre_norm')
    wrap(AR.Block, 'hc_post', 'hc.post')
    wrap(AR.Block, 'fused_ffn', 'moe.fused_ffn')
    wrap(AR.Block, 'hc_head', 'hc.head')
    wrap(AR.ParallelHead, 'forward', 'head.logits')
    wrap(AR.ParallelEmbedding, 'forward', 'embed')
    wrap(AR.DSparkBlock, 'forward_q', 'mtp.layer')
    wrap(AR.DSparkBlock, 'forward_head', 'mtp.head')
    wrap(AR.DSparkBlock, 'forward_embed', 'mtp.embed')
    wrap(AR.Transformer, 'step_g_batch', 'T.step')

    RESULT = {}
    for B in BLIST:
        slots = list(range(B))
        qin = torch.stack([torch.full((Q,), firsts[b], dtype=torch.long, device=dev) for b in range(B)])
        pos_t = torch.tensor(lens[:B], dtype=torch.long, device=dev)
        for b in range(B):
            model.pool.ensure(slots[b], lens[b] + ROOM)

        # ---------- eager warm + timing ----------
        for _ in range(2):
            model.step_g_batch(qin.clone(), pos_t.clone(), slots)
        torch.cuda.synchronize(); dist.barrier()
        ts = []
        for _ in range(N_PROF):
            q2, p2 = qin.clone(), pos_t.clone()
            torch.cuda.synchronize(); ta = time.time()
            model.step_g_batch(q2, p2, slots)
            torch.cuda.synchronize(); ts.append(time.time() - ta)
        eager_ms = sum(ts) / len(ts) * 1000

        # ---------- eager profile -> layer-class + module view ----------
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N_PROF):
                model.step_g_batch(qin.clone(), pos_t.clone(), slots)
            torch.cuda.synchronize()
        mod_view = {}
        if RANK == 0:
            for k in prof.key_averages():
                if k.key in TAGS:
                    mod_view[k.key] = [k.device_time_total / N_PROF / 1000.0, k.count // N_PROF]

        # ---------- capture graph, time replay ----------
        qg, pg = qin.clone(), pos_t.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.step_g_batch(qg, pg, slots)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.step_g_batch(qg, pg, slots)
        torch.cuda.synchronize(); dist.barrier()
        tg = []
        for _ in range(N_PROF + 3):
            torch.cuda.synchronize(); ta = time.time(); graph.replay()
            torch.cuda.synchronize(); tg.append(time.time() - ta)
        graph_ms = sum(tg[3:]) / len(tg[3:]) * 1000

        # ---------- graph replay profile -> operator buckets + per-layer wall ----------
        with profile(activities=[ProfilerActivity.CUDA]) as gp:
            for _ in range(N_PROF):
                graph.replay()
            torch.cuda.synchronize()
        op_view, buck, layers, tot_ms, nk = {}, {}, [], 0.0, 0
        # per-rank own compute (non-nccl kernel sum) + nccl sum + first->last kernel span, gathered to rank0
        _own = _ncc = 0.0; _ev = [e for e in gp.events() if e.device_type.name == 'CUDA']
        for e in _ev:
            (_ncc if 'nccl' in e.name else _own)
            if 'nccl' in e.name: _ncc += e.time_range.elapsed_us()
            else: _own += e.time_range.elapsed_us()
        _span = (max(e.time_range.end for e in _ev) - min(e.time_range.start for e in _ev)) if _ev else 0.0
        _t = torch.tensor([_own / N_PROF / 1000.0, _ncc / N_PROF / 1000.0, _span / N_PROF / 1000.0], device='cuda')
        _g = [torch.zeros_like(_t) for _ in range(dist.get_world_size())]; dist.all_gather(_g, _t)
        if RANK == 0:
            for r_, v in enumerate(_g):
                log(f'    rank{r_}: own_kernels {v[0].item():6.2f}ms  nccl {v[1].item():6.2f}ms  span/replay {v[2].item():6.2f}ms')
        if RANK == 0:
            agg = defaultdict(lambda: [0.0, 0]); tot = 0.0
            for e in gp.events():
                if e.device_type.name == 'CUDA':
                    us = e.time_range.elapsed_us()
                    agg[e.name][0] += us; agg[e.name][1] += 1; tot += us
            tot_ms = tot / N_PROF / 1000.0
            nk = sum(v[1] for v in agg.values()) // N_PROF
            bk = defaultdict(lambda: [0.0, 0])
            for name, (us, n) in agg.items():
                b_ = bucket(name)
                bk[b_][0] += us / N_PROF / 1000.0; bk[b_][1] += n // N_PROF
            buck = {k: v for k, v in sorted(bk.items(), key=lambda kv: -kv[1][0])}
            op_view = {name: [us / N_PROF / 1000.0, n // N_PROF]
                       for name, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:int(os.environ.get('TOPK_OPS','30'))]}
            # per-layer wall from last replay, split by hc_fused_pre launches (2 per layer)
            ks = sorted([e for e in gp.events() if e.device_type.name == 'CUDA'],
                        key=lambda e: e.time_range.start)
            per = len(ks) // N_PROF
            if per > 0:
                last = ks[-per:]
                pre = [k for k in last if 'hc_fused_pre' in k.name]
                bnd = [last[0].time_range.start] + [k.time_range.start for k in pre] + [last[-1].time_range.end]
                def seg(i):
                    L = [k for k in last if bnd[i] <= k.time_range.start < bnd[i + 1]]
                    nc = sum(k.time_range.elapsed_us() for k in L if 'nccl' in k.name)
                    return (bnd[i + 1] - bnd[i], nc, len(L))
                S = [seg(i) for i in range(len(bnd) - 1)]
                for li, i in enumerate(range(1, len(S) - 1, 2)):
                    a, c = S[i], S[i + 1]
                    layers.append([li, (a[0] + c[0]) / 1000.0, (a[1] + c[1]) / 1000.0, a[2] + c[2]])

        if RANK == 0:
            RESULT[B] = dict(eager_ms=eager_ms, graph_ms=graph_ms, kernel_ms=tot_ms, kernels=nk,
                             modules=mod_view, buckets=buck, top_kernels=op_view, layers=layers)
            log(f'B={B}: eager {eager_ms:.2f}ms  graph {graph_ms:.2f}ms  kernel_sum {tot_ms:.2f}ms  kernels/step {nk}')
            for k, v in list(buck.items()):
                log(f'    {k:14s} {v[0]:7.2f}ms  {v[1]:5d} kernels')

        del graph
        torch.cuda.synchronize(); torch.cuda.empty_cache(); dist.barrier()

    if RANK == 0:
        json.dump(RESULT, open('/tmp/amdahl_b.json', 'w'), indent=1)
        log('saved /tmp/amdahl_b.json')
        # ---- comparison tables ----
        log('')
        log('=== LAYER-CLASS / MODULE view: CUDA ms per step (eager, nested tags overlap) ===')
        keys = sorted({k for B in RESULT for k in RESULT[B]['modules']},
                      key=lambda k: -RESULT[BLIST[0]]['modules'].get(k, [0, 0])[0])
        hdr = 'tag'.ljust(18) + ''.join([f'B{B}_ms'.rjust(9) + f'B{B}_n'.rjust(7) for B in BLIST])
        log(hdr + '   B4/B1')
        for k in keys:
            row = k.ljust(18)
            for B in BLIST:
                ms, n = RESULT[B]['modules'].get(k, [0.0, 0])
                row += f'{ms:9.2f}{n:7d}'
            a = RESULT[BLIST[0]]['modules'].get(k, [0.0, 0])[0]
            b = RESULT[BLIST[-1]]['modules'].get(k, [0.0, 0])[0]
            row += f'   {(b/a if a > 0 else 0):5.2f}x'
            log(row)
        log('')
        log('=== OPERATOR bucket view: CUDA ms per step (graph replay, disjoint) ===')
        bkeys = sorted({k for B in RESULT for k in RESULT[B]['buckets']})
        hdr = 'bucket'.ljust(16) + ''.join([f'B{B}_ms'.rjust(9) + f'B{B}_n'.rjust(7) for B in BLIST])
        log(hdr + '   B4/B1   dAbs')
        for k in bkeys:
            row = k.ljust(16)
            for B in BLIST:
                ms, n = RESULT[B]['buckets'].get(k, [0.0, 0])
                row += f'{ms:9.2f}{n:7d}'
            a = RESULT[BLIST[0]]['buckets'].get(k, [0.0, 0])[0]
            b = RESULT[BLIST[-1]]['buckets'].get(k, [0.0, 0])[0]
            row += f'   {(b/a if a > 0 else 0):5.2f}x {b-a:7.2f}'
            log(row)
        log('')
        log('=== step time scaling ===')
        for B in BLIST:
            r = RESULT[B]
            log(f'B={B}: graph {r["graph_ms"]:7.2f}ms  kernels {r["kernels"]:5d}  '
                f'per-row {r["graph_ms"]/B:6.2f}ms  speedup_vs_ideal {RESULT[BLIST[0]]["graph_ms"]*B/r["graph_ms"]:.2f}x')
    dist.barrier()
except Exception:
    if RANK == 0:
        traceback.print_exc()
    sys.stdout.flush()
finally:
    try:
        dist.destroy_process_group()
    except Exception:
        pass
