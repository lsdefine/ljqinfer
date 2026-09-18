"""B1Q8 decode profile: eager step_g under torch.profiler (rank0), plus graph replay step time. No gate."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, sys, json, time, torch, torch.distributed as dist
from collections import defaultdict
RANK = int(os.environ.get('RANK', '0'))
def log(*a):
    if RANK == 0: print(*a, flush=True)
N_PROF = int(os.environ.get('N_PROF', '4')); Q = 8
TAGS = {'attn.q_proj','attn.kv_proj','attn.o_proj','attn.compressor','attn.indexer','attn.write_ckv','attn.all_rows','attn.sparse_core','L.base','L.r4','L.r128','hc.pre_norm','hc.post','moe.fused_ffn','hc.head','head.logits','embed','mtp.layer','mtp.head','mtp.embed','T.target','T.step'}
os.environ['LJQ_DECODE_G'] = '1'
try:
    dist.init_process_group('nccl'); torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()
    ids = json.load(open('/tmp/c1_ids.json'))
    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args(); model = Transformer(args); bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]; assert not bad, bad
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu': m._buffers[k] = v.to(dev)
    torch.set_default_device(dev); dist.barrier()
    P = len(ids); toks = torch.tensor([ids], device=dev)
    slot = 0
    model.pool.ensure(slot, P + 64)
    out, _, mh = model(toks, start_pos=0, slot=slot)
    first = out.view(-1)[-1:].clone()
    model.forward_spec(first, mh, 0, slot=slot)
    d0, _, _ = model.forward_spec(first, mh[:, -1:], P - 1, slot=slot)
    qin = torch.zeros(1, Q, dtype=torch.int64, device=dev)
    qin[0, :d0.size(1)] = d0[0]; qin[0, d0.size(1):] = d0[0, -1]
    pos_t = torch.full((1,), P, dtype=torch.int64, device=dev)
    # eager warm
    from ops import peer_ar
    peer_ar.prewarm()
    import collections as _col
    _AR_HIST = _col.Counter()
    _ar_orig = dist.all_reduce
    def _ar_probe(t, *a, **k):
        _AR_HIST[(tuple(t.shape), str(t.dtype))] += 1
        return _ar_orig(t, *a, **k)
    dist.all_reduce = _ar_probe
    for _ in range(2): model.step_g(qin, pos_t, slot)
    torch.cuda.synchronize(); dist.barrier()
    # eager timed
    ts = []
    for _ in range(N_PROF):
        torch.cuda.synchronize(); ta = time.time(); model.step_g(qin, pos_t, slot); torch.cuda.synchronize(); ts.append(time.time() - ta)
    log(f'eager step {sum(ts)/len(ts)*1000:.1f}ms x{len(ts)}')
    log('[AR-HIST] fallback all_reduce calls per eager step:')
    for _k, _v in sorted(_AR_HIST.items(), key=lambda kv: -kv[1])[:8]:
        log('   ', _k, _v // max(1, 2 + N_PROF))
    log('[AR-HIST] peer fast path:', peer_ar.ready())
    if os.environ.get('AB') == '1':
        import ops.attn_decode_g as _ad
        _res = {}
        for _tag in ('base', 'rs'):
            _ad.set_idx_bf16(_tag == 'rs')
            _s = torch.cuda.Stream(); _s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_s):
                for _ in range(2): model.step_g(qin, pos_t, slot)
            torch.cuda.current_stream().wait_stream(_s); torch.cuda.synchronize(); dist.barrier()
            _g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_g): model.step_g(qin, pos_t, slot)
            torch.cuda.synchronize(); dist.barrier()
            _t = []
            for _ in range(12):
                torch.cuda.synchronize(); _a = time.time(); _g.replay(); torch.cuda.synchronize(); _t.append(time.time() - _a)
            _t = sorted(_t[2:])
            _res[_tag] = (sum(_t) / len(_t) * 1000, _t[0] * 1000)
            log('[AB] %s: mean %.2fms min %.2fms' % (_tag, _res[_tag][0], _res[_tag][1]))
            log('[AB] %s hits rs=%d ar=%d qin=%s' % (_tag, _ad._IDX_RS_HITS[0], _ad._IDX_RS_HITS[1], tuple(qin.shape)))
            _ad._IDX_RS_HITS[0] = 0; _ad._IDX_RS_HITS[1] = 0
            del _g; torch.cuda.synchronize(); torch.cuda.empty_cache(); dist.barrier()
        log('[AB] delta mean %.2fms min %.2fms (neg = peer faster)' % (_res['rs'][0] - _res['base'][0], _res['rs'][1] - _res['base'][1]))
        sys.stdout.flush(); os._exit(0)
    # profile
    # ---- module-level record_function wrappers (profile only) ----
    from torch.profiler import record_function
    import ops.attn_decode_g as G, ops, model.arch as AR
    def wrap(mod, name, tag):
        fn = getattr(mod, name)
        def w(*a, **k):
            with record_function(tag): return fn(*a, **k)
        setattr(mod, name, w)
    for n, t in [('_q', 'attn.q_proj'), ('_kv', 'attn.kv_proj'), ('_o', 'attn.o_proj'), ('_compress_g', 'attn.compressor'),
                 ('_indexer_g', 'attn.indexer'), ('_write_ckv_g', 'attn.write_ckv'), ('_all_rows_g', 'attn.all_rows'),
                 ('sparse_attn_flat', 'attn.sparse_core')]:
        wrap(G, n, t)
    for r, t in [(0, 'L.base'), (4, 'L.r4'), (128, 'L.r128')]:
        fn = G.OPS_G[r]
        def mk(fn, t):
            def w(*a, **k):
                with record_function(t): return fn(*a, **k)
            return w
        G.OPS_G[r] = mk(fn, t)
    wrap(ops, 'hc_pre_norm', 'hc.pre_norm'); wrap(AR.Block, 'hc_post', 'hc.post'); wrap(AR.Block, 'fused_ffn', 'moe.fused_ffn')
    wrap(AR.Block, 'hc_head', 'hc.head'); wrap(AR.ParallelHead, 'forward', 'head.logits'); wrap(AR.ParallelEmbedding, 'forward', 'embed')
    wrap(AR.DSparkBlock, 'forward_q', 'mtp.layer'); wrap(AR.DSparkBlock, 'forward_head', 'mtp.head'); wrap(AR.DSparkBlock, 'forward_embed', 'mtp.embed')
    wrap(AR.Transformer, 'forward_q_g', 'T.target'); wrap(AR.Transformer, 'step_g', 'T.step')
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(N_PROF): model.step_g(qin, pos_t, slot)
        torch.cuda.synchronize()
    if RANK == 0:
        agg = defaultdict(lambda: [0.0, 0]); tot = 0.0
        for e in prof.events():
            if e.device_type.name == 'CUDA':
                agg[e.name][0] += e.time_range.elapsed_us(); agg[e.name][1] += 1; tot += e.time_range.elapsed_us()
        log(f'CUDA kernel time per step: {tot/N_PROF/1000:.1f}ms  (kernels/step {sum(v[1] for v in agg.values())//N_PROF})')
        for name, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:35]:
            log(f'{us/N_PROF/1000:8.2f}ms {n//N_PROF:5d}x  {name[:110]}')
        # per-module: key_averages() attributes child kernel device time to user annotations (kineto correlation)
        log('--- per-module CUDA time / kernel launches per step (nested tags overlap) ---')
        ka = prof.key_averages()
        rows = [(k.key, k.device_time_total, k.count) for k in ka if k.key in TAGS]
        for nm, us, n in sorted(rows, key=lambda r: -r[1]):
            log(f'{us/N_PROF/1000:8.2f}ms {n//N_PROF:5d} calls  {nm}')
        prof.export_chrome_trace('/tmp/prof_decode_rank0.json')
    dist.barrier()
    if RANK == 0:
        log('--- fp32 matmul callers (eager, grouped by shape):')
        for ev in prof.key_averages(group_by_input_shape=True):
            if ev.key in ('aten::mm','aten::addmm','aten::bmm','aten::matmul','aten::linear') and ev.input_shapes:
                log(f'{ev.key:14s} n={ev.count:4d} cuda={ev.device_time_total/1000:.2f}ms shapes={ev.input_shapes}')
    # graph replay timing
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): model.step_g(qin, pos_t, slot)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): model.step_g(qin, pos_t, slot)
    torch.cuda.synchronize(); dist.barrier()
    tg = []
    for _ in range(N_PROF + 2):
        torch.cuda.synchronize(); ta = time.time(); graph.replay(); torch.cuda.synchronize(); tg.append(time.time() - ta)
    log(f'graph step {sum(tg[2:])/len(tg[2:])*1000:.1f}ms x{len(tg)-2}')
    # kernel-level breakdown of graph replay (CUPTI still sees kernels inside a graph)
    with profile(activities=[ProfilerActivity.CUDA]) as gp:
        for _ in range(N_PROF): graph.replay()
        torch.cuda.synchronize()
    if RANK == 0:
        agg = defaultdict(lambda: [0.0, 0]); tot = 0.0
        for e in gp.events():
            if e.device_type.name == 'CUDA':
                agg[e.name][0] += e.time_range.elapsed_us(); agg[e.name][1] += 1; tot += e.time_range.elapsed_us()
        log(f'--- GRAPH replay kernel time per step: {tot/N_PROF/1000:.1f}ms (kernels/step {sum(v[1] for v in agg.values())//N_PROF})')
        for name, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:45]:
            log(f'{us/N_PROF/1000:8.2f}ms {n//N_PROF:5d}x  {name[:120]}')
        gp.export_chrome_trace('/tmp/prof_graph_rank0.json')
        # per-layer wall from graph trace: split last replay by hc_fused_pre launches (2 per layer: attn-pre, ffn-pre)
        ks = sorted([e for e in gp.events() if e.device_type.name == 'CUDA'], key=lambda e: e.time_range.start)
        per = len(ks) // N_PROF; last = ks[-per:]
        pre = [k for k in last if 'hc_fused_pre' in k.name]
        b = [last[0].time_range.start] + [k.time_range.start for k in pre] + [last[-1].time_range.end]
        def seg(i):
            L = [k for k in last if b[i] <= k.time_range.start < b[i + 1]]
            nc = sum(k.time_range.elapsed_us() for k in L if 'nccl' in k.name)
            cp = sum(k.time_range.elapsed_us() for k in L if 'nccl' not in k.name)
            return (b[i + 1] - b[i], nc, cp, len(L))
        S = [seg(i) for i in range(len(b) - 1)]
        log(f'--- GRAPH last replay: wall {(b[-1]-b[0])/1000:.2f}ms kernels {per} hc_pre {len(pre)} (head seg {S[0][0]/1000:.2f}ms, tail seg {S[-1][0]/1000:.2f}ms)')
        log('layer: wall_ms nccl_ms compute_ms nk  (layer = attn-pre seg + ffn-pre seg)')
        for li, i in enumerate(range(1, len(S) - 1, 2)):
            a, c = S[i], S[i + 1]
            log(f'L{li:02d}: {(a[0]+c[0])/1000:6.3f} {(a[1]+c[1])/1000:6.3f} {(a[2]+c[2])/1000:6.3f} {a[3]+c[3]:4d}')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
