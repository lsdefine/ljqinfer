# -*- coding: utf-8 -*-
"""prefill-vs-decode equivalence probe (torchrun --nproc_per_node=8).
Sequence S = prompt + rep0_out[:K] (K=1799: prefill-path says EOS, decode-path said </think>).
A: full prefill of S (chunked, 128-aligned) -> last-token argmax + KV snapshot.
B: prefill S[:-m] then m Q-decode steps (teacher forced with real tokens) -> argmax + KV.
Compare per-layer KV/ckv/ickv rows at positions [N-m, N) and the final logits.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, time, traceback
import torch
import torch.distributed as dist

t0 = time.time()
RANK = int(os.environ['RANK'])
def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

K = int(os.environ.get('K', '1799'))
MS = [int(x) for x in os.environ.get('MS', '8,64').split(',')]
CHUNK = 1024

try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()

    prompt = json.load(open('/tmp/g5k/prompt_ids.json'))
    out = json.load(open('/tmp/g5k/rep0_out.json'))
    base = list(prompt) + list(out[:K])
    NTOT = int(os.environ.get('NTOT', '17408'))
    S = (base * ((NTOT + len(base) - 1) // len(base)))[:NTOT]
    N = len(S)
    log(f'seq N={N} (prompt {len(prompt)} + out {K})')

    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    dev = f'cuda:{RANK}'
    for m_ in model.modules():
        for k_, v_ in list(m_._buffers.items()):
            if v_ is not None and v_.device.type == 'cpu':
                m_._buffers[k_] = v_.to(dev)
    torch.set_default_device(dev)
    dist.barrier()
    log('model ready')

    
    SPEC_Q = 8
    DRAFT = os.environ.get('DRAFT', 'mtp')     # mtp: real MTP drafts, only row0 teacher-forced | true: all rows true tokens
    GRAPH = int(os.environ.get('GRAPH', '0'))

    def prefill(seq, slot, draft=True):
        pos = 0; lg = None; mh = None
        while pos < len(seq):
            n = min(CHUNK, len(seq) - pos)
            if pos + n < len(seq):
                n = (n // 128) * 128 or n
            toks = torch.tensor([seq[pos:pos + n]], device=dev)
            with torch.no_grad():
                oid, lg, mh = model(toks, start_pos=pos, full_logits=False, slot=slot)
                if draft:
                    model.write_draft_kv(mh, pos, slot)
            pos += n
        lg = (lg[0] if lg.dim() == 3 else lg)[-1].float()
        return lg, mh[:, -1:]

    def snap(slot, t0_, t1_):
        d = {}
        for lid, p in model.pool.layers.items():
            d[f'kv{lid}'] = p.kv(slot, t0_, t1_).float().clone()
            if hasattr(p, 'ckv'): d[f'ckv{lid}'] = p.ckv(slot, t1_).float().clone()
            if hasattr(p, 'ickv'): d[f'ickv{lid}'] = p.ickv(slot, t1_).float().clone()
        return d

    def kvreport(snapA, snapB, P):
        for lid in sorted(model.pool.layers.keys()):
            a, b = snapA[f'kv{lid}'], snapB[f'kv{lid}']
            d = (a - b).abs().amax(dim=tuple(range(1, a.dim())))
            nzp = torch.nonzero(d > 0).flatten()
            s_ = f'L{lid:2d} kv first_diff={int(nzp[0]) if nzp.numel() else -1} relA={float(a.abs().max()):.3g} tail_max={d[P:].max().item():.3g} tail_mean={d[P:].mean().item():.3g} prefix_max={d[:P].max().item():.3g}'
            for nm in ('ckv', 'ickv'):
                k_ = f'{nm}{lid}'
                if k_ in snapA:
                    a, b = snapA[k_], snapB[k_]
                    if a.shape != b.shape: s_ += f' {nm} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}'; continue
                    d2 = (a - b).abs().amax(dim=tuple(range(1, a.dim()))) if a.dim() > 1 else (a - b).abs()
                    nz2 = torch.nonzero(d2 > 0).flatten()
                    s_ += f' {nm} n={a.shape[0]} first_diff={int(nz2[0]) if nz2.numel() else -1} max={d2.max().item():.3g} mean={d2.mean().item():.3g}'
            log('   ' + s_)

    last_lg = {}
    _orig_fqg = model.forward_q_g
    def _hook_fqg(qin, pos_t, slot):
        lg, mh = _orig_fqg(qin, pos_t, slot)
        last_lg['lg'] = lg
        return lg, mh
    model.forward_q_g = _hook_fqg




    # ---- long-context probe: cross RING_TOKENS=16384 wrap (main_kv row = t % RING)
    P = int(os.environ.get('P', '15872'))       # 128-aligned, < 16384
    assert P % 128 == 0 and P < N
    from model.past import RING_TOKENS
    log(f'N={N} P={P} RING={RING_TOKENS} base_len={len(base)}')

    # A: full prefill
    slotA = model.pool.alloc()
    lgA, _ = prefill(S, slotA); torch.cuda.synchronize()
    W0 = max(N - RING_TOKENS, 0)
    snapA = snap(slotA, W0, N)
    log(f'A full-prefill argmax={int(lgA.argmax())} top5={lgA.topk(5).indices.tolist()}')
    # A2: noise floor
    slotA2 = model.pool.alloc()
    lgA2, _ = prefill(S, slotA2); torch.cuda.synchronize()
    log(f"A' argmax={int(lgA2.argmax())} maxdiff_vs_A={(lgA-lgA2).abs().max().item():.4f} (NOISE FLOOR)")
    kvreport(snapA, snap(slotA2, W0, N), P - W0)
    model.pool.release(slotA2)

    # B: prefill S[:P], then Q8 teacher-forced windows (forward_q_g) to N
    slotB = model.pool.alloc()
    prefill(S[:P], slotB); torch.cuda.synchronize()
    model.pool.ensure(slotB, N + 16)
    pos = P; lgB = None
    with torch.no_grad():
        while pos < N:
            q = torch.tensor([S[pos:pos + SPEC_Q]], dtype=torch.int64, device=dev)
            nv = q.shape[1]
            if nv < SPEC_Q:
                q = torch.cat([q, torch.zeros(1, SPEC_Q - nv, dtype=torch.int64, device=dev)], 1)
            lg, _ = model.forward_q_g(q, torch.tensor([pos], device=dev), slotB)
            lgB = lg[0, nv - 1].float()
            pos += nv
    torch.cuda.synchronize()
    log(f'B Q8-decode({P}->{N}) argmax={int(lgB.argmax())} same_as_A={int(lgB.argmax())==int(lgA.argmax())} top5={lgB.topk(5).indices.tolist()} maxdiff_vs_A={(lgA-lgB).abs().max().item():.4f}')
    kvreport(snapA, snap(slotB, W0, N), P - W0)
    model.pool.release(slotB)

    # C: prefill S[:P], then Q=1 forward_q to N
    if int(os.environ.get('DO_C', '1')):
        slotC = model.pool.alloc()
        prefill(S[:P], slotC); torch.cuda.synchronize()
        model.pool.ensure(slotC, N + 16)
        with torch.no_grad():
            for j in range(P, N):
                lgC, _ = model.forward_q(torch.tensor([[S[j]]], device=dev), start_pos=j, slot=slotC)
        lgC = lgC[0, -1].float(); torch.cuda.synchronize()
        log(f'C Q1-decode({P}->{N}) argmax={int(lgC.argmax())} same_as_A={int(lgC.argmax())==int(lgA.argmax())} top5={lgC.topk(5).indices.tolist()} maxdiff_vs_A={(lgA-lgC).abs().max().item():.4f}')
        kvreport(snapA, snap(slotC, W0, N), P - W0)
        model.pool.release(slotC)

    dist.barrier(); dist.destroy_process_group()
    log('LONG_PROBE_DONE')
except Exception:
    traceback.print_exc()
    log('LONG_PROBE_FAILED')
