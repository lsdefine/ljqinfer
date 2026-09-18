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
    S = list(prompt) + list(out[:K])
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



    # ---- cross-step contamination probe
    # window1 at P (rows>=k random == rejected drafts), then window2 at P+k with true rows, vs all-true.
    import random
    random.seed(0)
    P0 = int(os.environ.get('P0', '6656'))
    KS = [int(x) for x in os.environ.get('KS', '1,3,4,5,7').split(',')]
    DEPTH = int(os.environ.get('DEPTH', '3'))   # how many true windows after the dirty one
    def run(P, k, dirty):
        slot = model.pool.alloc()
        prefill(S[:P], slot); torch.cuda.synchronize()
        model.pool.ensure(slot, P + 8 * SPEC_Q + 8)
        out = []
        pos = P
        q = torch.tensor([S[pos:pos + SPEC_Q]], device=dev)
        if dirty:
            for r in range(k, SPEC_Q): q[0, r] = random.randrange(1000, 100000)
        with torch.no_grad(): model.forward_q_g(q, torch.tensor([pos], device=dev), slot)
        pos += k
        for d in range(DEPTH):
            if pos + SPEC_Q > N: break
            q = torch.tensor([S[pos:pos + SPEC_Q]], dtype=torch.int64, device=dev)
            with torch.no_grad(): lg, _ = model.forward_q_g(q, torch.tensor([pos], device=dev), slot)
            out.append(lg[0].float().clone())
            pos += SPEC_Q
        torch.cuda.synchronize()
        st = snap(slot, 0, pos)
        model.pool.release(slot)
        return out, st, pos
    for k in KS:
        random.seed(k)
        oc, sc, pos = run(P0, k, False)
        random.seed(k)
        od, sd, _ = run(P0, k, True)
        rd = [[round((a[r] - b[r]).abs().max().item(), 2) for r in range(SPEC_Q)] for a, b in zip(oc, od)]
        am = [[int(a[r].argmax() == b[r].argmax()) for r in range(SPEC_Q)] for a, b in zip(oc, od)]
        log(f'P0={P0} k={k} (dirty rows {P0+k}..{P0+7}) window2+ rowdiff={rd} argmax_same={am}')
        worst = []
        for lid in sorted(model.pool.layers.keys()):
            for nm in ('kv', 'ckv', 'ickv'):
                key_ = f'{nm}{lid}'
                if key_ not in sc: continue
                a, b = sc[key_], sd[key_]
                if a.shape != b.shape: worst.append(f'L{lid}{nm}SHAPE{tuple(a.shape)}/{tuple(b.shape)}'); continue
                d = (a - b).abs().amax(dim=tuple(range(1, a.dim()))) if a.dim() > 1 else (a - b).abs()
                nz = torch.nonzero(d > 0).flatten()
                if nz.numel(): worst.append(f'L{lid}{nm}:first={int(nz[0])},n={int(nz.numel())},max={d.max().item():.3g}')
        log(f'P0={P0} k={k} state diffs (rows<{pos}): ' + (' '.join(worst[:24]) if worst else 'NONE'))

    dist.barrier(); dist.destroy_process_group()
    log('LEAK2_DONE')
except Exception:
    traceback.print_exc()
    log('LEAK2_FAILED')
