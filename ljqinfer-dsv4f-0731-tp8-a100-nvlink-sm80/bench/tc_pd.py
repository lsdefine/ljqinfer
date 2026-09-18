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

    def prefill(seq, slot):
        pos = 0
        while pos < len(seq):
            n = min(CHUNK, len(seq) - pos)
            if pos + n < len(seq):
                n = (n // 128) * 128 or n
            toks = torch.tensor([seq[pos:pos + n]], device=dev)
            oid, lg, mh = model(toks, start_pos=pos, full_logits=True, slot=slot)
            pos += n
        return lg

    def snap(slot, t0_, t1_):
        d = {}
        for lid, p in model.pool.layers.items():
            d[f'kv{lid}'] = p.kv(slot, t0_, t1_).float().clone()
            if hasattr(p, 'ckv'):
                d[f'ckv{lid}'] = p.ckv(slot, t1_).float().clone()
            if hasattr(p, 'ickv'):
                d[f'ickv{lid}'] = p.ickv(slot, t1_).float().clone()
        return d

    # ---- A: full prefill
    slotA = model.pool.alloc()
    lgA = prefill(S, slotA)
    lgA = (lgA[0] if lgA.dim() == 3 else lgA)[-1].float()
    topA = int(lgA.argmax())
    snapA = snap(slotA, 0, N)
    log(f'A full-prefill argmax={topA} top5={lgA.topk(5).indices.tolist()}')

    def report(tag, snapB, m, lgB):
        lgB = lgB.float()
        topB = int(lgB.argmax())
        log(f'{tag} m={m} argmax={topB} top5={lgB.topk(5).indices.tolist()} same_as_A={topB==topA} logit_maxdiff={(lgA-lgB).abs().max().item():.4f}')
        lids = sorted(model.pool.layers.keys())
        rows = []
        for lid in lids:
            a, b = snapA[f'kv{lid}'], snapB[f'kv{lid}']
            d = (a - b).abs().amax(dim=tuple(range(1, a.dim())))
            nzp = torch.nonzero(d > 0).flatten()
            first = int(nzp[0]) if nzp.numel() else -1
            kvmax = d[N - m:].max().item() if m else 0.0
            kvmax_pre = d[:N - m].max().item() if N - m > 0 else 0.0
            s_ = f'L{lid:2d} kv first_diff_pos={first} (N-m={N-m}) relA={float(a.abs().max()):.3g} maxdiff_tail={kvmax:.3g} maxdiff_prefix={kvmax_pre:.3g}'
            for nm in ('ckv', 'ickv'):
                k_ = f'{nm}{lid}'
                if k_ in snapA:
                    a, b = snapA[k_], snapB[k_]
                    if a.shape != b.shape:
                        s_ += f' {nm} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}'; continue
                    d2 = (a - b).abs().amax(dim=tuple(range(1, a.dim()))) if a.dim() > 1 else (a - b).abs()
                    nz2 = torch.nonzero(d2 > 0).flatten()
                    s_ += f' {nm} n={a.shape[0]} first_diff={int(nz2[0]) if nz2.numel() else -1} max={d2.max().item():.3g}'
            rows.append(s_)
        for r_ in rows: log('   ' + r_)

    # --- A' : identical full prefill into another slot -> noise floor
    slotA2 = model.pool.alloc(); model._set_pos(slotA2, 0)
    lgA2 = prefill(S, slotA2)[0, -1].float()
    log(f"A' full-prefill argmax={int(lgA2.argmax())} logit_maxdiff_vs_A={(lgA-lgA2).abs().max().item():.4f}  (NOISE FLOOR)")
    report("A'(noise)", snap(slotA2, 0, N), 0, lgA2)
    model.pool.release(slotA2)
    for m in MS:
        # B: decode path (forward_q Q=1, teacher forced)
        slotB = model.pool.alloc()
        prefill(S[:N - m], slotB)
        lgB = None
        for j in range(N - m, N):
            toks = torch.tensor([[S[j]]], device=dev)
            lgB, mh = model.forward_q(toks, start_pos=j, slot=slotB)
        report('B(decode)', snap(slotB, 0, N), m, lgB[0, -1])
        model.pool.release(slotB)
        # C: control -- second prefill of the same m tokens (S>1 path); needs 128-aligned start
        if (N - m) % 128: continue
        slotC = model.pool.alloc()
        prefill(S[:N - m], slotC)
        toks = torch.tensor([S[N - m:N]], device=dev)
        oid, lgC, mh = model(toks, start_pos=N - m, full_logits=True, slot=slotC)
        lgC = (lgC[0] if lgC.dim() == 3 else lgC)[-1]
        report('C(prefill-m)', snap(slotC, 0, N), m, lgC)
        model.pool.release(slotC)

    dist.barrier(); dist.destroy_process_group()
    log('PD_PROBE_DONE')
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] PD_PROBE_FAILED', flush=True)
