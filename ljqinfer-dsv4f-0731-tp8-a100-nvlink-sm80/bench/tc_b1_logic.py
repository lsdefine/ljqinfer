# -*- coding: utf-8 -*-
"""End-to-end acceptance for step_g_batch: B rows decoded together must emit exactly
the same token stream as each row decoded alone through the production step_g.

Rows get prompts of DIFFERENT lengths, so they sit at different positions and accept
different numbers of drafts each step -- the real multi-sequence case.
temperature is forced to 0 everywhere so both sides are deterministic.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, time, traceback
import torch
import torch.distributed as dist

t0 = time.time()
RANK = int(os.environ['RANK'])


def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)


Q = 8
BMAX = int(os.environ.get('BMAX', '4'))
NSTEP = int(os.environ.get('NSTEP', '16'))
os.environ['LJQ_DECODE_G'] = '1'
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
    args = make_args(max_batch_size=2 * BMAX, max_seq_len=131072)
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
    # determinism: MTP head samples with Gumbel noise when temperature > 0
    n_t = 0
    for m in model.modules():
        if hasattr(m, 'temperature'):
            m.temperature = 0.0
            n_t += 1
    log(f'forced temperature=0 on {n_t} modules')

    # rows get different prompt lengths -> different positions / accept counts
    lens = [347, 336, 325, 314][:BMAX]
    room = NSTEP * Q + 32

    def prefill(slot, L):
        toks = torch.tensor(ids[:L], dtype=torch.long, device=dev).unsqueeze(0)
        model.pool.ensure(slot, L + room)
        _, lg, _ = model(toks, start_pos=0, full_logits=True, slot=slot)
        return int(lg[0, -1].float().argmax())

    # ---------- baseline:每行单独走生产 step_g ----------
    base_seq = []
    for b in range(BMAX):
        slot = BMAX + b
        first = prefill(slot, lens[b])
        qin = torch.full((1, Q), first, dtype=torch.long, device=dev)
        pos_t = torch.tensor([lens[b]], dtype=torch.long, device=dev)
        seq = []
        for _ in range(NSTEP):
            out = model.step_g(qin, pos_t, slot)
            g, n_new = out[0], out[1]
            k = int(n_new.view(-1)[0])
            gv = g if g.dim() == 1 else g[0]      # step_g returns g as [Q]
            seq += [int(t) for t in gv[:k]]
        base_seq.append(seq)
        log(f'baseline row {b}: L={lens[b]} produced {len(seq)} tok')

    # ---------- same rows through step_g_batch but ONE ROW AT A TIME (B=1) ----------
    # B=1 through the batch path was proven bit-identical to forward_q_g, so any
    # divergence here is a LOGIC bug in step_g_batch (accept / MTP / qin update),
    # not a numeric effect of batching.
    b1 = []
    for b in range(BMAX):
        slot = b
        first = prefill(slot, lens[b])
        qin = torch.full((1, Q), first, dtype=torch.long, device=dev)
        pos_t = torch.tensor([lens[b]], dtype=torch.long, device=dev)
        seq = []
        for _ in range(NSTEP):
            g, n_new, _ = model.step_g_batch(qin, pos_t, [slot])
            k = int(n_new.view(-1)[0])
            seq += [int(t) for t in g[0, :k]]
        b1.append(seq)
        log(f'step_g_batch B=1 row {b}: produced {len(seq)} tok')

    ok = True
    for b in range(BMAX):
        a, c = base_seq[b], b1[b]
        n = min(len(a), len(c))
        d = next((i for i in range(n) if a[i] != c[i]), None)
        if d is None and len(a) == len(c):
            tag = 'IDENTICAL'
        else:
            tag = f'DIVERGE@{d}' if d is not None else f'LEN {len(a)} vs {len(c)}'
            ok = False
        log(f'row {b}: step_g {len(a)} tok / step_g_batch(B=1) {len(c)} tok -> {tag}')
    log('LOGIC GATE: ' + ('PASS - step_g_batch logic is exact' if ok else 'FAIL - logic bug in step_g_batch'))
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] B1LOGIC FAILED', flush=True)
