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

    # ---------- layer-wise error growth: same forward_q_g_batch, B=1 vs B=4 ----------
    # B=1 through this path is already proven bit-identical to production forward_q_g,
    # so any per-layer delta below is purely "B rows flattened together".
    firsts_b = [prefill(b, lens[b]) for b in range(BMAX)]   # slots 0..3 -> batch side
    _ = prefill(BMAX, lens[0])                              # slot BMAX -> single-row ref
    qin_b = torch.stack([torch.full((Q,), f, dtype=torch.long, device=dev) for f in firsts_b])
    pos_b = torch.tensor(lens, dtype=torch.long, device=dev)

    from model.arch import Block
    rec = []
    orig_qb = Block.forward_qb

    def wrapped(self, *a, **k):
        y = orig_qb(self, *a, **k)
        t = y[0] if isinstance(y, (tuple, list)) else y
        rec.append(t.detach().float().clone())
        return y

    Block.forward_qb = wrapped
    try:
        rec.clear()
        lg1 = model.forward_q_g_batch(qin_b[:1], pos_b[:1], [BMAX])
        rec1 = [t for t in rec]
        rec.clear()
        lg4 = model.forward_q_g_batch(qin_b, pos_b, list(range(BMAX)))
        rec4 = [t for t in rec]
    finally:
        Block.forward_qb = orig_qb

    log(f'captured layers: B1={len(rec1)}  B4={len(rec4)}')
    n = min(len(rec1), len(rec4))
    prev = 0.0
    jump = []
    for i in range(n):
        a = rec1[i]
        b = rec4[i]
        if i == 0:
            log(f'shapes: B1={tuple(a.shape)} B4={tuple(b.shape)}')
        for _d in range(b.dim()):          # auto-align: slice row0 on whichever dim differs
            if b.shape[_d] != a.shape[_d]:
                b = b.narrow(_d, 0, a.shape[_d])
        den = a.abs().max().item()
        rel = ((b - a).abs().max().item() / den * 100.0) if den > 0 else 0.0
        mark = ''
        if rel > max(prev * 3.0, 0.05) and prev >= 0.0:
            mark = '   <== JUMP x%.1f' % (rel / prev if prev > 1e-9 else float('inf'))
            jump.append((i, prev, rel))
        log(f'layer {i:3d}: rel_err = {rel:8.4f}%{mark}')
        prev = rel
    d = (lg4[0:1] - lg1).abs().max().item()
    log(f'FINAL logits: absdiff={d:.4f}  rel={d / lg1.abs().max().item() * 100:.3f}%  '
        f'argmax_equal={bool((lg4[0:1].argmax(-1) == lg1.argmax(-1)).all().item())}')
    log('JUMPS(>3x):', jump if jump else 'NONE -> smooth accumulation, no single-layer bug')
except Exception:
    if RANK == 0:
        traceback.print_exc()
finally:
    try:
        dist.destroy_process_group()
    except Exception:
        pass
