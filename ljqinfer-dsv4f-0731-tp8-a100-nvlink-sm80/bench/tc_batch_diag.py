# -*- coding: utf-8 -*-
"""Verdict test for the B>=2 logits drift seen in tc_batch_b234.

Two batch rows are fed the IDENTICAL prompt, identical qin and identical pos, but
live on different slots.  Then:
  row0 vs row1   -> if they differ, rows leak into each other (real bug).
  row0 vs B1     -> if rows match each other but drift from the single-row baseline,
                    the MoE kernel merely changes its blocking/reduction order with
                    the token count (8 -> 16), i.e. numerical noise, not a bug.
Also prints the logit scale so the absolute maxdiff can be judged in context.
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
    args = make_args(max_batch_size=4, max_seq_len=131072)
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
    dist.barrier()

    P = len(ids)
    toks = torch.tensor([ids], device=dev)
    for slot in (0, 1, 2):
        model.pool.ensure(slot, P + Q + 8)
        _, lg, _ = model(toks, start_pos=0, full_logits=True, slot=slot)
    first = int(lg[0, -1].float().argmax())
    log(f'prefill done on slots 0,1,2; prompt {P} tok; first {first}')

    qin = torch.zeros(1, Q, dtype=torch.int64, device=dev)
    qin[0, 0] = first
    pos1 = torch.zeros(1, dtype=torch.int64, device=dev); pos1.fill_(P)

    # ---- single-row baseline on slot 2
    lg_b1, _ = model.forward_q_g(qin, pos1, slot=2)
    lg_b1 = lg_b1.float().reshape(Q, -1)
    log(f'B1 logits scale: absmax {lg_b1.abs().max():.3f}  std {lg_b1.std():.3f}')

    # ---- B=2, both rows IDENTICAL content/pos, on slots 0 and 1
    qin2 = torch.cat([qin, qin], dim=0)                       # [2, Q]
    pos2 = torch.stack([pos1[0], pos1[0]])                    # [2]
    lg_b2, _ = model.forward_q_g_batch(qin2, pos2, [0, 1])
    lg_b2 = lg_b2.float()                                     # [2, Q, V]

    d_rows = (lg_b2[0] - lg_b2[1]).abs().max().item()
    d_0_b1 = (lg_b2[0] - lg_b1).abs().max().item()
    d_1_b1 = (lg_b2[1] - lg_b1).abs().max().item()
    rel = d_0_b1 / max(lg_b1.abs().max().item(), 1e-9)
    am_b1 = lg_b1.argmax(-1).tolist()
    am_r0 = lg_b2[0].argmax(-1).tolist()
    am_r1 = lg_b2[1].argmax(-1).tolist()
    log(f'row0 vs row1 (identical inputs): maxdiff {d_rows:.6f}')
    log(f'row0 vs B1: maxdiff {d_0_b1:.6f}  (rel {rel*100:.2f}% of absmax)')
    log(f'row1 vs B1: maxdiff {d_1_b1:.6f}')
    log(f'argmax  B1 {am_b1}')
    log(f'argmax  r0 {am_r0}')
    log(f'argmax  r1 {am_r1}')
    if d_rows == 0.0:
        log('VERDICT: rows are bit-identical -> NO cross-row leakage; '
            'drift vs B1 is token-count-dependent kernel numerics')
    else:
        log('VERDICT: rows DIFFER on identical input -> real cross-row bug')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] DIAG FAILED', flush=True)
