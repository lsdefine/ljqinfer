# -*- coding: utf-8 -*-
"""B-batch decode correctness gate (torchrun --nproc_per_node=8).

Checks forward_q_g_batch (flattened [1, B*Q] stream, per-row attention) against the
already-accepted single-row forward_q_g:
  * rows have DIFFERENT lengths -> per-row pos_t must be honoured
  * baseline rows live on their own slots, so batch rows cannot read baseline KV
  * B=1 must reproduce forward_q_g; B=2..4 every row must match its own baseline
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
    # n_slots == args.max_batch_size (SlotPool(args.max_batch_size, ...)): we need
    # BMAX batch slots + BMAX baseline slots. 128K seq is plenty for a ~500 tok prompt
    # and keeps the compressed-pool budget (n_slots * max_seq) small.
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
    dist.barrier()

    # ---- rows with different prompt lengths (per-row pos must be honoured)
    lens = [len(ids), len(ids) - 13, len(ids) - 37, len(ids) - 71][:BMAX]
    log('row lens', lens)
    g = torch.Generator(device=dev); g.manual_seed(0)

    # baseline slots BMAX..2*BMAX-1 ; batch slots 0..BMAX-1  (identical prefill)
    first = {}
    for r, L in enumerate(lens):
        toks = torch.tensor([ids[:L]], device=dev)
        for slot in (r, BMAX + r):
            model.pool.ensure(slot, L + Q + 8)
            _, lg, _ = model(toks, start_pos=0, full_logits=True, slot=slot)
        first[r] = int(lg[0, -1].float().argmax())
    log('prefill done, first tokens', first)

    # identical qin per row for baseline and batch
    qin_row = []
    for r, L in enumerate(lens):
        draft = torch.randint(0, args.vocab_size, (1, Q - 1), device=dev, generator=g)
        qin_row.append(torch.cat([torch.tensor([[first[r]]], device=dev), draft], dim=1))

    # ---- baseline: single-row forward_q_g on slots BMAX+r
    base = {}
    for r, L in enumerate(lens):
        pos_b = torch.tensor([L], dtype=torch.int64, device=dev)
        lq, _ = model.forward_q_g(qin_row[r], pos_b, slot=BMAX + r)
        base[r] = lq[0].float().clone()          # [Q, V]
    log('baseline forward_q_g done')

    ok_all = True
    for B in range(1, BMAX + 1):
        qin = torch.cat(qin_row[:B], dim=0)                                   # [B, Q]
        pos_t = torch.tensor(lens[:B], dtype=torch.int64, device=dev)         # [B]
        slots = list(range(B))
        # fresh KV state for the batch slots: re-prefill so every B run starts clean
        for r in range(B):
            model.pool.ensure(r, lens[r] + Q + 8)
            model(torch.tensor([ids[:lens[r]]], device=dev), start_pos=0, full_logits=True, slot=r)
        torch.cuda.synchronize(); ta = time.time()
        lqb, _ = model.forward_q_g_batch(qin, pos_t, slots)                   # [B, Q, V]
        torch.cuda.synchronize(); dt = (time.time() - ta) * 1000
        assert lqb.shape[0] == B and lqb.shape[1] == Q, lqb.shape
        worst = 0.0; bad_rows = []
        for r in range(B):
            d = float((lqb[r].float() - base[r]).abs().max())
            am_b = int(lqb[r, 0].float().argmax()); am_o = int(base[r][0].argmax())
            worst = max(worst, d)
            if am_b != am_o or d > 0.05:
                bad_rows.append((r, d, am_o, am_b)); ok_all = False
        log(f'B={B}: eager {dt:7.1f}ms  maxdiff {worst:.5f}  '
            f'{"OK" if not bad_rows else "MISMATCH " + str(bad_rows)}')
    log('BATCH GATE PASS' if ok_all else 'BATCH GATE FAIL')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] GATE FAILED', flush=True)
