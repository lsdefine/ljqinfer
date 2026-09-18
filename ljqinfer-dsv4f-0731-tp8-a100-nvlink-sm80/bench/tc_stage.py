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

    # ---------- stage-level localisation of the B=1 vs B=4 delta ----------
    firsts_b = [prefill(b, lens[b]) for b in range(BMAX)]
    _ = prefill(BMAX, lens[0])
    qin_b = torch.stack([torch.full((Q,), f, dtype=torch.long, device=dev) for f in firsts_b])
    pos_b = torch.tensor(lens, dtype=torch.long, device=dev)

    from model.arch import Block
    from ops import hc_pre_norm
    recs = []
    orig_qb = Block.forward_qb

    def probe_qb(self, x, pos_list, slots, input_ids):
        residual = x
        x1, post, comb = hc_pre_norm(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                     self.attn_norm.weight, self.attn_norm.eps)
        B = len(slots)
        Qn = x1.size(1) // B
        outs = [self.attn.forward_q(x1[:, b * Qn:(b + 1) * Qn], pos_list[b], slots[b]) for b in range(B)]
        xa = outs[0] if B == 1 else torch.cat(outs, dim=1)
        xp = self.hc_post(xa, residual, post, comb)
        y = self.fused_ffn(xp, input_ids, q_decode=True)
        recs.append({'1_pre': x1.detach().float().clone(), '2_attn': xa.detach().float().clone(),
                     '3_post': xp.detach().float().clone(), '4_ffn_out': y.detach().float().clone()})
        return y

    Block.forward_qb = probe_qb
    try:
        recs.clear()
        model.forward_q_g_batch(qin_b[:1], pos_b[:1], [BMAX])
        r1 = list(recs)
        recs.clear()
        model.forward_q_g_batch(qin_b, pos_b, list(range(BMAX)))
        r4 = list(recs)
    finally:
        Block.forward_qb = orig_qb

    keys = ['1_pre', '2_attn', '3_post', '4_ffn_out']
    log('stage-wise rel_err(%) of row0, B=1 vs B=4   [layer 0 is the decisive one]')
    for i in (0, 1, 2, 3, 5, 10, 20, 42):
        if i >= min(len(r1), len(r4)):
            continue
        parts = []
        for k in keys:
            a, b = r1[i][k], r4[i][k]
            for _d in range(b.dim()):
                if b.shape[_d] != a.shape[_d]:
                    b = b.narrow(_d, 0, a.shape[_d])
            den = a.abs().max().item()
            bit = 'BITEXACT' if torch.equal(a, b) else '%.4f%%' % ((b - a).abs().max().item() / den * 100)
            parts.append(f'{k}={bit}')
        log(f'  layer {i:3d}: ' + '  '.join(parts))
except Exception:
    if RANK == 0:
        traceback.print_exc()
finally:
    try:
        dist.destroy_process_group()
    except Exception:
        pass
