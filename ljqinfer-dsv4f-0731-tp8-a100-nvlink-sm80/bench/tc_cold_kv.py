# -*- coding: utf-8 -*-
"""Gate 5: cold-KV export/import.
slot1: prefill [0,256) as 2 chunks -> export_cold [0,128) + [128,256)+tail -> CPU
round-trip -> import into slot2 -> both slots continue prefill [256,411).
Criteria:
  C1 slot2 continuation == slot1 continuation bit-exact (identical state, same M)
  C2 TF gold hits on slot2 >= whole-prefill (slot0) - 1
  C3 decode@411 argmax slot0 == slot1 == slot2
nohup python tc_cold_kv.py > /tmp/tc_cold_kv.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, traceback, torch
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

def to_cpu(seg):
    return {k: ({l: {n: t.cpu().clone() for n, t in b.items()} for l, b in v.items()} if k in ('layers', 'tail') else v)
            for k, v in seg.items()}

def to_dev(seg, dev):
    return {k: ({l: {n: t.to(dev) for n, t in b.items()} for l, b in v.items()} if k in ('layers', 'tail') else v)
            for k, v in seg.items()}

try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind
    from model.args_dsv4 import make_args
    ids = json.load(open('/tmp/c1_ids.json')); gold = json.load(open('/tmp/gold_c1_out.json'))
    full = ids + gold; T = len(full); P = len(ids)
    log('total', T, 'prompt', P)
    W = wcache.load('tp8')
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(make_args()); bind(model, W)
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0')
    torch.set_default_device('cuda:0')
    toks = torch.tensor([full], device='cuda:0')
    def hits(lg):
        lg = (lg[0] if lg.dim() == 3 else lg).float()
        return int(sum(int(lg[P - 1 + j].argmax()) == t for j, t in enumerate(gold)))

    # A: whole on slot 0
    _, lgA, _ = model(toks, start_pos=0, full_logits=True, slot=0)
    hA = hits(lgA); log('A whole hits', hA)

    # B: slot1 chunks [0,128),[128,256)
    model(toks[:, 0:128], start_pos=0, slot=1)
    model(toks[:, 128:256], start_pos=128, slot=1)
    segs = [model.pool.export_cold(1, 0, 128), model.pool.export_cold(1, 128, 256)]
    assert 'tail' in segs[-1] and 'tail' not in segs[0]
    nbytes = sum(t.numel() * t.element_size() for s in segs for d in (s['layers'], s.get('tail', {})) for b in d.values() for t in b.values())
    log(f'exported 2 segs, {nbytes/2**20:.1f} MiB')
    cold = [to_cpu(s) for s in segs]; del segs
    torch.cuda.synchronize()
    # import into slot 2
    newpos = model.pool.import_cold(2, [to_dev(s, 'cuda:0') for s in cold])
    log('imported -> pos', newpos)
    assert newpos == 256
    # continue both
    _, lg1, _ = model(toks[:, 256:], start_pos=256, full_logits=True, slot=1)
    _, lg2, _ = model(toks[:, 256:], start_pos=256, full_logits=True, slot=2)
    lg1 = (lg1[0] if lg1.dim() == 3 else lg1).float(); lg2 = (lg2[0] if lg2.dim() == 3 else lg2).float()
    d = float((lg1 - lg2).abs().max())
    c1 = d == 0.0
    log(f'C1 cont slot1 vs slot2 max|d| {d:.4f}', 'PASS' if c1 else 'FAIL')
    lgA = (lgA[0] if lgA.dim() == 3 else lgA).float()
    h2 = int(sum(int(lg2[P - 1 - 256 + j].argmax()) == t for j, t in enumerate(gold)))
    h1 = int(sum(int(lg1[P - 1 - 256 + j].argmax()) == t for j, t in enumerate(gold)))
    c2 = h2 >= hA - 1
    log(f'C2 gold hits whole {hA}/64 chunked-cont {h1}/64 imported-cont {h2}/64', 'PASS' if c2 else 'FAIL')
    nt = toks[:, T-1:T]
    am = []
    for s in (2, 0, 1):
        _, dl, _ = model(nt, start_pos=T, slot=s); am.append(int(dl.float().flatten().argmax()))
    nt2 = torch.tensor([[am[0]]], device=nt.device, dtype=nt.dtype)
    for s in (1, 0, 2):
        _, dl, _ = model(nt2, start_pos=T + 1, slot=s); am.append(int(dl.float().flatten().argmax()))

    pool = model.pool
    log('pos', pool.pos[1], pool.pos[2], 'pos_dev', int(pool.pos_dev[1]), int(pool.pos_dev[2]))
    for l, lp in sorted(pool.layers.items()):
        d = {}
        d['kv'] = (lp.kv(1, 0, T+1).float() - lp.kv(2, 0, T+1).float()).abs().max().item()
        if hasattr(lp, 'ckv'):
            d['ckv'] = (lp.ckv(1, T+1).float() - lp.ckv(2, T+1).float()).abs().max().item()
        if hasattr(lp, 'ickv'):
            d['ickv'] = (lp.ickv(1, T+1).float() - lp.ickv(2, T+1).float()).abs().max().item()
        if getattr(lp, 'res_x', None) is not None:
            d['res'] = (lp.res_x[1].float() - lp.res_x[2].float()).abs().max().item()
        bad = {k: v for k, v in d.items() if v > 0}
        if bad:
            dk = (lp.kv(1, 0, T+1).float() - lp.kv(2, 0, T+1).float()).abs().amax(-1)
            nz = (dk > 0).nonzero().flatten().tolist()
            log('L', l, type(lp).__name__, bad, 'kv diff pos first/last/n', nz[:3], nz[-2:], len(nz))
            if 'ckv' in bad:
                dc = (lp.ckv(1, T+1).float() - lp.ckv(2, T+1).float()).abs().amax(-1); nc=(dc>0).nonzero().flatten().tolist()
                log('   ckv diff idx', nc[:5], len(nc), 'of', dc.numel())
            if 'ickv' in bad:
                dc = (lp.ickv(1, T+1).float() - lp.ickv(2, T+1).float()).abs().amax(-1); nc=(dc>0).nonzero().flatten().tolist()
                log('   ickv diff idx', nc[:5], len(nc))
            if l >= 4: break
    c3 = len(set(am[:3])) == 1 and len(set(am[3:])) == 1
    log(f'C3 decode@{T} argmax {am}', 'PASS' if c3 else 'FAIL')
    log('GATE5', 'PASS' if c1 and c2 and c3 else 'FAIL')
except Exception:
    traceback.print_exc(); log('EXC')
