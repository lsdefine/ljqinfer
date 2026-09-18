# -*- coding: utf-8 -*-
"""Gate 4 (noise-floor version): chunked prefill vs whole prefill.
Whole-vs-prefix bit-exactness is NOT achievable (GEMM M-tiling noise, see
tc_prefix_l0.py: wq_a row0 differs 1e-3 between M=128 and M=411 even on HEAD).
Criteria:
  C1 first chunk (128 tok) == standalone prefill(128) bit-exact  (same M, same path)
  C2 TF gold hit-rate: chunked >= whole - 1
  C3 decode@T argmax equal
  info: argmax agreement whole-vs-chunked vs noise floor (prefill128 vs whole[:128])
nohup python tc_chunked_prefill.py > /tmp/tc_chunked_prefill.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, traceback, torch
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind
    from model.args_dsv4 import make_args
    ids = json.load(open('/tmp/c1_ids.json')); gold = json.load(open('/tmp/gold_c1_out.json'))
    full = ids + gold; P = len(ids); T = len(full)
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
    def hit(lg):
        return sum(int(lg[P - 1 + j].argmax()) == t for j, t in enumerate(gold))
    with torch.no_grad():
        _, lgA, _ = model(toks, start_pos=0, full_logits=True, slot=0); lgA = lgA[0].float()
        log('A whole done')
        _, lg1, _ = model(toks[:, :128], start_pos=0, full_logits=True, slot=2); lg1 = lg1[0].float()
        parts = []
        for s in range(0, T, 128):
            e = min(s + 128, T)
            _, lg, _ = model(toks[:, s:e], start_pos=s, full_logits=True, slot=1)
            parts.append(lg[0].float()); log(f'B chunk [{s},{e})')
        lgB = torch.cat(parts, 0)
        c1 = float((parts[0] - lg1).abs().max())
        log(f'C1 first chunk vs prefill128 max|d| {c1}', 'PASS' if c1 == 0 else 'FAIL')
        hA, hB = hit(lgA), hit(lgB)
        log(f'C2 TF gold hits: whole {hA}/{len(gold)} chunked {hB}/{len(gold)}', 'PASS' if hB >= hA - 1 else 'FAIL')
        noise = int((lg1.argmax(-1) == lgA[:128].argmax(-1)).sum())
        agree = int((lgA.argmax(-1) == lgB.argmax(-1)).sum())
        log(f'info argmax agree whole-vs-chunked {agree}/{T} ({agree/T:.3f}); noise floor prefill128-vs-whole[:128] {noise}/128 ({noise/128:.3f}); max|d| {float((lgA-lgB).abs().max()):.3f}')
        nt = toks[:, -1:]
        _, dA, _ = model(nt, start_pos=T, slot=0); _, dB, _ = model(nt, start_pos=T, slot=1)
        a, b = int(dA.float().flatten().argmax()), int(dB.float().flatten().argmax())
        log(f'C3 decode@{T} argmax A={a} B={b}', 'PASS' if a == b else 'FAIL')
        ok = c1 == 0 and hB >= hA - 1 and a == b
        log('GATE4', 'PASS' if ok else 'FAIL')
except Exception:
    traceback.print_exc(); log('EXC')
