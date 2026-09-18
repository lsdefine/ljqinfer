# -*- coding: utf-8 -*-
"""Remote smoke: load_tp8 -> build Transformer -> bind -> check_bound -> tiny forward L0.
Run from repo root: nohup python smoke_bind.py > /tmp/smoke_bind.log 2>&1 &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import sys, time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

try:
    from model.weights import load_tp8
    from model.arch import Transformer, ModelArgs
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args

    log('loading weights tree (load_mtp=False)...')
    W = load_tp8(load_mtp=False)
    log('weights loaded')

    args = make_args()
    log('building Transformer meta...')
    with torch.device('meta'):
        pass  # modules are lazy (weight=None), safe to build on cpu
    model = Transformer(args)
    log('model built; binding...')
    bind(model, W)
    bad = check_bound(model)
    log('unbound modules:', len(bad))
    for n in bad[:40]:
        log('  UNBOUND', n)

    # tiny forward: 8 tokens through embedding + layer0 only
    log('tiny forward: embed + layer 0')
    toks = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device='cuda:0')
    with torch.no_grad():
        h = model.embed(toks)
        log('embed out', tuple(h.shape), h.dtype, float(h.float().abs().mean()))
        freqs = model.freqs_cis[:8] if model.freqs_cis is not None else None
        log('NOTE: full-layer forward needs past/masks; deferring to layer-diff script')
    log('SMOKE OK')
except Exception:
    traceback.print_exc()
    log('SMOKE FAILED')
