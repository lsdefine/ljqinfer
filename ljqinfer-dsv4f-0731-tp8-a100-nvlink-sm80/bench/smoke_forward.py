# -*- coding: utf-8 -*-
"""Remote smoke 2: full-model prefill forward on 8 tokens.
nohup python smoke_forward.py > /tmp/smoke_fwd.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

try:
    from model.weights import load_tp8
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args

    log('loading weights tree...')
    W = load_tp8(load_mtp=False)
    log('weights loaded')

    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    log('model built; binding...')
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    log('non-mtp unbound:', bad[:10] if bad else 'NONE')

    # move all buffers (freqs_cis, kv_cache, ...) to cuda:0
    nb = 0
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0')
                nb += 1
    log(f'moved {nb} buffers to cuda:0')

    torch.set_default_device('cuda:0')  # arch creates masks/aranges without device
    toks = torch.tensor([[0, 3085, 344, 270, 6102, 294, 8760, 33]], device='cuda:0')  # bos + 'What is the capital of France?'
    log('full forward start_pos=0 ...')
    out_ids, logits, main_hidden = model(toks, start_pos=0)
    log('logits', tuple(logits.shape), logits.dtype,
        'mean_abs', float(logits.float().abs().mean()),
        'max', float(logits.float().max()))
    log('out_ids', out_ids.tolist())
    lg = logits.float()
    top = lg[-1] if lg.dim() == 2 else lg[0, -1]
    vals, idx = top.topk(5)
    log('last-token top5:', list(zip(idx.tolist(), [round(v, 3) for v in vals.tolist()])))
    log('main_hidden', None if main_hidden is None else tuple(main_hidden.shape))
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    log('top5 decoded:', [tk.decode([i]) for i in idx.tolist()])
    log('SMOKE FWD OK')
except Exception:
    traceback.print_exc()
    log('SMOKE FWD FAILED')
