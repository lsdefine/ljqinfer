# -*- coding: utf-8 -*-
"""Oracle greedy decode by repeated full prefill (absolutely safe: reuses the
TF-verified forward path only, no incremental cache semantics).
Each step: forward(prompt + generated), take last-position argmax, append.
nohup python tc_oracle_decode.py > /tmp/tc_oracle_decode.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

N_GEN = 12

try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    ids = json.load(open('/tmp/c1_ids.json'))
    log(f'prompt {len(ids)} tok; greedy decode {N_GEN} via repeated prefill')

    W = wcache.load('tp8')
    log('weights loaded')

    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0')
    torch.set_default_device('cuda:0')
    log('model ready')

    cur = list(ids)
    gen = []
    for step in range(N_GEN):
        toks = torch.tensor([cur], device='cuda:0')
        out_ids, logits, _ = model(toks, start_pos=0, full_logits=True)
        lg = logits[0] if logits.dim() == 3 else logits
        nxt = int(lg[-1].float().argmax())
        gen.append(nxt)
        cur.append(nxt)
        log(f'step {step}: id={nxt} {tk.decode([nxt], skip_special_tokens=False)!r}')

    log('GEN ids:', gen)
    log('GEN text:', repr(tk.decode(gen, skip_special_tokens=False)))
    gold = json.load(open('/tmp/gold_c1_out.json'))[:N_GEN]
    log('GOLD text:', repr(tk.decode(gold, skip_special_tokens=False)))
    log('match_gold_prefix:', gen == gold)
    log('DECODE DONE')
except Exception:
    traceback.print_exc()
    log('DECODE FAILED')
