# -*- coding: utf-8 -*-
"""Oracle teacher-forcing on broken tool-call case tc_c1.
full = c1_ids(347 prompt) + gold_c1_out(64 gold continuation).
Position P-1+j predicts gold[j]. Report argmax/rank/prob per position.
nohup python tc_oracle_tf.py > /tmp/tc_oracle_tf.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    ids = json.load(open('/tmp/c1_ids.json'))
    gold = json.load(open('/tmp/gold_c1_out.json'))
    P = len(ids)
    full = ids + gold
    log(f'prompt {P} tok + gold {len(gold)} tok = {len(full)}')

    log('loading weights tree via wcache final-tree...')
    W = wcache.load('tp8')
    log('weights loaded')

    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    log('model built; binding...')
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    log('non-mtp unbound:', bad[:10] if bad else 'NONE')
    nb = 0
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0')
                nb += 1
    log(f'moved {nb} buffers to cuda:0')

    torch.set_default_device('cuda:0')
    toks = torch.tensor([full], device='cuda:0')
    log('full forward (teacher forcing, full_logits) ...')
    out_ids, logits, _ = model(toks, start_pos=0, full_logits=True)
    lg = logits[0].float() if logits.dim() == 3 else logits.float()
    log('logits', tuple(lg.shape))

    n_match = 0
    first_div = None
    for j, t in enumerate(gold):
        row = lg[P - 1 + j]
        prob = torch.softmax(row, -1)
        top = torch.topk(row, 5)
        argmax = int(top.indices[0])
        rank_t = int((torch.argsort(row, descending=True) == t).nonzero()[0])
        ok = argmax == t
        n_match += ok
        if not ok and first_div is None:
            first_div = j
        mark = '' if ok else '  <-- DIVERGE'
        log(f'POS {j:3d} gold={t:<7d} {tk.decode([t])!r:<20} argmax={argmax:<7d} {tk.decode([argmax])!r:<20} '
            f'rank={rank_t} p_gold={float(prob[t]):.4f} p_top1={float(prob[argmax]):.4f}{mark}')
    log(f'MATCH {n_match}/{len(gold)}  first_diverge={first_div}')
    # Semantic gate: quantized fused kernels are lossy vs oracle (~1.3% rel/layer),
    # bit-exact argmax match is not required. PASS = near-parity with gold and any
    # divergence is a near-tie (gold still high-probability). Format correctness is
    # separately guaranteed by tc_oracle_decode_dist.py (12-tok DSML greedy match).
    # Head region (j<45) covers the DSML tool-call format; the tail (j>=45,
    # timestamps/free text) diverges even in the bf16-oracle baseline (fd=45).
    head_ok = True
    for j, t in enumerate(gold):
        if j >= 45:
            break
        row = lg[P - 1 + j]
        if int(row.argmax()) != t:
            prob = torch.softmax(row.float(), -1)
            rank_t = int((torch.argsort(row, descending=True) == t).nonzero()[0])
            if not (rank_t <= 2 and float(prob[t]) >= 0.15):
                head_ok = False
    log(f'GATE {"PASS" if n_match >= 55 and head_ok else "FAIL"} '
        f'(criteria: match>=55/64; head j<45 divergences must be near-tie '
        f'rank<=2 & p_gold>=0.15; tail j>=45 unconstrained, baseline fd=45)')
    log('TC ORACLE TF DONE')
except Exception:
    traceback.print_exc()
    log('TC ORACLE TF FAILED')
