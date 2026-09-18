# -*- coding: utf-8 -*-
"""Gate: BnQ8 eager decode step (ops/attn_decode_q.py) vs single-token decode oracle.
slot 0: single-token decode chain (attn_ref decode ops, verified path)
slot 1: Q=8 chain, row 0 = true token, rows 1..7 = random draft tokens (rejected)
Checks per step: argmax(row0) == oracle argmax, max|logit diff|, idempotency (re-run
the same step twice -> identical row0 logits). Reports mean Q-step time.
nohup python tc_decode_q.py > /tmp/tc_decode_q.log 2>&1 < /dev/null &
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, traceback
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

N_GEN, Q = 24, 8

try:
    from model import wcache
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    ids = json.load(open('/tmp/c1_ids.json'))
    W = wcache.load('tp8')
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
    log(f'model ready; prompt {len(ids)} tok')

    P = len(ids)
    toks = torch.tensor([ids], device='cuda:0')
    _, lg0, _ = model(toks, start_pos=0, full_logits=True, slot=0)
    _, lg1, _ = model(toks, start_pos=0, full_logits=True, slot=1)
    first = int(lg0[0, -1].float().argmax())
    assert first == int(lg1[0, -1].float().argmax())
    log('prefill done, first token', first)

    g = torch.Generator(device='cuda:0'); g.manual_seed(0)
    cur = first; gen = [first]; ok = True; times = []; maxdiff = 0.0
    for step in range(N_GEN):
        pos = P + step
        # oracle: single token
        _, lo, _ = model(torch.tensor([[cur]], device='cuda:0'), start_pos=pos, full_logits=True, slot=0)
        lo = lo.reshape(-1, lo.shape[-1])[-1].float()
        # Q step: true token + random drafts
        draft = torch.randint(0, args.vocab_size, (1, Q - 1), device='cuda:0', generator=g)
        qin = torch.cat([torch.tensor([[cur]], device='cuda:0'), draft], dim=1)
        torch.cuda.synchronize(); ta = time.time()
        lq, _ = model.forward_q(qin, pos, slot=1)
        torch.cuda.synchronize(); times.append(time.time() - ta)
        lq2, _ = model.forward_q(qin, pos, slot=1)   # idempotency re-run
        r0 = lq[0, 0].float(); r0b = lq2[0, 0].float()
        d = float((r0 - lo).abs().max()); maxdiff = max(maxdiff, d)
        idem = bool(torch.equal(r0, r0b))
        a_o, a_q = int(lo.argmax()), int(r0.argmax())
        top2 = torch.topk(lo, 2).values; gap = float(top2[0] - top2[1])
        if a_o != a_q or not idem:
            ok = False
        log(f'step {step} pos {pos}: oracle {a_o} q {a_q} {"OK" if a_o == a_q else "MISMATCH"} '
            f'maxdiff {d:.4f} gap {gap:.3f} idem {idem} t {times[-1]*1000:.1f}ms')
        cur = a_o; gen.append(cur)
    log('GEN text:', repr(tk.decode(gen, skip_special_tokens=False)))
    gold = json.load(open('/tmp/gold_c1_out.json'))[:len(gen)]
    log('match_gold_prefix:', gen == gold)
    log(f'Q-step mean {sum(times[2:])/len(times[2:])*1000:.1f}ms  maxdiff {maxdiff:.4f}')
    log('GATE PASS' if ok else 'GATE FAIL')
except Exception:
    traceback.print_exc()
    log('GATE FAILED')
