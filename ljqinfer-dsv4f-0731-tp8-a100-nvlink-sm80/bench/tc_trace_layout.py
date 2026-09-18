# -*- coding: utf-8 -*-
"""Layout-sensitivity tracer. Same as tc_oracle_tf but records a bit-hash of every module
output in call order. Run twice with PERTURB=0 / PERTURB=1 (different heap layout), then
python tc_trace_layout.py diff -> first module whose output differs."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, sys, time, hashlib
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)


def h(t):
    if not torch.is_tensor(t):
        return None
    if t.numel() * t.element_size() > 64 << 20:
        return 'BIG'  # weights: skip (immutable, hashing too slow)
    b = t.detach().contiguous().view(torch.uint8) if t.dtype.is_floating_point else t.detach().contiguous()
    return hashlib.blake2b(b.cpu().numpy().tobytes(), digest_size=8).hexdigest()


def outs(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, (list, tuple)):
        return [x for x in o if torch.is_tensor(x)]
    return []


if len(sys.argv) > 1 and sys.argv[1] == 'diff':
    a = torch.load('/tmp/trace_0.pt'); b = torch.load('/tmp/trace_1.pt')
    print(f'len {len(a)} vs {len(b)}')
    n = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f'#{i:5d} DIFF {x[0]:50s} {x[1]} != {y[1]}  shape={x[2]}')
            n += 1
            if n >= 25:
                break
    print('first diff above; total identical' if n == 0 else f'{n}+ diffs')
    sys.exit(0)

P = int(os.environ.get('PERTURB', '0'))
from model import wcache
from model.arch import Transformer
from model.bind import bind, check_bound
from model.args_dsv4 import make_args

ids = json.load(open('/tmp/c1_ids.json'))
gold = json.load(open('/tmp/gold_c1_out.json'))
full = ids + gold
W = wcache.load('tp8')
torch.set_default_dtype(torch.bfloat16)
args = make_args()
model = Transformer(args)
bind(model, W)
for m in model.modules():
    for k, v in list(m._buffers.items()):
        if v is not None and v.device.type == 'cpu':
            m._buffers[k] = v.to('cuda:0')
torch.set_default_device('cuda:0')
log('model ready, PERTURB =', P)

trace = []
junk = []
def mk(name):
    def hook(mod, inp, out):
        for j, o in enumerate(outs(out)):
            trace.append((f'{name}[{j}]', h(o), tuple(o.shape)))
    return hook
import ops as _ops
_orig_moe = _ops.moe_rank_fused_prefill_fp4
_moe_n = [0]
def _moe_traced(*a):
    k = _moe_n[0]; _moe_n[0] += 1
    for i, t in enumerate(a):
        if torch.is_tensor(t):
            trace.append((f'moe{k}.in{i}', h(t), tuple(t.shape)))
    outs = _orig_moe(*a)
    for i, t in enumerate(outs):
        trace.append((f'moe{k}.out{i}', h(t), tuple(t.shape)))
    return outs
_ops.moe_rank_fused_prefill_fp4 = _moe_traced

def pre(mod, inp):
    if P:
        w = getattr(mod, 'weight', None)
        if torch.is_tensor(w):
            torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)  # exact repro of the drift-triggering alloc
for name, mod in model.named_modules():
    if name:
        mod.register_forward_hook(mk(name))
        mod.register_forward_pre_hook(pre)

toks = torch.tensor([full], device='cuda:0')
with torch.no_grad():
    out_ids, logits, _ = model(toks, start_pos=0, full_logits=True)
lg = logits[0].float() if logits.dim() == 3 else logits.float()
Pn = len(ids)
p = torch.softmax(lg[Pn - 1], -1)
log(f'POS 0 p_gold={p[gold[0]].item():.4f} argmax={lg[Pn-1].argmax().item()} gold={gold[0]}')
trace.append(('logits', h(lg), tuple(lg.shape)))
torch.save(trace, f'/tmp/trace_{P}.pt')
log(f'saved {len(trace)} entries -> /tmp/trace_{P}.pt')
