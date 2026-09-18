import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os
# -*- coding: utf-8 -*-
"""Gate 6a: DSpark draft on canonical LayerPast (new) vs old ring-window impl (HEAD~1).
Teacher-forced on tc_c1 gold, temperature=0. Compare draft output_ids per step.
usage: python tc_dspark_draft.py [n_steps]   (single process, tp8 wcache)
"""
import json, sys, time, traceback, importlib.util
import torch

t0 = time.time()
def log(*a):
    print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 32

try:
    from model import wcache
    from model import arch as arch_new
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args

    ids = json.load(open('/tmp/c1_ids.json'))
    gold = json.load(open('/tmp/gold_c1_out.json'))
    P = len(ids)
    log(f'prompt {P} tok, gold {len(gold)} tok, steps {N_STEPS}')

    W = wcache.load('tp8')
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    args.temperature = 0
    model = Transformer(args)
    bind(model, W)
    bad = check_bound(model)
    log('unbound:', bad[:10] if bad else 'NONE')
    nb = 0
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to('cuda:0'); nb += 1
    log(f'moved {nb} buffers')
    torch.set_default_device('cuda:0')
    B = args.dspark_block_size

    attn_outs = {}
    def run(tag):
        attn_outs[tag] = []
        hs = [m.attn.register_forward_hook(lambda mod, i, o, k=k: attn_outs[tag].append((i[0].detach().float().cpu(), o.detach().float().cpu()))) for k, m in enumerate(model.mtp)]
        try:
            return _run(tag)
        finally:
            for h in hs: h.remove()

    def _run(tag):
        toks = torch.tensor([ids], device='cuda:0')
        out_ids, logits, main_hidden = model(toks, start_pos=0)
        model.forward_spec(out_ids, main_hidden, 0)
        drafts = []
        for j in range(N_STEPS):
            tok = torch.tensor([[gold[j]]], device='cuda:0')
            pos = P + j
            out_ids, logits, main_hidden = model(tok, start_pos=pos)
            d_ids, d_logits, conf = model.forward_spec(out_ids, main_hidden, pos)
            drafts.append((d_ids[0].tolist(), d_logits[0].float().cpu(), conf[0].float().cpu()))
        torch.cuda.synchronize()
        log(f'{tag}: done {N_STEPS} steps')
        return drafts

    drafts_new = run('NEW(past)')

    # ---- old ring impl pinned to 7cca5db (last commit before canonical-past DSparkAttention) ----
    import subprocess
    # old arch needs the old past module too (pinned to the same commit)
    open('model/_past_old.py', 'w').write(subprocess.check_output(['git', 'show', '7cca5db:model/past.py'], text=True))
    src = subprocess.check_output(['git', 'show', '7cca5db:model/arch.py'], text=True)
    src = src.replace('from .past import', 'from ._past_old import')
    open('model/_arch_old.py', 'w').write(src)
    spec = importlib.util.spec_from_file_location('model._arch_old', 'model/_arch_old.py')
    arch_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(arch_old)
    os.remove('model/_arch_old.py'); os.remove('model/_past_old.py')
    for g in ('world_size', 'rank', 'default_dtype', 'scale_fmt', 'scale_dtype'):
        if hasattr(arch_new, g): setattr(arch_old, g, getattr(arch_new, g))
    for k, blk in enumerate(model.mtp):
        new_attn = blk.attn
        old_attn = arch_old.DSparkAttention(args.n_layers + k, args)
        for n, mod in new_attn._modules.items(): old_attn._modules[n] = mod
        for n, p in new_attn._parameters.items(): old_attn._parameters[n] = p
        for n, b in new_attn._buffers.items():
            if n != 'kv_cache': old_attn._buffers[n] = b
        old_attn.kv_cache = old_attn.kv_cache.to('cuda:0')
        class _NoSlot(torch.nn.Module):   # old ring impl has no slot arg
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x, start_pos, main_x, slot=0): return self.m(x, start_pos, main_x)
        blk.attn = _NoSlot(old_attn)
    log('swapped mtp attention -> OLD ring impl')
    drafts_old = run('OLD(ring)')

    # ---- compare ----
    an_, ao_ = attn_outs['NEW(past)'], attn_outs['OLD(ring)']
    nm = len(model.mtp)
    for j in (0, 1, N_STEPS - 1):
        for k in range(nm):
            (ia, a), (ib, b) = an_[(1 + j) * nm + k], ao_[(1 + j) * nm + k]   # index 0..nm-1 = prefill call
            rel = ((a - b).norm() / (a.norm() + 1e-6)).item(); reli = ((ia - ib).norm() / (ia.norm() + 1e-6)).item()
            log(f'  attn step {j:2d} mtp[{k}] in_rel={reli:.3e} out_rel={rel:.3e} amp={rel/max(reli,1e-9):.1f}x')
    same = 0; first_div = None; acc_new = 0; acc_old = 0; maxdiff = 0.0
    for j, (dn, do) in enumerate(zip(drafts_new, drafts_old)):
        eq = dn[0] == do[0]
        same += eq
        if not eq and first_div is None: first_div = j
        maxdiff = max(maxdiff, (dn[1] - do[1]).abs().max().item())
        d = (dn[1] - do[1]).abs()
        if j < 4 or j == N_STEPS - 1:
            log(f'  step {j:2d} logits |diff| max={d.max().item():.3e} mean={d.mean().item():.3e}  |logit| max={dn[1].abs().max().item():.2f}  conf new={dn[2].tolist()} old={do[2].tolist()}')
        g = gold[j + 1:j + B + 2]  # draft follows the token predicted at pos P+j
        an = sum(1 for a, b in zip(dn[0], g) if a == b); ao = sum(1 for a, b in zip(do[0], g) if a == b)
        acc_new += an; acc_old += ao
        flag = '' if eq else '  <-- DIFF'
        log(f'step {j:2d} new={dn[0]} old={do[0]} gold={g} match_gold new={an}/{len(g)} old={ao}/{len(g)}{flag}')
    log(f'SAME draft ids {same}/{N_STEPS}  first_div={first_div}  logits maxdiff={maxdiff:.4e}')
    log(f'gold-hit new={acc_new} old={acc_old} (of {N_STEPS*(B+1)})')
    ok = same == N_STEPS or (first_div is not None and first_div >= N_STEPS - 2 and maxdiff < 0.5)
    log('GATE6A PASS' if ok else 'GATE6A FAIL')
    log('TC DSPARK DRAFT DONE')
except Exception:
    traceback.print_exc()
    log('TC DSPARK DRAFT FAILED')
