# -*- coding: utf-8 -*-
"""Semantic + tool-call gate for batched decode (torchrun --nproc_per_node=8).

Each row is a REAL request encoded with the production encoder (server.encoding_dsv4),
decoded both alone (step_g) and batched (step_g_batch). Verdict is not ulp-based:
  * text must be semantically correct (keyword)
  * tool-call rows must parse into valid OpenAI-format tool_calls
  * batch text must equal single-row text
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
NSTEP = int(os.environ.get('NSTEP', '40'))
BMAX = int(os.environ.get('BMAX', '4'))
os.environ['LJQ_DECODE_G'] = '1'
SM8 = os.environ.get('LJQ_SMALLM8', '1')

try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer
    from server.encoding_dsv4 import encode_messages, parse_message_from_completion_text, eos_token
    arch.init_distributed()

    TK = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')

    WEATHER = {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather of a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}},
            "required": ["city"]}}}
    ADD = {"type": "function", "function": {
        "name": "add_numbers", "description": "Add two integers",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]}}}

    TASKS = [
        dict(tag='sem-capital', kind='sem', want=['北京'],
             msgs=[{"role": "user", "content": "中国的首都是哪座城市？只回答城市名，不要解释。"}]),
        dict(tag='sem-math', kind='sem', want=['391'],
             msgs=[{"role": "user", "content": "计算 17 乘以 23 等于多少？只输出最终数字。"}]),
        dict(tag='tool-weather', kind='tool', want_name='get_weather', want_arg='北京',
             msgs=[{"role": "system", "content": "You are a helpful assistant.", "tools": [WEATHER]},
                   {"role": "user", "content": "帮我查一下北京现在的天气。"}]),
        dict(tag='tool-add', kind='tool', want_name='add_numbers', want_arg='3',
             msgs=[{"role": "system", "content": "You are a helpful assistant.", "tools": [ADD]},
                   {"role": "user", "content": "请用工具计算 3 加 5 等于几。"}]),
    ][:BMAX]

    prompts = [encode_messages(t['msgs'], thinking_mode='chat') for t in TASKS]
    ids_list = [TK.encode(p, add_special_tokens=False).ids for t, p in zip(TASKS, prompts)]
    lens = [len(x) for x in ids_list]
    log(f'LJQ_SMALLM8={SM8}  row lens {lens}')

    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    room = NSTEP * Q + 32
    args = make_args(max_batch_size=2 * BMAX, max_seq_len=131072)
    model = Transformer(args)
    bind(model, W)
    from ops import peer_ar_rows
    peer_ar_rows.prewarm(peer_ar_rows.nmax_for_pool(model.pool, int(args.max_batch_size) * 8))  # register IPC mailbox before any graph capture
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to(dev)
    torch.set_default_device(dev)
    n_t = 0
    for m in model.modules():
        if hasattr(m, 'temperature'):
            m.temperature = 0.0; n_t += 1
    log(f'forced temperature=0 on {n_t} modules')
    dist.barrier()

    def prefill(slot, b):
        toks = torch.tensor(ids_list[b], dtype=torch.long, device=dev).unsqueeze(0)
        model.pool.ensure(slot, lens[b] + room)
        _, lg, _ = model(toks, start_pos=0, full_logits=True, slot=slot)
        return int(lg[0, -1].float().argmax())

    # ---------- baseline: each row alone through production step_g ----------
    base_seq = []
    for b in range(BMAX):
        first = prefill(BMAX + b, b)
        qin = torch.full((1, Q), first, dtype=torch.long, device=dev)
        pos_t = torch.tensor([lens[b]], dtype=torch.long, device=dev)
        seq = [first]
        for _ in range(NSTEP):
            out = model.step_g(qin, pos_t, BMAX + b)
            g, n_new = out[0], out[1]
            k = int(n_new.view(-1)[0])
            gv = g if g.dim() == 1 else g[0]
            seq += [int(t) for t in gv[:k]]
        base_seq.append(seq)
    log('baseline done', [len(s) for s in base_seq])

    # ---------- batch: BMAX rows together ----------
    firsts = [prefill(b, b) for b in range(BMAX)]
    qin_b = torch.stack([torch.full((Q,), f, dtype=torch.long, device=dev) for f in firsts])
    pos_b = torch.tensor(lens, dtype=torch.long, device=dev)
    slots = list(range(BMAX))
    bat_seq = [[f] for f in firsts]
    for _ in range(NSTEP):
        g, n_new, _ = model.step_g_batch(qin_b, pos_b, slots)
        for b in range(BMAX):
            k = int(n_new.view(-1)[b])
            bat_seq[b] += [int(t) for t in g[b, :k]]
    log('batch done', [len(s) for s in bat_seq])

    # ---------- verdict ----------
    def to_text(seq):
        txt = TK.decode(seq, skip_special_tokens=False)
        i = txt.find(eos_token)
        return (txt[:i], True) if i >= 0 else (txt, False)

    def judge(t, txt, hit_eos):
        if t['kind'] == 'sem':
            ok = any(w in txt for w in t['want'])
            return ok, ('keyword ok' if ok else f"MISSING {t['want']}")
        try:
            msg = parse_message_from_completion_text(txt + eos_token, 'chat')
        except Exception as e:
            return False, f'PARSE FAIL: {type(e).__name__}: {str(e)[:60]}'
        tcs = msg.get('tool_calls') or []
        if not tcs:
            return False, 'no tool_calls produced'
        fn = tcs[0]['function']
        ok = fn['name'] == t['want_name'] and t['want_arg'] in str(fn['arguments'])
        return ok, f"name={fn['name']} args={str(fn['arguments'])[:70]}"

    if RANK == 0:
        allok = True
        print('=' * 78, flush=True)
        for b, t in enumerate(TASKS):
            a, c = base_seq[b], bat_seq[b]
            n = min(len(a), len(c))
            div = next((i for i in range(n) if a[i] != c[i]), n)
            same = (div == n and len(a) == len(c))
            bt, be = to_text(a); ct, ce = to_text(c)
            ok_b, why_b = judge(t, bt, be)
            ok_c, why_c = judge(t, ct, ce)
            if not (ok_b and ok_c and (same or bt == ct)):
                allok = False
            print(f"[{t['tag']}] tokstream {'MATCH' if same else f'DIVERGE@{div}'} | "
                  f"text_equal={bt == ct} | eos b={be} c={ce}", flush=True)
            print(f"   B=1    {'PASS' if ok_b else 'FAIL'}  {why_b}", flush=True)
            print(f"   B={BMAX}    {'PASS' if ok_c else 'FAIL'}  {why_c}", flush=True)
            print(f"   text(B=1)   {bt[:150]!r}", flush=True)
            print(f"   text(B={BMAX})   {ct[:150]!r}", flush=True)
        print('=' * 78, flush=True)
        print(f'SEMGATE LJQ_SMALLM8={SM8} ' + ('PASS' if allok else 'FAIL'), flush=True)
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] SEMGATE CRASHED', flush=True)
