# -*- coding: utf-8 -*-
"""NCCL TP8 B1Q8 decode step CUDA-graph capture gate (torchrun --nproc_per_node=8).
Per step: oracle single-token eager (slot0) vs eager forward_q (slot1) vs graph replay (slot1)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, time, traceback
import torch
import torch.distributed as dist

t0 = time.time()
RANK = int(os.environ['RANK'])
def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
N_GEN = int(os.environ.get('N_GEN', '24')); Q = 8
os.environ['LJQ_DECODE_G'] = '1'
try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    from tokenizers import Tokenizer
    arch.init_distributed()
    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    ids = json.load(open('/tmp/c1_ids.json'))
    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    model = Transformer(args)
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith('mtp.')]
    assert not bad, bad
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to(dev)
    torch.set_default_device(dev)
    dist.barrier()
    P = len(ids)
    log(f'model ready; prompt {P} tok')
    toks = torch.tensor([ids], device=dev)
    _, lg0, _ = model(toks, start_pos=0, full_logits=True, slot=0)
    _, lg1, _ = model(toks, start_pos=0, full_logits=True, slot=1)
    first = int(lg0[0, -1].float().argmax()); assert first == int(lg1[0, -1].float().argmax())
    log('prefill done, first token', first)
    model.pool.ensure(1, P + N_GEN + Q + 1)

    g = torch.Generator(device=dev); g.manual_seed(0)
    qin_buf = torch.zeros(1, Q, dtype=torch.int64, device=dev)
    pos_buf = torch.zeros(1, dtype=torch.int64, device=dev)
    pos_buf.fill_(P); qin_buf[0, 0] = first
    # warmup on side stream then capture (NCCL collectives inside graph)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            model.forward_q_g(qin_buf, pos_buf, slot=1)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lq_g, _ = model.forward_q_g(qin_buf, pos_buf, slot=1)
    torch.cuda.synchronize(); dist.barrier()
    log('graph captured')

    cur = first; gen = [first]; ok = True; t_e = []; t_g = []; maxdiff = 0.0; maxdiff_eg = 0.0
    for step in range(N_GEN):
        pos = P + step
        _, lo, _ = model(torch.tensor([[cur]], device=dev), start_pos=pos, full_logits=True, slot=0)
        lo = lo.reshape(-1, lo.shape[-1])[-1].float()
        draft = torch.randint(0, args.vocab_size, (1, Q - 1), device=dev, generator=g)
        qin = torch.cat([torch.tensor([[cur]], device=dev), draft], dim=1)
        torch.cuda.synchronize(); ta = time.time()
        lq, _ = model.forward_q(qin, pos, slot=1)
        torch.cuda.synchronize(); t_e.append(time.time() - ta)
        r_e = lq[0, 0].float().clone()
        qin_buf.copy_(qin); pos_buf.fill_(pos)
        torch.cuda.synchronize(); ta = time.time()
        graph.replay()
        torch.cuda.synchronize(); t_g.append(time.time() - ta)
        r_g = lq_g[0, 0].float()
        d_o = float((r_g - lo).abs().max()); d_e = float((r_g - r_e).abs().max())
        maxdiff = max(maxdiff, d_o); maxdiff_eg = max(maxdiff_eg, d_e)
        a_o, a_e, a_g = int(lo.argmax()), int(r_e.argmax()), int(r_g.argmax())
        top2 = torch.topk(lo, 2).values; gap = float(top2[0] - top2[1])
        if a_o != a_g: ok = False
        log(f'step {step} pos {pos}: oracle {a_o} eager {a_e} graph {a_g} {"OK" if a_o == a_g else "MISMATCH"} '
            f'd_oracle {d_o:.4f} d_eager {d_e:.4f} gap {gap:.3f} eager {t_e[-1]*1000:.1f}ms graph {t_g[-1]*1000:.1f}ms')
        cur = a_o; gen.append(cur)
    if RANK == 0:
        log('GEN text:', repr(tk.decode(gen, skip_special_tokens=False)))
        gold = json.load(open('/tmp/gold_c1_out.json'))[:len(gen)]
        log('match_gold_prefix:', gen == gold)
        log(f'eager mean {sum(t_e[2:])/len(t_e[2:])*1000:.1f}ms  graph mean {sum(t_g[2:])/len(t_g[2:])*1000:.1f}ms  '
            f'maxdiff_oracle {maxdiff:.4f} maxdiff_eager {maxdiff_eg:.4f}')
        log('GATE PASS' if ok else 'GATE FAIL')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] GATE FAILED', flush=True)
