# -*- coding: utf-8 -*-
"""NCCL TP8 DSpark draft step CUDA-graph capture gate (torchrun --nproc_per_node=8).
Teacher-forced on gold. Per step: oracle eager forward_spec (slot0) vs eager forward_spec (slot1)
vs graph forward_spec_g replay (slot1). Compare draft ids / logits / confidence."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, time, traceback
import torch
import torch.distributed as dist

t0 = time.time()
RANK = int(os.environ['RANK'])
def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)
N_GEN = int(os.environ.get('N_GEN', '24'))
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
    gold = json.load(open('/tmp/gold_c1_out.json'))
    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
    args.temperature = 0.0
    model = Transformer(args)
    bind(model, W)
    bad = check_bound(model)
    log('unbound:', bad[:20])
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu':
                m._buffers[k] = v.to(dev)
    torch.set_default_device(dev)
    dist.barrier()
    P = len(ids); B = args.dspark_block_size
    log(f'model ready; prompt {P} tok; block_size {B}; mtp layers {len(model.mtp)}')
    toks = torch.tensor([ids], device=dev)
    out0, _, mh0 = model(toks, start_pos=0, slot=0)
    out1, _, mh1 = model(toks, start_pos=0, slot=1)
    model.forward_spec(out0, mh0, 0, slot=0)
    model.forward_spec(out1, mh1, 0, slot=1)
    first = int(out0.flatten()[-1]); assert first == int(out1.flatten()[-1]) == gold[0], (first, gold[0])
    log('prefill done, first token', first, 'mh shape', tuple(mh0.shape), mh0.dtype)
    model.pool.ensure(1, P + N_GEN + B + 2)

    tok_buf = torch.zeros(1, dtype=torch.int64, device=dev)
    mh_buf = torch.zeros(1, 1, mh0.shape[-1], dtype=mh0.dtype, device=dev)
    pos_buf = torch.zeros(1, dtype=torch.int64, device=dev)
    tok_buf.fill_(first); pos_buf.fill_(P)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            model.forward_spec_g(tok_buf, mh_buf, pos_buf, slot=1)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        d_g, l_g, c_g = model.forward_spec_g(tok_buf, mh_buf, pos_buf, slot=1)
    torch.cuda.synchronize(); dist.barrier()
    log('graph captured')

    ok = True; t_e = []; t_g = []; md_o = 0.0; md_e = 0.0; md_eg = 0.0; same_o = 0
    for step in range(N_GEN):
        pos = P + step
        tok = torch.tensor([gold[step]], device=dev)
        _, _, m0 = model(tok.view(1, 1), start_pos=pos, slot=0)
        d_o, l_o, c_o = model.forward_spec(tok, m0, pos, slot=0)
        _, _, m1 = model(tok.view(1, 1), start_pos=pos, slot=1)
        torch.cuda.synchronize(); ta = time.time()
        d_e, l_e, c_e = model.forward_spec(tok, m1, pos, slot=1)
        torch.cuda.synchronize(); t_e.append(time.time() - ta)
        d_e = d_e.clone(); l_e = l_e.float().clone(); c_e = c_e.float().clone()
        tok_buf.copy_(tok); mh_buf.copy_(m1); pos_buf.fill_(pos)
        d_eg, l_eg, _ = model.forward_spec_g(tok_buf, mh_buf, pos_buf, slot=1)   # eager run of the graph fn
        d_eg = d_eg.clone(); l_eg = l_eg.float().clone()
        torch.cuda.synchronize(); ta = time.time()
        graph.replay()
        torch.cuda.synchronize(); t_g.append(time.time() - ta)
        de = float((l_g.float() - l_e).abs().max()); do = float((l_g.float() - l_o.float()).abs().max())
        md_e = max(md_e, de); md_o = max(md_o, do)
        deg = float((l_g.float() - l_eg).abs().max()); md_eg = max(md_eg, deg)
        if not torch.equal(d_g, d_eg): ok = False
        eq_e = bool(torch.equal(d_g, d_e)); eq_o = bool(torch.equal(d_g, d_o)); same_o += eq_o
        if not eq_e: ok = False
        g_ids = d_g[0].tolist(); gold_next = gold[step + 1: step + 1 + B]
        hit = sum(int(a == b) for a, b in zip(g_ids[1:], gold_next))
        log(f'step {step} pos {pos}: graph==eager {eq_e} graph==oracle {eq_o} d_eager {de:.4f} d_eagerG {deg:.4f} d_oracle {do:.4f} '
            f'conf_g {float(c_g.float().mean()):.3f} hit_gold {hit}/{len(gold_next)} eager {t_e[-1]*1000:.1f}ms graph {t_g[-1]*1000:.1f}ms')
    if RANK == 0:
        log(f'eager mean {sum(t_e[2:])/len(t_e[2:])*1000:.1f}ms  graph mean {sum(t_g[2:])/len(t_g[2:])*1000:.1f}ms  '
            f'maxdiff_eager {md_e:.4f} maxdiff_eagerG {md_eg:.4f} maxdiff_oracle {md_o:.4f} same_oracle {same_o}/{N_GEN}')
        log('GATE PASS' if ok else 'GATE FAIL')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] GATE FAILED', flush=True)
