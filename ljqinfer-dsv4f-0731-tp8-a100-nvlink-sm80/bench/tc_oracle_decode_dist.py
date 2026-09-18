# -*- coding: utf-8 -*-
"""NCCL TP8 oracle greedy decode (torchrun, 8 ranks).
Same TF-verified forward path as tc_oracle_decode.py, but each rank holds only
its weight shard; comms via NCCL (all_reduce / all_gather).
torchrun --nproc_per_node=8 tc_oracle_decode_dist.py
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

N_GEN = 128  # generous cap; early-stop once the DSML tool-call block closes

try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)

    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()

    ids = json.load(open('/tmp/c1_ids.json'))
    log(f'prompt {len(ids)} tok; NCCL TP8 greedy decode {N_GEN} via repeated prefill')

    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    log('weights loaded')

    torch.set_default_dtype(torch.bfloat16)
    args = make_args()
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
    dist.barrier()
    log('model ready')

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file('/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json')
    CLOSE_MARK = '</｜DSML｜tool_calls>'  # closing tag of the tool-call block (verified from c1 prompt)

    cur = list(ids)
    gen = []
    closed = False
    for step in range(N_GEN):
        toks = torch.tensor([cur], device=dev)
        out_ids, logits, _ = model(toks, start_pos=0, full_logits=True)
        lg = logits[0] if logits.dim() == 3 else logits
        # logits are all_gather'ed -> identical on every rank; argmax is deterministic
        nxt = int(lg[-1].float().argmax())
        gen.append(nxt)
        cur.append(nxt)
        txt = tk.decode(gen, skip_special_tokens=False)
        log(f'step {step}: id={nxt} tail={repr(txt[-40:])}')
        if CLOSE_MARK in txt:
            closed = True
            break

    if RANK == 0:
        log('GEN ids:', gen)
        txt = tk.decode(gen, skip_special_tokens=False)
        log('GEN text:', repr(txt))
        gold = json.load(open('/tmp/gold_c1_out.json'))
        n = min(len(gold), len(gen))
        log(f'match_gold_prefix({n}):', gen[:n] == gold[:n])
        log('DSML_CLOSED:', closed)
        log('DECODE DONE' if closed else 'DECODE DONE (no close within cap)')
    dist.barrier()
    dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] DECODE FAILED', flush=True)
