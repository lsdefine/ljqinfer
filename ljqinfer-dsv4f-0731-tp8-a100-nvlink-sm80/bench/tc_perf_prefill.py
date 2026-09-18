# -*- coding: utf-8 -*-
"""Speed gate: prefill latency at L tokens (default 8192), NCCL TP8 multi-process
(same load path as tc_oracle_decode_dist.py = production form).
torchrun --nproc_per_node=8 tc_perf_prefill.py [L] [reps]  -> rank0 prints PREFILL L=... ms and tok/s
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, sys, time, traceback
import torch
import torch.distributed as dist

RANK = int(os.environ.get('RANK', 0))
t0 = time.time()
def log(*a):
    if RANK == 0:
        print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

try:
    dist.init_process_group('nccl')
    torch.cuda.set_device(RANK)

    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()

    W = wcache.load('tp8', rank=RANK, verbose=(RANK == 0))
    log('weights loaded')

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
    if RANK==0:
        from model.arch import Compressor
        cnt=0;ok=0
        for n,m in model.named_modules():
            if isinstance(m, Compressor):
                for a in ('wkv','wgate'):
                    w=getattr(m,a).weight; cnt+=1; ok+=int(w.dtype==torch.float32 and bool((w==w.bfloat16().float()).all()))
        log('BF16EXACT compressors', cnt, 'exact', ok)
    log('model ready')

    torch.manual_seed(0)
    toks = torch.randint(1000, 100000, (1, L), device=dev)  # same seed -> identical on all ranks

    def run():
        with torch.no_grad():
            model(toks, start_pos=0, full_logits=False)
        torch.cuda.synchronize()
        dist.barrier()

    run()
    log('warmup done')
    times = []
    for r in range(REPS):
        t = time.perf_counter()
        run()
        dt = time.perf_counter() - t
        times.append(dt)
        log(f'rep {r}: {dt*1000:.1f} ms')
    best = min(times)
    log(f'PREFILL L={L} best={best*1000:.1f} ms  mean={sum(times)/len(times)*1000:.1f} ms  {L/best:.0f} tok/s  (torchrun tp8)')
    log(f'peak mem rank0 {torch.cuda.max_memory_allocated(RANK)/2**30:.1f} GiB')
    dist.barrier()
    dist.destroy_process_group()
except Exception:
    traceback.print_exc()
    print(f'[rank {RANK}] PERF PREFILL FAILED', flush=True)
