# -*- coding: utf-8 -*-
"""Profile prefill L tokens under torchrun tp8. rank0 dumps top CUDA kernels + host-side ops."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, traceback
import torch, torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

RANK = int(os.environ.get('RANK', 0))

import torch.nn.functional as _F, traceback as _tb, collections as _co
_SEEN=_co.Counter(); _ORIG_LIN=_F.linear; _ORIG_MM=torch.matmul
def _rec(tag,a,b):
    k=(tag,tuple(a.shape[-2:]),tuple(b.shape[-2:]),str(a.dtype),str(b.dtype))
    if a.numel()>4096*4096 and (a.dtype==torch.float32 or b.dtype==torch.float32) and _SEEN[k]==0 and RANK==0:
        fr=[f'{x.filename.split("/")[-1]}:{x.lineno}:{x.name}' for x in _tb.extract_stack()[-8:-1] if 'ljqinfer' in x.filename]
        print('FP32GEMM',k,' <- ',' | '.join(fr),flush=True)
    _SEEN[k]+=1
def _lin(a,b,bias=None): _rec('linear',a,b); return _ORIG_LIN(a,b,bias)
def _mm(a,b,*r,**kw): _rec('matmul',a,b); return _ORIG_MM(a,b,*r,**kw)
_F.linear=_lin; torch.matmul=_mm
L = int(sys.argv[1]) if len(sys.argv) > 1 else 12288
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/prof_prefill'
def log(*a):
    if RANK == 0: print(*a, flush=True)
try:
    dist.init_process_group('nccl'); torch.cuda.set_device(RANK)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()
    W = wcache.load('tp8', rank=RANK, verbose=False)
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(make_args()); bind(model, W)
    dev = f'cuda:{RANK}'
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == 'cpu': m._buffers[k] = v.to(dev)
    torch.set_default_device(dev); dist.barrier()
    torch.manual_seed(0)
    toks = torch.randint(1000, 100000, (1, L), device=dev)
    def run():
        with torch.no_grad(): model(toks, start_pos=0, full_logits=False)
        torch.cuda.synchronize(); dist.barrier()
    run(); run()
    t = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        run()
    dt = time.perf_counter() - t
    log(f'PROF L={L} {dt*1000:.1f} ms (profiled)')
    if RANK == 0:
        ka = prof.key_averages()
        log(ka.table(sort_by='cuda_time_total', row_limit=45, max_name_column_width=70))
        # host-visible sync/copy counts
        syncs = [(e.key, e.count, e.cpu_time_total/1e3) for e in ka if any(s in e.key for s in ('Memcpy DtoH','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync','aten::item','aten::_local_scalar_dense','cudaEventSynchronize','aten::to','aten::copy_'))]
        log('SYNC/COPY:', syncs)
        ka = prof.key_averages(group_by_input_shape=True, group_by_stack_n=6)
        rows=[e for e in ka if e.key in ('aten::mm','aten::addmm','aten::bmm','aten::mul_','aten::linear') and e.device_time_total>15000]
        rows.sort(key=lambda e:-e.device_time_total)
        for e in rows[:14]:
            log('MM', e.key, f'{e.device_time_total/1000:.1f}ms', e.count, e.input_shapes, '\n   STACK:', ' | '.join(x for x in [x for x in (e.stack or []) if 'ljqinfer' in x][:8]))
        prof.export_chrome_trace(OUT + '.json')
        log('trace ->', OUT + '.json')
    dist.barrier(); dist.destroy_process_group()
except Exception:
    traceback.print_exc(); print(f'[rank {RANK}] PROF FAILED', flush=True)
