"""8192-token leaf A/B; synthetic tensors, no model construction or engine.
Single rank first (compiles extensions), then torch.distributed.run --nproc_per_node=8.
TP8 uses production PrefillParallel; results are NOT full encoder chunk latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import types

import torch
import torch_npu
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--reference', default='c66d283')
p.add_argument('--rounds', type=int, default=30)
p.add_argument('--rows', type=int, nargs='+', default=[1, 128, 2047, 2048, 2049, 8192])
p.add_argument('--seeds', type=int, nargs='+', default=[4201, 4202])
a = p.parse_args()
rank = int(os.environ.get('LOCAL_RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 1))
torch.npu.set_device(rank)
if world > 1:
    dist.init_process_group('hccl')
from ops.prefill import attention
from ops.prefill.native import ops
from test_prefill_leaf_ab import exact

output = Path(a.output)
output.mkdir(parents=True, exist_ok=True)
def git_source(path):
    return subprocess.check_output(['git', 'show', f'{a.reference}:{path}'], cwd=ROOT)

reference = types.ModuleType('ops.prefill.chunk_reference')
reference.__package__ = 'ops.prefill'
exec(compile(git_source('ops/prefill/attention.py'), 'frozen_attention', 'exec'), reference.__dict__)
source = output / 'native_reference'
if rank == 0:
    source.mkdir(exist_ok=True)
    for name in ('prefill_torch.cpp', 'prefill_attention_torch.cpp', 'prefill_cann_torch.cpp', 'prefill_native.h'):
        text = git_source('ops/kernels/' + name).decode().replace('TORCH_LIBRARY(ljq_prefill,', 'TORCH_LIBRARY(ljq_chunk_reference,').replace('TORCH_LIBRARY_IMPL(ljq_prefill,', 'TORCH_LIBRARY_IMPL(ljq_chunk_reference,').replace('TORCH_LIBRARY_FRAGMENT(ljq_prefill,', 'TORCH_LIBRARY_FRAGMENT(ljq_chunk_reference,')
        (source / name).write_text(text)
if world > 1:
    dist.barrier()
from torch.utils.cpp_extension import load
npu = Path(torch_npu.__file__).parent
cann = Path(os.environ['ASCEND_HOME_PATH'])
kernels = ROOT / 'ops/kernels'
build = output / 'build_reference'
build.mkdir(exist_ok=True)
load(build_directory=str(build), name='ljq_chunk_reference', sources=[str(source / n) for n in ('prefill_torch.cpp', 'prefill_attention_torch.cpp', 'prefill_cann_torch.cpp')],
     extra_include_paths=[str(npu/'include'), str(cann/'include')], extra_cflags=['-O2', '-std=c++17'],
     extra_ldflags=[str(kernels/n) for n in ('libhc.so', 'libattention.so', 'libdq.so')] + [f'-L{npu}/lib', '-ltorch_npu', f'-Wl,-rpath,{npu}/lib', f'-Wl,-rpath,{kernels}', f'-L{cann}/lib64', '-lopapi', '-lnnopbase', f'-Wl,-rpath,{cann}/lib64'], is_python_module=False)
old_ops = torch.ops.ljq_chunk_reference
if world == 8:
    from model.prefill_build import PrefillParallel
    par = PrefillParallel()
else:
    assert world == 1
    par = None

class Comm:
    def __init__(self):
        self.record = False
        self.values = []
    def sum(self, x):
        if par is not None:
            par.sum(x)
        if self.record:
            self.values.append(x.clone())

class Scratch:
    tile, group = 32, 128
    def __init__(self, width):
        self.bank = torch.empty(width, 128, device='npu', dtype=torch.bfloat16)
        self.dot = torch.empty(self.tile*4*width, device='npu')
        self.score = torch.empty(self.group*width, device='npu')
        self.values = torch.empty(self.group*512, device='npu')
        self.ids = torch.empty(self.group*512, device='npu', dtype=torch.int64)
    def views(self, rows, heads, width, topk):
        n = min(rows, self.tile)
        return (self.dot[:n*heads*width].view(n*heads, width), self.score[:rows*width].view(rows, width), self.values[:rows*topk].view(rows, topk), self.ids[:rows*topk].view(rows, topk))

report = dict(reference=a.reference, world=world, rank=rank, scope='synthetic leaves; NOT encoder chunk latency', rounds=a.rounds, tests=[], status='RUNNING')
report['candidate_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('ops/prefill/attention.py', 'ops/kernels/prefill_torch.cpp')}
def save():
    (output/f'rank{rank}.json').write_text(json.dumps(report, indent=2))
def bench(name, rows, seed, old, new):
    for _ in range(4):
        old(); new()
    torch.npu.synchronize()
    retries = torch.npu.memory_stats().get('num_alloc_retries', 0)
    wall, event, peak = [[], []], [[], []], [0, 0]
    for i in range(a.rounds):
        for v in (i % 2, 1-i % 2):
            if world > 1:
                dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            allocated = torch.npu.memory_allocated()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            t = time.perf_counter()
            start.record()
            out = (old, new)[v]()
            end.record(); end.synchronize()
            wall[v].append((time.perf_counter()-t)*1000)
            event[v].append(start.elapsed_time(end))
            peak[v] = max(peak[v], torch.npu.max_memory_allocated()-allocated)
            del out
    result = dict(op=name, rows=rows, seed=seed, exact=True, wall_samples_ms=wall, event_samples_ms=event, peak_extra_bytes=peak, allocation_retries=torch.npu.memory_stats().get('num_alloc_retries', 0)-retries)
    result['wall_ms'] = [statistics.median(x) for x in wall]
    result['speedup'] = result['wall_ms'][0]/result['wall_ms'][1]
    report['tests'].append(result); save()
    if rank == 0:
        print(json.dumps({k:v for k,v in result.items() if 'samples' not in k}), flush=True)

try:
    with torch.inference_mode():
        for seed in a.seeds:
            for rows in a.rows:
                torch.manual_seed(seed+rank)
                q = torch.randn(rows, 4, 128, device='npu', dtype=torch.bfloat16)
                weight = torch.rand(rows, 4, device='npu')
                torch.manual_seed(seed)
                keys = torch.randn(4096, 128, device='npu', dtype=torch.bfloat16)
                pos = torch.arange(rows, device='npu', dtype=torch.int64)+max(0,8192-rows)
                valid = torch.tensor([4096], device='npu', dtype=torch.int64)
                scratch, comm = Scratch(4096), Comm()
                inputs = (q, weight, keys, pos, valid)
                copies = tuple(x.clone() for x in inputs)
                kw = dict(ratio=2, total_heads=32, parallel=comm, workspace=scratch, topk=512)
                old = lambda: reference.select(*inputs, **kw)
                new = lambda: attention.select(*inputs, **kw)
                comm.record = True
                before = old(); trace = comm.values
                comm.values = []
                after = new()
                exact(before, after)
                assert len(trace) == len(comm.values) == (rows+31)//32
                for x,y in zip(trace, comm.values):
                    # Scores legitimately contain -inf in masked columns.
                    assert x.shape == y.shape and torch.equal(x.cpu().contiguous().view(torch.uint8), y.cpu().contiguous().view(torch.uint8))
                comm.record = False; comm.values = []
                del trace, before, after
                exact(new(), old()); exact(inputs, copies)
                bench('index_select', rows, seed, old, new)
                exact(inputs, copies)
                del copies, inputs, q, weight, keys, scratch, old, new
                h = torch.randn(rows,4,5120,device='npu',dtype=torch.bfloat16)
                y = torch.randn(rows,5120,device='npu',dtype=torch.bfloat16)
                post = torch.rand(rows,4,device='npu')
                mix = torch.randn(rows,4,4,device='npu').softmax(-1)
                inputs = (y,h,post,mix)
                copies = tuple(x.clone() for x in inputs)
                old = lambda: old_ops.hc_expand(*inputs)
                new = lambda: ops.hc_expand(*inputs)
                exact(old(),new()); exact(new(),old())
                bench('hc_expand',rows,seed,old,new)
                exact(inputs,copies)
                del inputs,copies,h,y,post,mix,old,new
        report['status'] = 'PASS'; save()
finally:
    if par is not None:
        par.close()
    if world > 1:
        dist.destroy_process_group()
