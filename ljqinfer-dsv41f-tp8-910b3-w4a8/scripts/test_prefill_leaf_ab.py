"""Standalone frozen-commit A/B. No model construction or checkpoint loading.
PYTHONPATH=. python scripts/test_prefill_leaf_ab.py --output /tmp/leaf-ab.json
Reference Python is read from git, never from the modified working tree.
Unmodified native kernels are shared; this is composition parity, not an
independent mathematical oracle or TP8/end-to-end acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch
import torch_npu
from ops.prefill import residual, hc, moe_units, attention, attention_units, engram
from ops.prefill.native import ops

ROOT = Path(__file__).resolve().parents[1]


def frozen(revision):
    package = types.ModuleType('_leaf_reference')
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + '.native'] = sys.modules['ops.prefill.native']
    hashes = {}
    for name in ('residual', 'hc', 'moe_units', 'attention', 'attention_units', 'engram'):
        path = f'ops/prefill/{name}.py'
        source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)
        hashes[path] = hashlib.sha256(source).hexdigest()
        module = types.ModuleType(package.__name__ + '.' + name)
        module.__package__ = package.__name__
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
    return package, hashes


def exact(a, b):
    if a is None or b is None:
        assert a is b
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
        assert torch.equal(a, b), (a.float() - b.float()).abs().max().item()
        assert torch.equal(a.detach().cpu().contiguous().view(torch.uint8),
                           b.detach().cpu().contiguous().view(torch.uint8)), 'bit mismatch'


class LocalComm:
    """Single-device communication fixture, explicitly NOT a TP performance test."""
    world = 1
    rank = 0

    def sum(self, x):
        return x

    def scatter(self, x, *, out):
        out.copy_(x)

    def logits(self, x, *, out):
        out.copy_(x)


class EngramScratch:
    def __init__(self, rows):
        self.buffers = (torch.empty(rows, 25600, device='npu'),
                        torch.empty(rows, 25600, device='npu'),
                        torch.empty(rows, 25600, device='npu', dtype=torch.bfloat16),
                        torch.empty(rows, 4, 5120, device='npu', dtype=torch.bfloat16),
                        torch.empty(rows, 4, 5120, device='npu', dtype=torch.bfloat16))

    def views(self, h):
        return self.buffers


def main(args):
    torch.npu.set_device(args.device)
    torch.manual_seed(4201)
    ref, hashes = frozen(args.reference)
    report = dict(reference=args.reference, reference_sha256=hashes,
                  scope='single NPU; synthetic weights; no engine; LocalComm world=1',
                  rounds=args.rounds, tests=[], status='RUNNING',
                  candidate_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                    for p in hashes},
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2))

    def bench(name, rows, old, new, readonly=()):
        snapshots = [x.clone() for x in readonly]
        a = old()
        for x, y in zip(readonly, snapshots):
            exact(x, y)
        b = new()
        exact(a, b)
        for _ in range(2):
            exact(a, old())
            exact(b, new())
        for x, y in zip(readonly, snapshots):
            exact(x, y)
        del a, b, snapshots
        for _ in range(4):
            old(); new()
        torch.npu.synchronize()
        wall, events = [[], []], [[], []]
        for i in range(args.rounds):
            for version in ((0, 1) if i % 2 == 0 else (1, 0)):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                torch.npu.synchronize()
                t = time.perf_counter()
                start.record()
                value = (old, new)[version]()
                end.record()
                end.synchronize()
                wall[version].append((time.perf_counter() - t)*1000)
                events[version].append(start.elapsed_time(end))
                del value
        peaks = []
        for f in (old, new):
            torch.npu.synchronize()
            base = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            value = f()
            torch.npu.synchronize()
            peaks.append(torch.npu.max_memory_allocated() - base)
            del value
        wm = [statistics.median(v) for v in wall]
        em = [statistics.median(v) for v in events]
        item = dict(op=name, rows=rows, exact=True, input_readonly=True,
                    wall_ms=wm, event_ms=em, speedup=wm[0]/wm[1],
                    peak_extra_bytes=peaks, wall_samples=wall, event_samples=events)
        report['tests'].append(item)
        save()
        print(json.dumps({k:v for k,v in item.items() if not k.endswith('samples')}), flush=True)

    rand = lambda *shape: torch.randn(*shape, device='npu', dtype=torch.bfloat16)
    comm = LocalComm()
    with torch.inference_mode():
        for n in args.rows:
            h = rand(n, 4, 5120)
            incoming = torch.rand(n, 4, device='npu')
            fn = torch.randn(24, 20480, device='npu')*.002
            scale = torch.ones(3, device='npu')
            base = torch.zeros(24, device='npu')
            norm = rand(5120)
            kwargs = dict(norm_eps=1e-6, hc_eps=1e-6, iters=20)
            bench('hc_prepare', n,
                  lambda: ref.hc.hc_prepare(h,incoming,fn,scale,base,norm,**kwargs),
                  lambda: hc.hc_prepare(h,incoming,fn,scale,base,norm,**kwargs),
                  (h,incoming,fn,scale,base,norm))
            x, _, post, comb = hc.hc_prepare(h,incoming,fn,scale,base,norm,**kwargs)
            bench('hc_finish', n, lambda: ref.hc.hc_finish(x,h,post,comb),
                  lambda: hc.hc_finish(x,h,post,comb), (x,h,post,comb))
            ids = torch.arange(n*6, device='npu').reshape(n,6).remainder(7)
            bench('dispatch_quant',n,lambda:ref.moe_units.dispatch_quant(x,ids,experts=16),
                  lambda:moe_units.dispatch_quant(x,ids,experts=16),(x,ids))
            hidden = rand(n*6,576)
            prob = torch.rand(n*6,device='npu')
            for pad in (288,320):
                bench(f'activation_quant_{pad}',n,
                      lambda:ref.moe_units.activation_quant(hidden,prob,padded_dim=pad),
                      lambda:moe_units.activation_quant(hidden,prob,padded_dim=pad),(hidden,prob))
            order = ids.flatten().float().argsort(stable=True)
            values = rand(n*6,5120)
            bench('combine',n,lambda:ref.moe_units.combine(values,order),
                  lambda:moe_units.combine(values,order),(values,order))
            # BF16 and FP32 protect the .float() aliasing boundary.
            for dtype in (torch.bfloat16,torch.float32):
                inp = x.to(dtype)
                bench(f'rms_{dtype}',n,lambda:ref.residual.rms(inp,norm),
                      lambda:residual.rms(inp,norm),(inp,norm))
            prepared = rand(n,64)
            project_weight = rand(64,25600)*.02
            gate_weight = rand(4,5120).float()
            rotation = torch.eye(32,device='npu',dtype=torch.float32)
            scratch_a, scratch_b = EngramScratch(n), EngramScratch(n)
            ek = dict(project=lambda z:(z@project_weight).float(), weight=gate_weight,
                      rotation=rotation,eps=1e-6,comm=comm)
            bench('engram_apply_world1',n,
                  lambda:ref.engram.engram_apply(h,prepared,workspace=scratch_a,**ek),
                  lambda:engram.engram_apply(h,prepared,workspace=scratch_b,**ek),
                  (h,prepared,project_weight,gate_weight,rotation))
            # Real matmuls, small synthetic projections; no checkpoint or engine.
            ax = rand(n,256)
            weights = {key:rand(a,b)*.02 for key,a,b in (
                ('wq_a',256,256),('wq_b',256,512),('wkv',256,512),
                ('i_wq_b',256,128),('i_weights',256,1),('c_wkv',256,512),
                ('c_wgate',256,512),('i_wk',512,128),
                ('wo_a',512,128),('wo_b',128,256))}
            linear = lambda key,z:z@weights[key]
            freqs = torch.stack((torch.ones(n,32,device='npu'),torch.zeros(n,32,device='npu')),dim=-1)
            norms = {'q':rand(256),'kv':rand(512)}
            ak = dict(linear=linear,norms=norms,eps=1e-6,heads=1,head_dim=512,
                      index_heads=1,index_dim=128,needs_index=True)
            bench('attention_prepare',n,
                  lambda:ref.attention_units.attention_prepare(ax,freqs,**ak),
                  lambda:attention_units.attention_prepare(ax,freqs,**ak),(ax,*weights.values(),*norms.values()))
            out = rand(n,1,512)
            fk = dict(linear=linear,comm=comm,dtype=torch.bfloat16)
            bench('attention_finish',n,
                  lambda:ref.attention_units.attention_finish(out,freqs,**fk),
                  lambda:attention_units.attention_finish(out,freqs,**fk),(out,*weights.values()))
            cv,cs = torch.zeros(4,512,device='npu'),torch.zeros(4,512,device='npu')
            nw,iw = rand(512),rand(128)
            freq_table = torch.stack((torch.ones(n+2,32,device='npu'),torch.zeros(n+2,32,device='npu')),dim=-1)
            for ratio in (1,2):
                def source(module):
                    v,s = cv.clone(),cs.clone()
                    result = module.source_append(ax,linear=linear,norm_weight=nw,
                        index_norm_weight=iw,eps=1e-6,ratio=ratio,carry_kv=v,carry_score=s,
                        start=0,frequencies=lambda pos:freq_table.index_select(0,pos))
                    return result,v,s
                bench(f'source_append_r{ratio}',n,
                      lambda:source(ref.attention_units),lambda:source(attention_units),
                      (ax,nw,iw,*weights.values()))
            del scratch_a,scratch_b,h,hidden,values
    report['status'] = 'PASS'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--reference',default='57f8f5f')
    parser.add_argument('--rows',type=int,nargs='+',default=[1,127,513,2048])
    parser.add_argument('--rounds',type=int,default=30)
    parser.add_argument('--output',default='/tmp/prefill_leaf_ab.json')
    main(parser.parse_args())
