"""Local test only: independent Torch oracle for four compressed-weight kernels."""
import gc,json,sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from ops.moe_common import Workspace
from ops.moe_decode import decode_int4,decode_int4_fp8
from ops.moe_prefill import prefill_int4,prefill_int4_fp8


def packed(e,n,k,g,device):
    q=torch.randint(-8,8,(e,n,k),device=device,dtype=torch.int8)
    s=(torch.rand((e,n,k//g),device=device)*0.018+0.002).half()
    p=((q[:,:,::2].to(torch.uint8)&15)|((q[:,:,1::2].to(torch.uint8)&15)<<4))
    return p,s


def deq4(p,s,g):
    # Independent library oracle, not ops.moe_common helpers.
    q=torch.stack((p.int()%16,p.int()//16),dim=-1).reshape(*p.shape[:-1],p.shape[-1]*2)
    q=torch.where(q>=8,q-16,q).float()
    return (q*s.float().repeat_interleave(g,-1)).bfloat16()


def oracle(x,ids,rw,gu,gs,d,ds,g,fp8):
    gw=deq4(gu,gs,g);i=gw.shape[1]//2;e=gw.shape[0]
    if fp8:
        dw=(d.view(torch.float8_e4m3fn).float()*ds.repeat_interleave(128,1).repeat_interleave(128,2)[:,:d.shape[1],:d.shape[2]]).bfloat16()
    else:dw=deq4(d,ds,g)
    out=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
    cpu_ids=ids.cpu().tolist()
    for t,row in enumerate(cpu_ids):
        for k,ex in enumerate(row):
            if not 0<=ex<e:continue
            v=(x[t].float()@gw[ex].float().T).bfloat16().float()
            a=(F.silu(v[:i])*v[i:]).bfloat16()
            out[t].add_((a.float()@dw[ex].float().T)*rw[t,k])
    return out


def compare(y,ref):
    assert torch.isfinite(y).all()
    diff=(y-ref).norm()/ref.norm().clamp_min(1e-20)
    maxnorm=(y-ref).abs().max()/ref.abs().max().clamp_min(1e-20)
    assert diff<0.012 and maxnorm<0.035,(float(diff),float(maxnorm))
    return dict(rel_l2=float(diff),max_abs=float((y-ref).abs().max()),max_normalized=float(maxnorm))


def case(shape,g,fp8,seed,graph):
    t,h,i,e,k=shape;device='cuda:7';torch.manual_seed(seed)
    x=torch.randn((t,h),device=device,dtype=torch.bfloat16)*0.5
    ids=torch.randint(e,(t,k),device=device,dtype=torch.int32)
    ids[0,0]=-1
    if k>1:ids[-1,-1]=e+2
    if t>1:ids[1].fill_(min(e-1,2))
    rw=torch.randn((t,k),device=device).float()*0.3
    gu,gs=packed(e,2*i,h,g,device)
    if fp8:
        d=(torch.randn((e,h,i),device=device)*4).to(torch.float8_e4m3fn).view(torch.uint8)
        ds=torch.rand((e,(h+127)//128,(i+127)//128),device=device)*0.015+0.003
    else:d,ds=packed(e,h,i,g,device)
    ws=Workspace.create(t,k,e,h,i,device);y=torch.empty((t,h),device=device)
    ws.hidden.fill_(float('nan'));ws.partial.fill_(float('nan'))
    ref=oracle(x,ids,rw,gu,gs,d,ds,g,fp8)
    originals=(x.clone(),ids.clone(),rw.clone())
    results=[]
    for name,fn in [('decode',decode_int4_fp8 if fp8 else decode_int4),('prefill',prefill_int4_fp8 if fp8 else prefill_int4)]:
        x.copy_(originals[0]);ids.copy_(originals[1]);rw.copy_(originals[2])
        ref=oracle(x,ids,rw,gu,gs,d,ds,g,fp8)
        assert ref.norm()>0, 'Vacuous initial oracle'
        fn(x,ids,rw,gu,gs,d,ds,y,ws,g)
        torch.cuda.synchronize();m=compare(y,ref)
        rec=dict(shape=shape,group=g,fp8=fp8,seed=seed,phase=name,eager=m)
        if graph:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):fn(x,ids,rw,gu,gs,d,ds,y,ws,g)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            cg=torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):fn(x,ids,rw,gu,gs,d,ds,y,ws,g)
            cg.replay();torch.cuda.synchronize();rec['graph']=compare(y,ref)
            # Replay with changed data/routing and all-invalid row: no stale slots.
            x.mul_(-0.7);ids.copy_(torch.randint(e,(t,k),device=device,dtype=torch.int32));ids[0].fill_(-1)
            rw.mul_(-0.5)
            ref2=oracle(x,ids,rw,gu,gs,d,ds,g,fp8)
            cg.replay();torch.cuda.synchronize();rec['graph_changed']=compare(y,ref2)
            assert torch.equal(y[0],torch.zeros_like(y[0]))
            del cg
            ref=ref2
        print(json.dumps(rec),flush=True);results.append(rec)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.cuda.set_device(7)
    torch.cuda.set_per_process_memory_fraction(2/80)
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    shapes=[(3,256,128,5,3)] if args.smoke else [(1,256,128,5,3),(5,192,192,7,3),(33,256,128,5,3),(3,256,128,256,8),(2,6144,256,8,8),(17,6144,256,8,8)]
    for shape in shapes:
        for g in ([64] if shape[1]%128 else [64,128]):
            for fp8 in [False,True]:
                records+=case(shape,g,fp8,7300+shape[0],not args.smoke)
                gc.collect();torch.cuda.empty_cache()
    report=dict(status='passed',records=records,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(records),'peak',report['peak_reserved_MiB'],flush=True)


if __name__=='__main__':main()
