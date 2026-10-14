# Run with torchrun --nproc_per_node=8 scripts/test_top_p_sampling.py on idle TP8.
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,torch_npu,torch.distributed as dist
from model.prefill_build import PrefillParallel
from ops.decode.sampling import TopPSampler
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);dist.init_process_group('hccl')
p=PrefillParallel();p._call('HcclSetConfig',0,0)
a=lambda sh,dt=torch.float32:torch.empty(sh,device='npu',dtype=dt)
src=a((6,16160));local=a(src.shape);temp=a((1,));seed=a((1,),torch.int32);bias=a((1,6));out=a((6,),torch.int64)
sampler=TopPSampler(1,a,p,temp,seed,bias,out)
base=torch.full((129280,),-100.);tokens=torch.tensor([2,16165,65000,128822,129279]);base[tokens]=torch.tensor([.50,.28,.16,.04,.02]).log()
src.copy_(base[rank*16160:(rank+1)*16160].to('npu').expand(6,-1));temp.fill_(1);seed.fill_(123);bias.fill_(.7)
def run():
 local.copy_(src);sampler.sample(local)
run();torch.npu.synchronize();g=torch.npu.NPUGraph()
with torch.npu.graph(g):run()
samples=a((1000,6),torch.int64);results=[]
for temperature in [0.,.5,1.,2.]:
 temp.fill_(temperature);seed.fill_(123)
 for i in range(1000):g.replay();samples[i].copy_(out)
 torch.npu.synchronize();draws=samples.cpu().flatten();truth=base.clone();truth[128822]+=.7
 if temperature==0:
  assert (draws==truth.argmax()).all();row={'temperature':0.,'greedy':int(draws[0])}
 else:
  values,ids=torch.sort(truth/temperature);probs=torch.softmax(values,0);remove=probs.cumsum(0)<=.05;remove[-1]=False;probs[remove]=0;probs/=probs.sum()
  expected=torch.zeros_like(probs).scatter_(0,ids,probs);observed=torch.bincount(draws,minlength=129280).float()/draws.numel()
  assert not bool((expected[draws]==0).any()),'excluded tail sampled'
  err=float((expected-observed).abs().max());assert err<.025,(temperature,err)
  row={'temperature':temperature,'draws':draws.numel(),'max_abs_frequency_error':err,'expected':expected[tokens].tolist(),'observed':observed[tokens].tolist()}
 allout=[torch.empty_like(samples) for _ in range(8)];dist.all_gather(allout,samples);assert all(torch.equal(samples,x) for x in allout)
 results.append(row)
 if rank==0:print(json.dumps(row),flush=True)
# Explicit greedy tie must choose lowest token ID across shards.
src.fill_(-10)
for token in [2,16165,129279]:
 if token//16160==rank:src[:,token%16160]=5
bias.zero_();temp.zero_();g.replay();torch.npu.synchronize();assert out.cpu().tolist()==[2]*6
if rank==0:
 print('ALL_PASS',flush=True)
g.reset();p.close();dist.destroy_process_group()
