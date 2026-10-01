"""CUDA top-k/nucleus candidate contract; sampler is outside verify graphs."""
import torch, json
from model.sampling import sample_pairs, select_candidates
base=torch.tensor([-.9,.1,.5,1.2,-.4,.7,-1.,.3],device='cuda')
result={}
for t in (0.,.5,1.,2.):
 x=base.repeat(12000,1);before=x.clone()
 p=[sample_pairs(x[:,r*4:(r+1)*4].contiguous(),[t],r,2).cpu().tolist() for r in range(2)]
 ids=torch.tensor(select_candidates(p,[t]));observed=torch.bincount(ids,minlength=8)/len(ids)
 if t:
  w=(base.cpu()/t).softmax(0);v,ix=w.sort(descending=True);keep=v.cumsum(0)-v < .95
  expected=torch.zeros(8);expected[ix[keep]]=v[keep]/v[keep].sum()
 else:expected=torch.nn.functional.one_hot(base.cpu().argmax(),8).float()
 err=(observed-expected).abs().max().item();assert err<.02,(t,err)
 assert torch.equal(before,x);result[str(t)]={'error':err,'draws':len(ids)}
x=torch.ones(32,160,device='cuda')
p=[sample_pairs(x[:,r*40:(r+1)*40],[0,1],r,4).cpu().tolist() for r in range(4)]
assert select_candidates(p,[0,1])[:16]==[0]*16
x=torch.arange(160,device='cuda',dtype=torch.float32).repeat(1000,1)/20
p=[sample_pairs(x[:,r*40:(r+1)*40],[1],r,4).cpu().tolist() for r in range(4)]
assert min(select_candidates(p,[1]))>=140
result['mixed_ties_and_tail_exclusion']='PASS'
print(json.dumps(result))
