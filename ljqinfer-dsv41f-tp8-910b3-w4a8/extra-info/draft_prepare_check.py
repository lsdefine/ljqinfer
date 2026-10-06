import ctypes as C,json
from pathlib import Path
import torch,torch_npu
root=Path('/tmp/draft_attention_opt_v1');torch.npu.set_device(0)
f=C.CDLL(str(root/'libnorm_candidate.so')).dec_ds_prepare
P=C.c_void_p;U=C.c_uint32;f.argtypes=[P]*7+[U]*2;f.restype=C.c_int
rows=[]
with torch.inference_mode():
 for b in (1,4,2,1,3):
  receipt=torch.zeros(b,8,dtype=torch.int64,device='npu');inv=torch.linspace(.0001,1,32,device='npu')
  outs=[[torch.empty(b,6,32,2,device='npu'),torch.empty(b,6,32,2,device='npu'),torch.empty(b,6,4,device='npu'),torch.empty(b,6,dtype=torch.int64,device='npu')] for _ in range(2)]
  def run(i):
   for _ in range(i+1):assert f(P(torch.npu.current_stream().npu_stream),*[P(t.data_ptr()) for t in [receipt,inv,*outs[i]]],b,4)==0
  graphs=[]
  for i in range(2):
   s=torch.npu.Stream()
   with torch.npu.stream(s):run(i)
   s.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=s):run(i)
   graphs.append((g,s))
  for mode in ('live','inactive','error','duplicate','reused'):
   r=torch.zeros(b,8,dtype=torch.int64)
   for i in range(b):r[i]=torch.tensor([3-i,1000+i*137,i%6+1,100+i,1,0,0,0])
   if mode=='inactive':r[0,4]=0
   if mode=='error':r[0,5]=1
   if mode=='duplicate' and b>1:r[1,0]=r[0,0]
   if mode=='reused':r[:,1]=0;r[:,2]=6
   receipt.copy_(r);torch.npu.synchronize()
   for i,(g,s) in enumerate(graphs):
    for t in outs[i]:t.fill_(-17 if i else 19)
    torch.npu.synchronize()
    with torch.npu.stream(s):g.replay()
    s.synchronize()
   for x,y in zip(*outs):assert torch.equal(x.cpu(),y.cpu()),(b,mode)
   for i,row in enumerate(r.tolist()):
    valid=row[4] and not row[5] and not any(j!=i and rr[4] and rr[0]==row[0] for j,rr in enumerate(r.tolist()))
    expected=([row[3]]+[128799]*5) if valid else [-1]*6
    assert outs[0][3][i].cpu().tolist()==expected
   rows.append(dict(batch=b,mode=mode,bitwise_equal=True))
  for g,s in graphs:g.reset()
(root/'prepare_results.json').write_text(json.dumps(dict(complete=True,cases=rows),indent=2));print('PREPARE_PASS',len(rows),flush=True)
