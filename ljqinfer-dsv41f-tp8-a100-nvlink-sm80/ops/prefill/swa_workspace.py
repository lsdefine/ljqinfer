"""Bounded SWA workspace; BF16 tensor-core QK/PV with FP32 softmax.

Serial single-stream use; output borrowed until next call.
Runs eager: prefill chunk lengths vary per request, so a captured graph
would be rebuilt on nearly every call and the rebuild costs more than the
kernel launches it saves.
Only shape is cached; positions and KV bounds are device inputs.
"""
import torch

class Workspace:
 def __init__(self,capacity,heads,dim,device,window=128):
  self.capacity=capacity;self.h=heads;self.d=dim;self.w=window
  self.calls=0
  def alloc(shape,dtype=torch.float32):return torch.empty(shape,device=device,dtype=dtype)
  self.block=512
  self.q=alloc((capacity,heads,dim),torch.bfloat16);self.k=alloc((capacity+window-1,dim),torch.bfloat16)
  self.pos=alloc((capacity,),torch.long);self.sink=alloc((heads,))
  self.ls=alloc((),torch.long);self.last=alloc((),torch.long)
  self.rows=alloc((self.block,window,dim),torch.bfloat16);self.ix=alloc((self.block,window),torch.long)
  self.valid=alloc((self.block,window),torch.bool);self.bad=alloc((self.block,window),torch.bool)
  self.delta=torch.arange(window-1,-1,-1,device=device)
  self.score=alloc((self.block,heads,window),torch.bfloat16);self.logits=alloc((self.block,heads,window+1));self.prob=alloc(self.logits.shape)
  self.weights=alloc((self.block,heads,window),torch.bfloat16);self.out=alloc((capacity,heads,dim),torch.bfloat16)
  self.scale=dim**-.5
 def __call__(self,q,history,kv,*,start,local_start,sink,scale):
  t=len(q);old=len(history)
  if (q.device!=self.q.device or q.dtype!=torch.bfloat16 or q.shape!=(t,self.h,self.d)
      or not 0<t<=self.capacity or kv.shape!=(t,self.d) or history.shape!=(old,self.d)
      or old!=start-local_start or not 0<=old<self.w or local_start<0
      or sink.shape!=(self.h,) or scale!=self.scale
      or any(x.device!=q.device for x in (history,kv,sink))
      or kv.dtype!=q.dtype or history.dtype!=q.dtype):
   raise ValueError('invalid SWA workspace geometry/device/dtype/position')
  self.t=t
  self.q[:t].copy_(q);self.k[:old].copy_(history);self.k[old:old+t].copy_(kv)
  torch.arange(start,start+t,device=q.device,out=self.pos[:t])
  self.ls.fill_(local_start);self.last.fill_(old+t-1);self.sink.copy_(sink)
  self.run();self.calls+=1
  return self.out[:t]
 def run(self):
  for lo in range(0,self.t,self.block):
   n=min(self.block,self.t-lo);ix=self.ix[:n];valid=self.valid[:n];bad=self.bad[:n];rows=self.rows[:n]
   torch.sub(self.pos[lo:lo+n,None],self.delta,out=ix)
   torch.ge(ix,self.ls,out=valid);torch.logical_not(valid,out=bad)
   ix.sub_(self.ls).clamp_(min=0)
   torch.minimum(ix,self.last,out=ix)
   torch.index_select(self.k,0,ix.view(-1),out=rows.view(-1,self.d))
   torch.bmm(self.q[lo:lo+n],rows.transpose(1,2),out=self.score[:n])
   logits=self.logits[:n];logits[:,:,:self.w].copy_(self.score[:n]);logits[:,:,:self.w].mul_(self.scale)
   logits[:,:,:self.w].masked_fill_(bad[:,None],-torch.inf);logits[:,:,self.w].copy_(self.sink)
   torch.softmax(logits,dim=-1,out=self.prob[:n])
   self.weights[:n].copy_(self.prob[:n,:,:self.w])
   torch.bmm(self.weights[:n],rows,out=self.out[lo:lo+n])
  return self.out

