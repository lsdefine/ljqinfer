"""Joint DFlash compute; request-local convolution and KV mask boundaries."""
import torch
import torch.nn.functional as F
import torch.distributed as dist
import triton as tr
import triton.language as tl
from model.glm53_dflash import rms
from ops.dflash_fused import frequencies,rope_only,dflash_path_select

@tr.jit
def _grouped_batch(H,D,B,O,C:tl.constexpr,T:tl.constexpr,DS0:tl.constexpr,DS1:tl.constexpr,DS2:tl.constexpr,SIDE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);t=i//C;c=i%C;ok=i<T*C
    h0=tl.load(H+i,ok,0).to(tl.float32)
    h1=tl.load(H+i-C,ok&(t%8>0),0).to(tl.float32)
    b0=tl.load(B+SIDE*2*C+c,ok,0).to(tl.float32);b1=tl.load(B+(SIDE*2+1)*C+c,ok,0).to(tl.float32)
    d0=tl.load(D+t*DS0+c//16*DS2,ok,0).to(tl.float32);d1=tl.load(D+t*DS0+DS1+c//16*DS2,ok,0).to(tl.float32)
    w0=(b0+d0).to(H.dtype.element_ty).to(tl.float32);w1=(b1+d1).to(H.dtype.element_ty).to(tl.float32)
    p0=(h0*w0).to(H.dtype.element_ty).to(tl.float32);p1=(h1*w1).to(H.dtype.element_ty).to(tl.float32)
    tl.store(O+i,p0+p1,ok)

def grouped_batch(h,delta,base,side):
    out=torch.empty_like(h)
    _grouped_batch[(tr.cdiv(h.numel(),256),)](h,delta,base,out,h.shape[1],h.shape[0],*delta.stride(),side,256,enable_fp_fusion=False)
    return out

class JointDraft:
    def __init__(self,drafts):
        self.drafts=drafts;self.d=drafts[0];self.b=len(drafts);self.graph=None
    def forward(self):
        d=self.d;b=self.b;t=b*8
        ids=torch.cat([torch.cat((x.anchor,x.mask_ids)) for x in self.drafts])
        h=d._embedding(ids)
        positions=torch.cat([x.context+x.offsets for x in self.drafts]);freq=frequencies(positions,h.dtype)
        plans=[x._context_plan() for x in self.drafts]
        valid=torch.stack([p[1] for p in plans]);flat=torch.stack([p[0] for p in plans])
        # Slots share the same resident MTP pool, with disjoint physical page maps.
        if not all(x.pool.k.data_ptr()==d.pool.k.data_ptr() for x in self.drafts):
            raise ValueError('joint draft requires shared physical pool')
        mask=torch.cat((valid,torch.ones((b,8),device=h.device,dtype=torch.bool)),1)
        for i,layer in enumerate(d.layers):
            residual=h;h=rms(h,layer['input_norm'])
            coeff=F.linear(h,layer['attn_kernel']).reshape(t,2,2,384)
            h=grouped_batch(h,coeff[:,0],layer['attn_base'],0)
            q,k,v=F.linear(h,layer['qkv']).split((1024,128,128),-1)
            q=rope_only(rms(q.reshape(t,8,128),layer['q_norm']),freq).reshape(b,8,8,128).transpose(1,2)
            k=rope_only(rms(k.reshape(t,1,128),layer['k_norm']),freq).reshape(b,8,1,128)
            v=v.reshape(b,8,1,128)
            ck=d.pool.k[i].view(-1,1,128)[flat];cv=d.pool.v[i].view(-1,1,128)[flat]
            ck=torch.where(valid[:,:,None,None],ck,0);cv=torch.where(valid[:,:,None,None],cv,0)
            kk=torch.cat((ck,k),1).transpose(1,2);vv=torch.cat((cv,v),1).transpose(1,2)
            out=F.scaled_dot_product_attention(q,kk.expand(b,8,-1,-1),vv.expand(b,8,-1,-1),attn_mask=mask[:,None,None,:],is_causal=False)
            h=d.reduce(F.linear(out.transpose(1,2).reshape(t,1024),layer['o']))
            h=grouped_batch(h,coeff[:,1],layer['attn_base'],1)
            residual=(h+residual).bfloat16();h=rms(residual,layer['post_norm'])
            coeff=F.linear(h,layer['mlp_kernel']).reshape(t,2,2,384)
            h=grouped_batch(h,coeff[:,0],layer['mlp_base'],0)
            gate,up=F.linear(h,layer['gu']).chunk(2,-1)
            h=d.reduce(F.linear((F.silu(gate.float())*up.float()).bfloat16(),layer['down']))
            h=grouped_batch(h,coeff[:,1],layer['mlp_base'],1)
            h=(h+residual).bfloat16()
        hidden=rms(h,d.final_norm).view(b,8,-1)[:,1:].reshape(b*7,-1)
        local=F.linear(hidden,d.engine.w.lm_head)
        vals,ids=local.topk(16,-1);ids=ids+d.engine.w.vocab_start
        gv=torch.empty((8*b*7,16),device=h.device,dtype=vals.dtype)
        gi=torch.empty((8*b*7,16),device=h.device,dtype=ids.dtype)
        dist.all_gather_into_tensor(gv,vals.contiguous());dist.all_gather_into_tensor(gi,ids.contiguous())
        vals=gv.view(8,b*7,16).permute(1,0,2).reshape(b*7,128)
        ids=gi.view(8,b*7,16).permute(1,0,2).reshape(b*7,128)
        unary,which=vals.topk(16,-1);candidate=ids.gather(-1,which).view(b,7,16)
        hp=F.linear(hidden,d.selector).view(b,7,-1)
        anchors=torch.cat([x.anchor for x in self.drafts]).view(b,1,1).expand(b,1,16)
        pred_ids=torch.cat((anchors,candidate[:,:-1]),1)
        scores=unary.view(b,7,1,16)+torch.einsum('blpr,blcr,blr->blpc',d.pred[pred_ids],d.succ[candidate],hp)
        return dflash_path_select(scores,candidate)
    def prepare(self):
        if self.graph is not None:return
        stream=torch.cuda.Stream(priority=-1);stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.result=self.forward()
    def run(self,anchors,use_graph=True):
        for d,anchor in zip(self.drafts,anchors):
            if d.engine.length!=d.pool.lengths[0]:raise RuntimeError('draft length mismatch')
            d.anchor.fill_(int(anchor));d.context.fill_(d.pool.lengths[0])
        if not use_graph:return self.forward()
        self.prepare();self.graph.replay();return self.result
    def close(self):
        if self.graph is not None:self.graph.reset();self.graph=None
