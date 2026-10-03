"""GLM-5.3 DFlash2 TP8 sidecar; committed target features are the only KV source.
Adapted from the GPU140 Qwen DFlash2 execution contract (not native GLM MTP).
"""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from ops.dflash_glm53 import grouped
from .glm53_mtp_pool import TARGET_LAYERS

DEFAULT_DIR = '/mnt/data2/kw/GLM-5.3-DFlash2'


def rms(x, weight, eps=1e-5):
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)*weight.float()).to(x.dtype)


def rope(x, positions, theta=1000000.):
    inv=theta**(-torch.arange(0,128,2,device=x.device,dtype=torch.float32)/128)
    angles=positions.float()[:,None]*inv[None,:]
    angles=torch.cat((angles,angles),-1)[:,None,:]
    c,s=angles.cos().to(x.dtype),angles.sin().to(x.dtype)
    rotate=torch.cat((-x[...,64:],x[...,:64]),-1)
    return (x*c+rotate*s).to(x.dtype)


from ops.dflash_fused import rope_only, dflash_path_select, frequencies

class DFlash2:
    def __init__(self, engine, model_dir=DEFAULT_DIR):
        self.engine=engine;self.device=engine.device;self.rank=engine.rank
        self.pool=engine.mtp_kv;self.graph=None;self.append_graphs=None
        self.config=json.loads((Path(model_dir)/'config.json').read_text())
        cfg=self.config;dc=cfg['dflash_config']
        expected={'hidden_size':6144,'intermediate_size':12288,'num_hidden_layers':6,
                  'num_attention_heads':64,'num_key_value_heads':8,'head_dim':128,
                  'sliding_window':2048,'rms_norm_eps':1e-5}
        for k,v in expected.items():
            if cfg[k]!=v:raise ValueError(f'DFlash {k}: {cfg[k]} != {v}')
        if tuple(dc['target_layer_ids'])!=TARGET_LAYERS or dc['block_size']!=8:
            raise ValueError('DFlash feature/Q contract mismatch')
        if dc['conv_group_size']!=16 or dc['conv_kernel_size']!=2 or dc['selector_top_k']!=16:
            raise ValueError('DFlash convolution/selector contract mismatch')
        self.mask_id=dc['mask_token_id'];self.eos=set(cfg['eos_token_id'])
        self.feature_start=self.rank*768;self.feature_end=self.feature_start+768
        with safe_open(str(Path(model_dir)/'model.safetensors'),framework='pt',device='cpu') as store:
            def get(name,rows=None,cols=None):
                t=store.get_tensor(name)
                if rows is not None:t=t[rows]
                if cols is not None:t=t[:,cols]
                return t.contiguous().to(self.device)
            fc=store.get_tensor('fc.weight')
            self.fc=torch.cat([fc[:,i*6144+self.feature_start:i*6144+self.feature_end] for i in range(6)],1).contiguous().to(self.device)
            self.hidden_norm=get('hidden_norm.weight');self.final_norm=get('norm.weight')
            self.pred=get('candidate_selector.predecessor_codebook')
            self.succ=get('candidate_selector.successor_codebook')
            self.selector=get('candidate_selector.hidden_projection.weight')
            self.layers=[]
            for i in range(6):
                p=f'layers.{i}.';q=slice(self.rank*1024,(self.rank+1)*1024)
                kv=slice(self.rank*128,(self.rank+1)*128);mlp=slice(self.rank*1536,(self.rank+1)*1536)
                self.layers.append(dict(
                    input_norm=get(p+'input_layernorm.weight'),post_norm=get(p+'post_attention_layernorm.weight'),
                    qkv=torch.cat([get(p+'self_attn.'+n+'_proj.weight',rows=z) for n,z in [('q',q),('k',kv),('v',kv)]],0),
                    o=get(p+'self_attn.o_proj.weight',cols=q),
                    q_norm=get(p+'self_attn.q_norm.weight'),k_norm=get(p+'self_attn.k_norm.weight'),
                    gu=torch.cat([get(p+'mlp.'+n+'_proj.weight',rows=mlp) for n in ['gate','up']],0),
                    down=get(p+'mlp.down_proj.weight',cols=mlp),
                    attn_base=get(p+'attention_conv.base_kernel'),attn_kernel=get(p+'attention_conv.kernel_projection.weight'),
                    mlp_base=get(p+'mlp_conv.base_kernel'),mlp_kernel=get(p+'mlp_conv.kernel_projection.weight')))
        self.anchor=torch.zeros(1,device=self.device,dtype=torch.long)
        self.context=torch.zeros(1,device=self.device,dtype=torch.long)
        self.offsets=torch.arange(8,device=self.device)
        self.context_indices=torch.arange(2040,device=self.device)
        self.mask_ids=torch.full((7,),self.mask_id,device=self.device,dtype=torch.long)

    def reduce(self,x):
        dist.all_reduce(x);return x

    @torch.inference_mode()
    def append(self,result):
        n=result.features[0].shape[0];start=result.start
        if self.append_graphs is not None and 1 <= n <= 8:
            self.append_graphs(result)
            return
        if start!=self.pool.lengths[0]:raise RuntimeError('target/draft committed context diverged')
        local=torch.cat([x[:,self.feature_start:self.feature_end].bfloat16() for x in result.features],-1)
        h=rms(self.reduce(F.linear(local,self.fc)),self.hidden_norm)
        positions=torch.arange(start,start+n,device=self.device)
        freq=frequencies(positions,h.dtype)
        keys=[];values=[]
        for layer in self.layers:
            k,v=F.linear(h,layer['qkv'][1024:]).split(128,-1)
            k=rope_only(rms(k.view(n,1,128),layer['k_norm']),freq)
            keys.append(k);values.append(v.view(n,1,128))
        self.pool.append(0,start,keys,values)

    def _embedding(self,ids):
        w=self.engine.w
        valid=(ids>=w.vocab_start)&(ids<w.vocab_end)
        local=(ids-w.vocab_start).clamp(0,w.vocab_end-w.vocab_start-1)
        return self.reduce(w.embed[local].bfloat16()*valid[:,None])

    def _context_plan(self):
        count=self.context.clamp(0,2040)
        absolute=self.context-count+self.context_indices-(2040-count)
        valid=absolute>=self.context-count
        safe=absolute.clamp(min=0)
        physical=self.pool.page_table[0,safe//self.pool.page_size].clamp(min=0)
        flat=physical*self.pool.page_size+safe%self.pool.page_size
        return flat,valid

    def _attention(self,h,positions,layer,layer_idx,plan):
        q,k,v=F.linear(h,layer['qkv']).split((1024,128,128),-1)
        q=rope_only(rms(q.reshape(8,8,128),layer['q_norm']),self._freq)
        k=rope_only(rms(k.reshape(8,1,128),layer['k_norm']),self._freq)
        v=v.reshape(8,1,128)
        flat,valid=plan
        ck=self.pool.k[layer_idx].view(-1,1,128)[flat]
        cv=self.pool.v[layer_idx].view(-1,1,128)[flat]
        ck=torch.where(valid[:,None,None],ck,0);cv=torch.where(valid[:,None,None],cv,0)
        kk=torch.cat((ck,k),0).transpose(0,1).unsqueeze(0)
        vv=torch.cat((cv,v),0).transpose(0,1).unsqueeze(0)
        mask=torch.cat((valid,torch.ones(8,device=self.device,dtype=torch.bool)))
        out=F.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),
              kk.expand(1,8,-1,-1),vv.expand(1,8,-1,-1),attn_mask=mask[None,None,None,:],is_causal=False)
        return self.reduce(F.linear(out[0].transpose(0,1).reshape(8,1024),layer['o']))

    def _topk(self,h):
        local=F.linear(h,self.engine.w.lm_head)
        vals,ids=local.topk(16,dim=-1);ids=ids+self.engine.w.vocab_start
        gathered_v=torch.empty((56,16),device=self.device,dtype=vals.dtype)
        gathered_i=torch.empty((56,16),device=self.device,dtype=ids.dtype)
        dist.all_gather_into_tensor(gathered_v,vals.contiguous())
        dist.all_gather_into_tensor(gathered_i,ids.contiguous())
        vals=gathered_v.view(8,7,16).permute(1,0,2).reshape(7,128)
        ids=gathered_i.view(8,7,16).permute(1,0,2).reshape(7,128)
        vals,which=vals.topk(16,-1)
        return ids.gather(-1,which),vals

    def _forward(self):
        h=self._embedding(torch.cat((self.anchor,self.mask_ids)))
        positions=self.context+self.offsets;plan=self._context_plan()
        self._freq=frequencies(positions,h.dtype)
        for i,layer in enumerate(self.layers):
            residual=h;h=rms(h,layer['input_norm'])
            coeff=F.linear(h,layer['attn_kernel']).reshape(8,2,2,384)
            h=grouped(h,coeff[:,0],layer['attn_base'],0)
            h=self._attention(h,positions,layer,i,plan)
            h=grouped(h,coeff[:,1],layer['attn_base'],1)
            residual=(h+residual).bfloat16();h=rms(residual,layer['post_norm'])
            coeff=F.linear(h,layer['mlp_kernel']).reshape(8,2,2,384)
            h=grouped(h,coeff[:,0],layer['mlp_base'],0)
            gate,up=F.linear(h,layer['gu']).chunk(2,-1)
            h=self.reduce(F.linear((F.silu(gate.float())*up.float()).bfloat16(),layer['down']))
            h=grouped(h,coeff[:,1],layer['mlp_base'],1)
            h=(h+residual).bfloat16()
        hidden=rms(h,self.final_norm)[1:]
        candidate,unary=self._topk(hidden)
        hp=F.linear(hidden,self.selector)
        pred_ids=torch.cat((self.anchor.expand(1,16),candidate[:-1]),0)
        scores=unary[:,None,:]+torch.einsum('lpr,lcr,lr->lpc',self.pred[pred_ids],self.succ[candidate],hp)
        return dflash_path_select(scores[None],candidate[None])[0],candidate,unary

    @torch.inference_mode()
    def draft(self,anchor):
        if self.engine.length!=self.pool.lengths[0]:raise RuntimeError('target/draft length mismatch')
        self.anchor.fill_(int(anchor));self.context.fill_(self.pool.lengths[0])
        if self.graph is None:return self._forward()[0]
        self.graph.replay();return self.graph_result[0]

    @torch.inference_mode()
    def capture(self):
        if self.graph is not None:raise RuntimeError('draft graph already captured')
        self.context.fill_(self.pool.lengths[0])
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self._forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.graph_result=self._forward()
        self.capture_append()

    def capture_append(self):
        if self.append_graphs is not None:return
        # Commit graph warmup writes into the first page: preserve resident KV.
        page=int(self.pool.page_table[0,0].item())
        saved_k=self.pool.k[:,page].clone();saved_v=self.pool.v[:,page].clone()
        append_graphs=AppendGraphs(self)
        try:
            for n in range(1,9):append_graphs.build(n)
        except BaseException:
            append_graphs.close()
            raise
        finally:
            self.pool.k[:,page].copy_(saved_k);self.pool.v[:,page].copy_(saved_v)
            torch.cuda.synchronize()
        self.append_graphs=append_graphs

    def close(self):
        torch.cuda.synchronize()
        if self.graph is not None:self.graph.reset();self.graph=None
        if self.append_graphs is not None:
            self.append_graphs.close();self.append_graphs=None


class AppendGraphs:
 # B1 committed-feature projection, exact matrix shape for each accepted length.
 # External offsets and inputs must live until graph destruction.
 # Generator owns all page reservations; replay has no page allocation.
 def __init__(self,d):
  self.d=d;self.graphs={};self.inputs={};self.offsets={}
  self.start=torch.zeros(1,device=d.device,dtype=torch.long)
 @torch.inference_mode()
 def build(self,n):
  d=self.d
  inputs=[torch.zeros((n,6144),device=d.device,dtype=torch.float32) for _ in range(6)]
  offsets=torch.arange(n,device=d.device);self.offsets[n]=offsets
  def compute():
   local=torch.cat([x[:,d.feature_start:d.feature_end].bfloat16() for x in inputs],-1)
   h=rms(d.reduce(F.linear(local,d.fc)),d.hidden_norm)
   positions=self.start+offsets
   freq=frequencies(positions,h.dtype)
   flat=d.pool.page_table[0,positions//d.pool.page_size]*d.pool.page_size+positions%d.pool.page_size
   for l,layer in enumerate(d.layers):
    k,v=F.linear(h,layer['qkv'][1024:]).split(128,-1)
    k=rope_only(rms(k.view(n,1,128),layer['k_norm']),freq)
    d.pool.k[l].view(-1,1,128).index_copy_(0,flat,k)
    d.pool.v[l].view(-1,1,128).index_copy_(0,flat,v.view(n,1,128))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):compute()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):compute()
  self.graphs[n]=g;self.inputs[n]=inputs
 @torch.inference_mode()
 def __call__(self,result):
  d=self.d;n=result.features[0].shape[0];start=result.start
  assert start==d.pool.lengths[0] and 1<=n<=8
  assert start+n<=d.pool.max_sequence_tokens
  self.start.fill_(start)
  for dst,src in zip(self.inputs[n],result.features):dst.copy_(src)
  self.graphs[n].replay();d.pool.lengths[0]=start+n
 def close(self):
  for g in self.graphs.values():g.reset()
  self.graphs.clear();self.inputs.clear();self.offsets.clear()
