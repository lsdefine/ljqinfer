"""Q8-per-request projections with joint GEMMs and TP collectives.
KV/index selection stays request-local; no cross-request attention is permitted.
"""
import torch
import torch.distributed as dist
from model.glm53_block import _rms_half
from ops.sparse_index_decode import _score_query
from ops import sparse_topk_v2
from ops.sparse_ops import sparse_mla


def select_joint(bindings,queries,weights,engines):
    b=len(bindings);device=queries.device
    packed=torch.empty((b,8,4,144),device=device,dtype=queries.dtype)
    packed[...,:128].copy_(queries.reshape(b,8,4,128))
    packed.view(torch.float32)[...,64].copy_(weights.reshape(b,8,4))
    gathered=torch.empty((8*b,8,4,144),device=device,dtype=queries.dtype)
    parallel=bindings[0].parallel;parallel.gather_rows(packed,gathered)
    gathered=gathered.view(8,b,8,4,144)
    selected=torch.empty((b,1,bindings[0].topk),device=device,dtype=torch.int32)
    for row,(binding,e) in enumerate(zip(bindings,engines)):
        q=gathered[:,row].contiguous();n=binding.capacity
        score=torch.empty((1,n),device=device,dtype=torch.float32)
        _score_query[(1,min(128,(n+127)//128))](q,binding.index_pool,q.view(torch.float32),binding.index_table,
            e.positions,e.context,score,8,n,binding.index_pool.shape[1],parallel.rank,1,2,128,num_warps=4)
        pos=torch.minimum(e.positions[parallel.rank:parallel.rank+1],e.context.reshape(())-1).clamp_min(-1)
        sparse_topk_v2.select_out(score,pos,selected[row],tuple(x[:1] for x in binding.topk_workspace))
    ids=torch.empty((8*b,1,bindings[0].topk),device=device,dtype=torch.int32)
    parallel.gather_rows(selected,ids)
    ids=ids.view(8,b,bindings[0].topk)
    for row,binding in enumerate(bindings):binding.ids.copy_(ids[:,row])


def attention_joint(blocks,x,engines,layer):
    """One layer; x ordered as contiguous Q8 blocks in engines order."""
    bindings=[block.sparse for block in blocks];a=blocks[0].attn
    rows=x.shape[0]
    xn=_rms_half(x,a['norm']);qa=_rms_half(xn@a['q_a'].T,a['q_a_norm'])
    qb=(qa@a['q_b'].T).view(rows,8,256)
    ql=torch.bmm(qb[...,:192].transpose(0,1),a['k_b'].transpose(1,2)).transpose(0,1)
    raw=xn@a['kv_a'].T;kvn=_rms_half(raw[:,:512],a['kv_a_norm'])
    full=bindings[0].shared_from is None
    if full:
        w=bindings[0].index_weights
        iq=(qa@w['wq_b'].T).view(rows,4,128)
        ik=torch.nn.functional.layer_norm(xn@w['wk'].T,(128,),w['k_norm'],w['k_bias'],1e-6)
        weights=(xn.float()@w['weights_proj'].T)*(32**-0.5)
    qs=[];iqs=[]
    for row,(binding,e) in enumerate(zip(bindings,engines)):
        sl=slice(row*8,(row+1)*8);ef=binding.prefill_elementwise
        ef.begin(e.positions,binding.inv_freq)
        qs.append(ef.query(ql[sl],qb[sl,:,192:]))
        ef.cache(kvn[sl],raw[sl,512:],e.kv[layer],e.table,e.positions)
        if full:
            iqs.append(ef.index_query(iq[sl]))
            ef.index_cache(ik[sl],binding.index_pool,binding.index_table,e.positions)
        elif binding.shared_from.metadata!=(e.positions.data_ptr(),e.context.data_ptr(),'decode'):
            raise RuntimeError('joint index producer metadata mismatch')
    if full:select_joint(bindings,torch.cat(iqs),weights,engines)
    latent=torch.empty((rows,8,512),device=x.device,dtype=x.dtype)
    for row,(binding,e,q) in enumerate(zip(bindings,engines,qs)):
        binding.metadata=(e.positions.data_ptr(),e.context.data_ptr(),'decode')
        sparse_mla(q,e.kv[layer],e.table,binding.ids.to(torch.int32),e.positions,e.context,
                   latent[row*8:(row+1)*8],e.decode_mla_workspace,splits=32)
    heads=torch.bmm(latent.transpose(0,1),a['v_b'].transpose(1,2)).transpose(0,1)
    partial=heads.reshape(rows,2048)@a['o'].T
    if blocks[0].all_reduce is not None:blocks[0].all_reduce(partial)
    return (x+partial).half()
