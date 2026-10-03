"""Explicit TP8 prefill: paired query ownership, replicated KV and metadata.
All ranks create groups in the same order before capture. No runtime fallback.
Workspace is shape-fixed; allocate one per concurrent invocation/graph.
"""
import torch
import torch.distributed as dist
from .sparse_mla_prefill import extension


def create_pair_group():
    if not dist.is_initialized() or dist.get_world_size() != 8:
        raise ValueError('paired MLA requires initialized TP8')
    groups = [dist.new_group([i, i + 1]) for i in range(0, 8, 2)]
    return groups[dist.get_rank() // 2]


class PairWorkspace:
    def __init__(self, q, ids, group):
        if q.ndim != 3 or q.shape[1:] != (8, 576) or q.dtype != torch.float16 or not q.is_cuda:
            raise ValueError('q must be CUDA FP16 [T,8,576]')
        if ids.ndim != 2 or ids.shape[0] != q.shape[0] or ids.dtype != torch.int64 or ids.device != q.device:
            raise ValueError('ids must be colocated int64 [T,K]')
        if dist.get_world_size(group) != 2:
            raise ValueError('two-rank group required')
        self.t, self.k = ids.shape
        if self.t < 1:
            raise ValueError('positive query count required')
        self.half = (self.t + 1) // 2
        self.lo = dist.get_rank(group) * self.half
        self.group = group
        options = dict(device=q.device, dtype=q.dtype)
        self.qsend = torch.zeros((2*self.half,8,576), **options)
        # Measured causal work balance is specific to the first 12K chunk.
        self.balanced_sizes = [6648,5640] if self.t==12288 else None
        local = self.balanced_sizes[dist.get_rank(group)] if self.balanced_sizes else self.half
        receive_rows = max(2*self.half,2*local)
        recv = torch.empty((receive_rows,8,576), **options)
        send = torch.empty((receive_rows,8,512), **options)
        self.qrecv = recv[:2*self.half]
        self.osend = send[:2*self.half]
        self.balanced_recv = recv[:2*local]
        self.balanced_send = send[:2*local]
        self.balanced_lo = 0 if dist.get_rank(group)==0 else 6648
        self.orecv = torch.empty_like(self.osend)
        self.ids = torch.full((2*self.half,self.k), -1, device=q.device, dtype=torch.int64)
        self.pos = torch.full((2*self.half,), -1, device=q.device, dtype=torch.int64)
        extension()

    def __call__(self, q, pool, table, ids, pos, ctx, out, *, first_chunk=False):
        if q.shape != (self.t,8,576) or ids.shape != (self.t,self.k) or out.shape != (self.t,8,512):
            raise ValueError('workspace shape mismatch')
        if q.device != self.qsend.device or q.dtype != torch.float16 or out.dtype != q.dtype or out.device != q.device:
            raise ValueError('workspace device/dtype mismatch')
        if not q.is_contiguous() or not out.is_contiguous():
            raise ValueError('contiguous q/out required')
        if first_chunk and self.t==12288 and self.balanced_sizes is not None:
            local = self.balanced_recv.shape[0]//2
            dist.all_to_all_single(self.balanced_recv,q,output_split_sizes=[local]*2,
                                   input_split_sizes=self.balanced_sizes,group=self.group)
            sl = slice(self.balanced_lo,self.balanced_lo+local)
            extension().forward(self.balanced_recv,pool,ids[sl],table,pos[sl],ctx,self.balanced_send,1/16)
            dist.all_to_all_single(out,self.balanced_send,output_split_sizes=self.balanced_sizes,
                                   input_split_sizes=[local]*2,group=self.group)
            return out
        if self.t % 2:
            self.qsend[:self.t].copy_(q)
            self.ids[:self.t].copy_(ids)
            self.pos[:self.t].copy_(pos)
            send, all_ids, all_pos = self.qsend, self.ids, self.pos
        else:
            send, all_ids, all_pos = q, ids, pos
        dist.all_to_all_single(self.qrecv, send, group=self.group)
        sl = slice(self.lo, self.lo+self.half)
        extension().forward(self.qrecv, pool, all_ids[sl], table, all_pos[sl], ctx, self.osend, 1/16)
        target = self.orecv if self.t % 2 else out
        dist.all_to_all_single(target, self.osend, group=self.group)
        if self.t % 2:
            out.copy_(target[:self.t])
        return out


class ProjectedPair:
    """First-chunk V projection before exchange; buffers are sequentially shared.
    Weight gathering is initialization-only, never on the forward path.
    """
    def __init__(self,base,weight,buffers=None):
        if base.t != 12288 or tuple(weight.shape) != (8,256,512):
            raise ValueError("projected pair requires first 12288 chunk and [8,256,512] V weights")
        self.base=base
        self.weights=[torch.empty_like(weight) for _ in range(2)]
        dist.all_gather(self.weights,weight.contiguous(),group=base.group)
        self.local=base.balanced_recv.shape[0]//2
        if buffers is None:
            buffers=(torch.empty((2*self.local,8,256),device=weight.device,dtype=weight.dtype),
                                                torch.empty((base.t,8,256),device=weight.device,dtype=weight.dtype))
        self.send,self.recv=buffers
    def __call__(self,q,pool,table,ids,pos,ctx):
        base=self.base;local=self.local
        dist.all_to_all_single(base.balanced_recv,q,output_split_sizes=[local]*2,
                                                                                                    input_split_sizes=base.balanced_sizes,group=base.group)
        sl=slice(base.balanced_lo,base.balanced_lo+local)
        extension().forward(base.balanced_recv,pool,ids[sl],table,pos[sl],ctx,base.balanced_send,1/16)
        for peer in range(2):
            latent=base.balanced_send[peer*local:(peer+1)*local]
            torch.bmm(latent.transpose(0,1),self.weights[peer].transpose(1,2),
                                                    out=self.send[peer*local:(peer+1)*local].transpose(0,1))
        dist.all_to_all_single(self.recv,self.send,output_split_sizes=base.balanced_sizes,
                                                                                                    input_split_sizes=[local]*2,group=base.group)
        return self.recv
