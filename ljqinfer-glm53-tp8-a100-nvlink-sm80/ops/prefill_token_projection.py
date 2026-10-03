"""TP8 token-owned projections. Fixed 12288-row first-chunk specialization.

All buffers belong to the phase owner and are reused across sequential layers.
Only replicated q_a/kv_a are distributed; no floating-point reduction collective.
Other lengths/chunks must use the original GEMM shapes to preserve rounding.
"""
import torch
import torch.distributed as dist
from .prefill_rms_full import extension

class TokenProjection:
    tokens = 12288

    def __init__(self, device, pair_group):
        if dist.get_world_size()!=8 or dist.get_world_size(pair_group)!=2:
            raise ValueError('TokenProjection requires TP8 with adjacent rank pairs')
        self.rank=dist.get_rank()
        self.pair_rank=dist.get_rank(pair_group)
        if self.pair_rank!=self.rank%2:
            raise ValueError('TokenProjection rank-pair order mismatch')
        self.group=pair_group
        self.rms=extension().forward
        def alloc(n,h):return torch.empty((n,h),device=device,dtype=torch.float16)
        self.q_local=alloc(1536,2048)
        self.q_norm=alloc(1536,2048)
        self.q_all=alloc(12288,2048)
        self.kv_local=alloc(6144,576)
        self.kv_all=alloc(12288,576)
        self.xq=alloc(1536,6144)
        self.xkv=alloc(6144,6144)
        self.kv_work=None

    def begin(self,x,a,*,normalize_input):
        if self.kv_work is not None:
            raise RuntimeError('previous projection has not been consumed')
        if x.shape!=(12288,6144) or x.dtype!=torch.float16:
            raise ValueError('TokenProjection fixed input shape/dtype')
        lo=self.rank*1536
        xq=x[lo:lo+1536]
        if normalize_input:
            self.rms(xq,a['norm'],self.xq)
            xq=self.xq
        torch.mm(xq,a['q_a'].t(),out=self.q_local)
        self.rms(self.q_local,a['q_a_norm'],self.q_norm)
        work=dist.all_gather_into_tensor(self.q_all,self.q_norm,async_op=True)
        lo=self.pair_rank*6144
        xkv=x[lo:lo+6144]
        if normalize_input:
            self.rms(xkv,a['norm'],self.xkv)
            xkv=self.xkv
        torch.mm(xkv,a['kv_a'].t(),out=self.kv_local)
        # Explicit synchronization before switching NCCL process groups.
        work.wait()
        self.kv_work=dist.all_gather_into_tensor(
            self.kv_all,self.kv_local,group=self.group,async_op=True)
        return self.q_all

    def finish(self):
        if self.kv_work is None:raise RuntimeError('projection has not begun')
        self.kv_work.wait()
        self.kv_work=None
        return self.kv_all
