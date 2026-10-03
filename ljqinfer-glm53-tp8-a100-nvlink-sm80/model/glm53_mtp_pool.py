"""DFlash2 independent TP8 K/V pool, adapted from GPU140 Qwen's paged pool.
Only accepted target context is appended; draft mask/anchor rows never persist.
Physical pages are shared across slots, logical spans are independent. Allocation
and lifecycle live outside CUDA graphs; kernels consume stable K/V/page tables.
"""
from math import ceil
import torch

VERIFY_Q = 8
TARGET_LAYERS = (5, 19, 33, 47, 61, 75)


class MTPKVPool:
    def __init__(self, *, max_tokens, max_sequence_tokens, max_sequences=1,
                 page_size=64, layers=6, kv_heads=1, head_dim=128,
                 device='cuda', dtype=torch.bfloat16):
        if min(max_tokens,max_sequence_tokens,max_sequences,page_size,layers,kv_heads,head_dim)<=0:
            raise ValueError('invalid MTP pool geometry')
        self.max_sequence_tokens=max_sequence_tokens
        self.max_sequences=max_sequences;self.page_size=page_size
        self.layers=layers;self.kv_heads=kv_heads;self.head_dim=head_dim
        self.num_pages=ceil(max_tokens/page_size)
        self.logical_pages=ceil(max_sequence_tokens/page_size)
        shape=(layers,self.num_pages,page_size,kv_heads,head_dim)
        self.k=torch.empty(shape,device=device,dtype=dtype)
        self.v=torch.empty_like(self.k)
        self.page_table=torch.full((max_sequences,self.logical_pages),-1,device=device,dtype=torch.int64)
        self.host_page_table=[[-1]*self.logical_pages for _ in range(max_sequences)]
        self.free_pages=list(range(self.num_pages-1,-1,-1))
        self.lengths=[0]*max_sequences

    def _sid(self,sid):
        if not isinstance(sid,int) or not 0<=sid<self.max_sequences:raise IndexError(sid)
        return sid

    def reserve(self,sid,end):
        sid=self._sid(sid)
        if not isinstance(end,int) or not 0<=end<=self.max_sequence_tokens:raise ValueError('MTP capacity')
        count=ceil(end/self.page_size);table=self.host_page_table[sid]
        missing=[i for i in range(count) if table[i]<0]
        if len(missing)>len(self.free_pages):raise MemoryError('MTP physical page budget exhausted')
        for i in missing:
            table[i]=self.free_pages.pop();self.page_table[sid,i]=table[i]
        return tuple(table[:count])

    def append(self,sid,start,keys,values):
        sid=self._sid(sid)
        if start!=self.lengths[sid]:raise ValueError('MTP append must start at committed context length')
        if len(keys)!=self.layers or len(values)!=self.layers:raise ValueError('MTP layer count')
        n=keys[0].shape[0]
        if n<1:raise ValueError('empty append')
        shape=(n,self.kv_heads,self.head_dim)
        for t in (*keys,*values):
            if t.shape!=shape or t.device!=self.k.device or t.dtype!=self.k.dtype:
                raise ValueError('MTP tensor geometry/device/dtype')
        self.reserve(sid,start+n)
        # Preflight completed before any write/publication. Per-page contiguous
        # copies avoid scalar GPU reads and never overwrite committed history.
        offset=0
        while offset<n:
            pos=start+offset;page=self.host_page_table[sid][pos//self.page_size]
            j=pos%self.page_size;count=min(n-offset,self.page_size-j)
            for l in range(self.layers):
                self.k[l,page,j:j+count].copy_(keys[l][offset:offset+count])
                self.v[l,page,j:j+count].copy_(values[l][offset:offset+count])
            offset+=count
        self.lengths[sid]=start+n

    def read_layer(self,layer,sid,start,end):
        sid=self._sid(sid)
        if not 0<=layer<self.layers or not 0<=start<=end<=self.lengths[sid]:
            raise ValueError('MTP read outside committed context')
        pos=torch.arange(start,end,device=self.k.device)
        physical=self.page_table[sid,pos//self.page_size]*self.page_size+pos%self.page_size
        shape=(-1,self.kv_heads,self.head_dim)
        return self.k[layer].view(shape)[physical],self.v[layer].view(shape)[physical]

    def truncate(self,sid,end):
        sid=self._sid(sid)
        if not 0<=end<=self.lengths[sid]:raise ValueError('MTP truncate cannot grow')
        self.lengths[sid]=end
        count=ceil(end/self.page_size);table=self.host_page_table[sid]
        for i in range(count,self.logical_pages):
            if table[i]>=0:self.free_pages.append(table[i]);table[i]=-1
        self.page_table[sid,count:].fill_(-1)

    def release(self,sid):
        self.truncate(sid,0)

    @property
    def resident_pages(self):return self.num_pages-len(self.free_pages)
