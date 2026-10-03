"""Bind canonical weights to the text-prefill model, independently of decode.

Caller owns weight loading/lifetime, process group, token history and Past.
Engram tables stay on CPU. No process-group initialization or private KV here.
"""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from model.engram import EngramRows
from model.prefill import PrefillModel
from model.prefill_config import validate_config, rotary_frequencies
from model.prefill_layer import PrefillAttention
from model.prefill_block import PrefillBlock, PrefillEngram, PrefillMoE, DenseRouted
from ops.prefill.gemm import PrefillLinear


class PrefillParallel:
    def __init__(self, group=None):
        if not dist.is_initialized():
            raise RuntimeError('caller must initialize the eight-rank process group')
        self.group = group
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.world != 8:
            raise ValueError('canonical layout requires exactly EP8/TP8')
        from ops.prefill.fast_allreduce import FastAllReduce
        self.fast = FastAllReduce(group=self.group)
        if self.rank == 0:
            print('[parallel] one-shot NVLink all-reduce active', flush=True)

    def fast_path(self, tensor):
        """One-shot NVLink reduce, which beats NCCL below ~64KB.

        The IPC handle exchange is a collective, so it runs at construction:
        graph capture starts before any warmup step and would otherwise bake
        the NCCL fallback into every replay.
        """
        return self.fast if (self.fast is not None and self.fast.eligible(tensor)) else None

    def sum(self, tensor):
        fast = self.fast_path(tensor)
        if fast is None:
            dist.all_reduce(tensor, group=self.group)
        else:
            fast(tensor)

    def __call__(self, tensor):
        self.sum(tensor)

    def scatter_sum(self, tensor, out):
        """Sum tensor across ranks; this rank keeps its own row shard."""
        dist.reduce_scatter_tensor(out, tensor, group=self.group)

    def gather_rows(self, local, out):
        dist.all_gather_into_tensor(out, local, group=self.group)

    def logits(self, tensor):
        """Gather a vocabulary-sharded logit block into the full vocabulary.

        Every decode-side caller (the head and the Markov chain's five links)
        passes a single row, and for one row the shard-major layout an
        into_tensor gather produces *is* the concatenation. The list form
        instead lands one copy per rank and then the cat copies the whole
        vocabulary a second time -- sixteen launches per link for a buffer
        that a single collective already places correctly.
        """
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            out = torch.empty(self.world * tensor.shape[1], dtype=tensor.dtype,
                              device=tensor.device)
            dist.all_gather_into_tensor(out, tensor.reshape(-1),
                                        group=self.group)
            return out.view(1, -1)
        parts = [torch.empty_like(tensor) for _ in range(self.world)]
        dist.all_gather(parts, tensor, group=self.group)
        return torch.cat(parts, -1)


def build_prefill(config, weights, hasher, host_tables, *, device, length,
                  parallel=None, embed=None, head=None, max_position=None):
    # A100 tensor cores for the residual fp32 GEMMs (index/compressor paths).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    """Full released model; weights are unsharded (1) or canonical rank-local (8).

    Optional embed/head are explicit storage callbacks, not model substitutes;
    distributed callbacks must already return replicated embeddings/full logits.
    """
    validate_config(config)
    c = config
    world = 1 if parallel is None else parallel.world
    rank = None if parallel is None else parallel.rank
    reduce_sum = None if parallel is None else parallel
    if world not in (1,8):
        raise ValueError('only unsharded or canonical EP8/TP8 weights')
    if set(host_tables) != set(c['engram_layer_ids']):
        raise ValueError('every configured host Engram table is required')
    lin = PrefillLinear(weights)
    if parallel is not None:
        from ops.prefill.projection_workspace import ProjectionWorkspace
        lin.projection_workspace = ProjectionWorkspace(length, c['dim'], device)
        from model.prefill_block import MoEReduceBuffer
        moe_buffer = MoEReduceBuffer(length, c['dim'], device)
    else:
        moe_buffer = None
    # Workspaces are sized by the chunk; rotary tables must instead span every
    # position the sequence can reach, or chunk two of a long prompt indexes
    # past the end of the table and receives an empty slice.
    span = max(int(max_position or length), length)
    fs = {b:rotary_frequencies(c,2 if b else 0,span,device=device) for b in (False,True)}
    if embed is None:
        def embed(tokens):
            w = weights['embed.weight']
            ids = torch.as_tensor(tokens,device=device,dtype=torch.long)
            if (ids < 0).any() or (ids >= c['vocab_size']).any():
                raise ValueError('token outside vocabulary')
            if world == 1:
                return F.embedding(ids,w)
            n = c['vocab_size']//world
            local = ids-rank*n
            mask = (local < 0) | (local >= n)
            out = F.embedding(local.masked_fill(mask,0),w).masked_fill(mask[:,None],0)
            reduce_sum(out)
            return out
    if head is None:
        def head(hidden):
            logits = F.linear(hidden.float(),weights['head.weight'].float())
            return logits if parallel is None else parallel.logits(logits)
    # Serial blocks share two ping-pong outputs; encoder tails are cloned.
    # Attention writes buffer 0 from buffer 1; FFN writes buffer 1 from 0.
    expand_workspace = None
    if parallel is not None:
        expand_workspace = tuple(torch.empty((length, 4, c['dim']),
            device=device, dtype=torch.bfloat16) for _ in range(2))
    sparse_workspace = None
    swa_workspace = None
    if parallel is not None:
        from ops.prefill.sparse_workspace import Workspace as SparseWorkspace
        sparse_workspace = SparseWorkspace(length, c['n_heads']//world,
            c['head_dim'], device, c['index_topk'], global_capacity=span)
    routed_workspace = None
    engram_workspace = None
    blocks = []
    for layer in range(c['n_layers']):
        p = f'layers.{layer}'
        attention = PrefillAttention(layer,c,weights,lin,fs[bool(c['compress_ratios'][layer])],
                                     world=world,reduce_sum=reduce_sum)
        if not c['compress_ratios'][layer] and parallel is not None:
            if swa_workspace is None:
                from ops.prefill.swa_workspace import Workspace
                swa_workspace = Workspace(length, c['n_heads']//world,
                    c['head_dim'], device)
            attention.swa_workspace = swa_workspace
        if c['compress_ratios'][layer]:
            attention.sparse_workspace = sparse_workspace
        if parallel is None:
            routed = DenseRouted(p+'.ffn',c['n_routed_experts'],lin,c['swiglu_limit'])
        elif routed_workspace is None:
            from ops.prefill.moe_workspace import WorkspaceRouted
            routed_workspace = WorkspaceRouted(p+'.ffn',c,weights,parallel,length)
            # blocks[20:] only ever run on the fixed-length CED tail, so the
            # fused FP4 path is pinned there; chunk layers stay on CUTLASS.
            routed_workspace.fused = layer >= 20
            routed = routed_workspace
        else:
            routed = routed_workspace.bind(p+'.ffn', fused=layer >= 20)
        moe = PrefillMoE(layer,c,weights,lin,routed,reduce_shared=reduce_sum,buffer=moe_buffer)
        engram = None
        if layer in host_tables:
            if engram_workspace is None and parallel is not None:
                from ops.prefill.engram_rows import RowWorkspace
                engram_workspace = RowWorkspace(length, 3, device)
            rows = EngramRows(hasher,layer,host_tables[layer],rank=rank,
                              workspace=engram_workspace)
            engram = PrefillEngram(layer,weights,lin,rows,eps=c['norm_eps'],
                                   reduce_sum=reduce_sum,row_cap=length)
        blocks.append(PrefillBlock(layer,c,weights,attention,moe,engram=engram,
            expand_workspace=expand_workspace))
    return PrefillModel(c,blocks,embed,head,weights['norm.weight'],
                        target_layers=c['dspark_target_layer_ids'])
