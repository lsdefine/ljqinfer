"""Serial routed-MoE implementation; borrowed output, fixed scratch capacity.

Routing sort/selection still allocate. Dense workspaces do not grow in __call__.
TP8 over the expert intermediate dim: every rank holds all experts and a
1/8 slice of each expert's intermediate width, so per-rank work never depends
on routing skew. Partial 
...[Truncated]...
        order = expert.argsort(stable=True)
        token, choice, expert = token[order], choice[order], expert[order]
"""
from pathlib import Path
from functools import lru_cache
import os
import torch
_STATS={'sync_ms':0.0,'calls':0}


@lru_cache(maxsize=1)
def modules():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    root = Path(__file__).parent / 'cuda'
    includes = [str(root/'cutlass/include')]
    chain = load(name='dsv41_prefill_moe_workspace_chain', sources=[str(root/'moe_workspace_chain.cu')],
                 extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)
    unpack = load(name='dsv41_prefill_moe_workspace_unpack', sources=[str(root/'moe_workspace_unpack.cu')],
                  extra_cuda_cflags=['-O3'], verbose=False)
    grouped = load(name='dsv41_prefill_moe_workspace_grouped', sources=[str(root/'moe_workspace_grouped.cu')],
                   extra_include_paths=includes,
                   extra_cuda_cflags=['-O3', '-DCUTLASS_GROUPED_STANDALONE'], verbose=False)
    route = load(name='dsv41_prefill_moe_workspace_route', sources=[str(root/'moe_workspace_route.cu')],
                 extra_cuda_cflags=['-O3'], verbose=False)
    return chain, unpack, grouped, route


class WorkspaceRouted:
    def __init__(self, prefix, config, weights, parallel, capacity):
        from ops.prefill.grouped_moe import extension
        self.prefix, self.c, self.w, self.parallel = prefix, config, weights, parallel
        self.capacity = int(capacity)
        self.count = config['n_routed_experts']
        self.d = config['dim']
        # Tensor parallel: every rank owns all experts, each holding 1/8 of the
        # intermediate dimension (w1/w3 rows, w2 reduction columns).
        self.k = config['moe_inter_dim']//8
        self.topk = config['n_activated_experts']
        if config['moe_inter_dim'] % 8 or not 1 <= self.count <= 512 or self.capacity <= 0:
            raise ValueError('positive capacity and TP8 expert geometry required')
        base = prefix+'.local_experts.'
        self.w13 = [weights[base+'w13.weight'][e] for e in range(self.count)]
        self.s13 = [weights[base+'w13.scale'][e] for e in range(self.count)]
        self.w2 = [weights[base+'w2.weight'][e] for e in range(self.count)]
        self.s2 = [weights[base+'w2.scale'][e] for e in range(self.count)]
        # Whole banks are what the fused FP4 GEMM indexes by expert on device.
        self.w13_bank = weights[base+'w13.weight']
        self.s13_bank = weights[base+'w13.scale']
        self.w2_bank = weights[base+'w2.weight']
        self.s2_bank = weights[base+'w2.scale']
        self.device = self.w13[0].device
        if self.device.type != 'cuda':
            raise ValueError('CUDA bank required')
        self.chain, self.unpack, self.grouped, self.route = modules()
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            n = self.capacity*self.topk
            def bf(*shape):
                return torch.empty(shape, device=self.device, dtype=torch.bfloat16)
            self.a = bf(n, self.d)
            self.both = bf(n, 2*self.k)
            self.mid = bf(n, self.k)
            self.down = bf(n, self.d)
            self.expanded = bf(self.count*2*self.k*self.d)
            self.cast = torch.empty((n,self.d), device=self.device, dtype=torch.float32)
            self.out = torch.empty((self.capacity,self.d), device=self.device, dtype=torch.float32)
            self.sizes = torch.empty(self.count, dtype=torch.int64)
            # Upper bound on descriptor storage, independently checked by C++.
            self.host = [torch.empty(1<<18, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            self.desc = [torch.empty(1<<18, dtype=torch.uint8, device=self.device) for _ in range(2)]
            self.work = [torch.empty(1<<20, dtype=torch.uint8, device=self.device) for _ in range(2)]
            self.done = [torch.cuda.Event() for _ in range(2)]
            self.pending = [False, False]
            # Canonical LUT built once using the existing decoder, no alternative math.
            packed = torch.arange(256, device=self.device).to(torch.int8)
            packed = packed.view(1,1,256).expand(256,1,256).contiguous()
            scales = torch.arange(256, device=self.device, dtype=torch.uint8)
            scales = scales.view(256,1,1).expand(256,1,16).contiguous()
            decoded = extension().unpack_fp4(packed, scales)
            self.lut = decoded[:,0,::2][:,:16].contiguous()
        # FP4 banks are consumed in place by the fused grouped GEMM; the tile
        # list is rebuilt on device each call, so shapes stay capture stable.
        from ops.prefill.fp4_gemm import extension as fp4_extension
        self.fp4 = fp4_extension()
        from ops.decode.fp4_moe_decode import extension as fused_moe_decode
        self.moe_dec = (None if __import__('os').environ.get('V41_MOE_DECODE', '1') == '0'
                        else fused_moe_decode())
        tiles = self.count + (self.capacity*self.topk)//64 + 2
        self.tile_e = torch.empty(tiles, dtype=torch.int32, device=self.device)
        self.tile_r0 = torch.empty(tiles, dtype=torch.int32, device=self.device)
        self.tile_rn = torch.empty(tiles, dtype=torch.int32, device=self.device)
        # Measured crossover on A100 sits between 512 and 1024 routed rows:
        # the fused FP4 path wins at CED/decode row counts, CUTLASS on unpacked
        # weights wins on full chunks. Which phase owns a layer is static
        # (prefill.py runs blocks[:20] per chunk and blocks[20:] on the fixed
        # 128-row CED tail), so the owner pins the path at bind time rather
        # than re-deriving it from row counts on every call.
        self.fused = False
        self.decode_only = False
        self.calls = self.gemm_calls = 0

    def bind(self, prefix, *, fused=False):
        """Bind another layer; scratch and descriptor events share serial ownership.

        The returned output is borrowed until any binding is invoked again.
        """
        from copy import copy
        bound = copy(self)
        bound.prefix = prefix
        bound.fused = bool(fused)
        bound.decode_only = getattr(self, 'decode_only', False)
        base = prefix+'.local_experts.'
        for attr, suffix in (('w13', 'w13.weight'), ('s13', 'w13.scale'),
                             ('w2', 'w2.weight'), ('s2', 'w2.scale')):
            bank = self.w[base+suffix]
            previous = getattr(self, attr)
            if (bank.shape[0] != self.count or bank.device != self.device
                    or any(bank[e].shape != previous[e].shape
                           or bank[e].dtype != previous[e].dtype
                           for e in range(self.count))):
                raise ValueError('shared MoE bank geometry/device/dtype')
            setattr(bound, attr, [bank[e] for e in range(self.count)])
            setattr(bound, attr+'_bank', bank)
        bound.calls = bound.gemm_calls = 0
        return bound

    def gemm(self, stage, a, w, sizes, out):
        if self.pending[stage]:
            self.done[stage].synchronize()
            self.pending[stage] = False
        try:
            self.grouped.grouped_gemm_sm80(a,w,sizes,0,out,
                                           self.host[stage],self.desc[stage],self.work[stage])
        finally:
            # Also protects a queued copy if the extension raises after enqueue.
            self.done[stage].record(self.stream)
            self.pending[stage] = True
        self.gemm_calls += 1

    def routing_buffers(self):
        """Fixed-capacity routing scratch; allocated once, never grown (capture-safe)."""
        if getattr(self, 'rt_token', None) is None:
            n = self.capacity*self.topk
            blocks = (n+1023)//1024
            def z(m):
                return torch.zeros(int(m), dtype=torch.int64, device=self.device)
            self.rt_token, self.rt_choice = z(n), z(n)
            self.rt_counts, self.rt_offsets = z(self.count), z(self.count+1)
            self.rt_blocks = z(blocks*self.count)
            self.rt_slot = z(self.capacity*self.topk)

    def gemm_device(self, stage, a, w, out):
        """Grouped GEMM with descriptors built on device from routing counts."""
        self.grouped.grouped_gemm_sm80_device(a, w, self.rt_counts, self.rt_offsets[:self.count],
                                              0, out, self.desc[stage], self.work[stage])
        self.gemm_calls += 1

    @torch.no_grad()
    def __call__(self, x, ids, probabilities, reduce=True):
        if (x.device != self.device or x.dtype != torch.bfloat16 or x.ndim != 2
                or x.shape[1] != self.d or x.shape[0] > self.capacity or not x.is_contiguous()):
            raise ValueError('rank-local contiguous BF16 activation capacity/geometry')
        if (ids.shape != (x.shape[0],self.topk) or probabilities.shape != ids.shape
                or ids.dtype != torch.int64 or probabilities.dtype != torch.float32
                or ids.device != self.device or probabilities.device != self.device
                or not probabilities.is_contiguous()):
            raise ValueError('routing geometry/device/dtype')
        # Serial ownership still holds; CUDA graph capture legitimately runs the
        # body on a private capture stream, which stays serial by construction.
        if (torch.cuda.current_stream(self.device) != self.stream
                and not torch.cuda.is_current_stream_capturing()):
            raise ValueError('workspace is owned by one serial CUDA stream')
        self.routing_buffers()
        # Decode: one fused kernel does the whole expert FFN (gate/up + SiLU +
        # down + routing-weight reduce), replacing the gather_quant / gemv /
        # glu_quant / gemv / scatter_add chain, which is a prefill shape
        # pipeline that degenerates at decode row counts.  Static shape and
        # no host readback, so it stays capture safe.
        # Structural dispatch: a workspace built for the decode graph always
        # takes the fused decode kernel (trunk topk=6 and draft topk=3 alike);
        # no row-count threshold, decode never falls back to the prefill chain.
        if self.decode_only and self.moe_dec is not None:
            self.calls += 1
            out = self.moe_dec.moe_rank_decode_fp4(
                x, ids, probabilities, self.w13_bank, self.s13_bank,
                self.w2_bank, self.s2_bank)
            if reduce:
                self.parallel.sum(out)
            return out
        # Tensor parallel: every expert is local, so every routed slot is kept and
        # the problem list is the full expert bank with empty experts left at M=0.
        # Counting sort, descriptors and the scatter all read counts on device:
        # no host readback, no data dependent shapes, capturable end to end.
        rows, count = x.shape[0]*self.topk, self.count
        self.route.route(ids.contiguous(), 0, count, self.rt_token, self.rt_choice,
                         self.rt_counts, self.rt_offsets, self.rt_blocks)
        token, choice = self.rt_token[:rows], self.rt_choice[:rows]
        out = self.out[:x.shape[0]]
        out.zero_()
        self.calls += 1
        a, both, mid, down = self.a[:rows], self.both[:rows], self.mid[:rows], self.down[:rows]
        self.chain.gather_quant(x, token, a)
        if self.fused:
            self.fp4.fp4_grouped_gemm_device(
                a, self.w13_bank.view(count,2*self.k,-1), self.s13_bank.view(count,2*self.k,-1),
                self.lut, self.rt_counts, self.rt_offsets, self.tile_e, self.tile_r0,
                self.tile_rn, both)
        else:
            w13 = self.expanded[:count*2*self.k*self.d].view(count,2*self.k,self.d)
            self.unpack.unpack(self.w13, self.s13, self.lut, w13)
            self.gemm_device(0, a, w13, both)
        self.chain.glu_quant(both, probabilities, token, choice, self.c['swiglu_limit'], mid)
        if self.fused:
            self.fp4.fp4_grouped_gemm_device(
                mid, self.w2_bank, self.s2_bank, self.lut, self.rt_counts, self.rt_offsets,
                self.tile_e, self.tile_r0, self.tile_rn, down)
        else:
            wd = self.expanded[:count*self.d*self.k].view(count,self.d,self.k)
            self.unpack.unpack(self.w2, self.s2, self.lut, wd)
            self.gemm_device(1, mid, wd, down)
        self.route.scatter_add(down, token, choice, self.rt_offsets[count:count+1],
                               out, self.rt_slot, rows, self.topk)
        if reduce:
            self.parallel.sum(out)
        return out
