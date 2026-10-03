"""SM80 prefill MoE; V4 grouped GEMM, V4.1 quantization/routing.

Only a bounded batch of local experts is expanded to BF16, never the model.
No decode dispatch, no cached expanded weights, no implicit backend fallback.
"""
from functools import lru_cache
from pathlib import Path
import os
import torch
from ops.prefill.gemm import activation_fp8
from ops.prefill.residual import swiglu


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / 'cuda'
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '4')
    return load(name='v41_prefill_grouped',
                sources=[str(root / n) for n in ('prefill_moe_cutlass_gemm.cu', 'fp4_unpack.cu')],
                extra_include_paths=[str(root / 'cutlass/include')],
                extra_cuda_cflags=['-O3'], verbose=False)


def unpack_fp4(weight, scale):
    """Canonical low-nibble-first E2M1, per-row K/32 E8M0 -> BF16."""
    return extension().unpack_fp4(weight, scale)


class GroupedBankRouted:
    def __init__(self, prefix, config, weights, parallel, *, expert_batch=8):
        if not 1 <= expert_batch <= config['n_routed_experts']//8:
            raise ValueError('invalid expert batch')
        self.prefix, self.c, self.w, self.parallel = prefix, config, weights, parallel
        self.expert_batch = expert_batch
        self.calls = self.gemm_calls = 0

    @torch.no_grad()
    def __call__(self, x, ids, probabilities):
        if x.dtype != torch.bfloat16 or not x.is_cuda or x.ndim != 2:
            raise ValueError('CUDA BF16 [tokens, dim] required')
        if ids.shape != probabilities.shape or ids.shape[0] != x.shape[0]:
            raise ValueError('routing geometry')
        count = self.c['n_routed_experts']//8
        local = ids - self.parallel.rank*count
        token, choice = torch.where((local >= 0) & (local < count))
        expert = local[token,choice]
        order = expert.argsort(stable=True)
        token, choice, expert = token[order], choice[order], expert[order]
        counts = torch.bincount(expert, minlength=count).cpu().tolist()
        active = [e for e,n in enumerate(counts) if n]
        out = torch.zeros_like(x, dtype=torch.float32)
        base = self.prefix+'.local_experts.'
        offset = 0
        mod = extension()
        self.calls += 1
        for start in range(0,len(active),self.expert_batch):
            group = active[start:start+self.expert_batch]
            sizes = torch.tensor([counts[e] for e in group], device=x.device, dtype=torch.int64)
            rows = sum(counts[e] for e in group)
            tok, ch = token[offset:offset+rows], choice[offset:offset+rows]
            a = activation_fp8(x[tok]).bfloat16().contiguous()
            # Read one canonical bank entry at a time, compatible with lazy loaders.
            ws, ss = [], []
            for e in group:
                ws.append(self.w[base+'w13.weight'][e])
                ss.append(self.w[base+'w13.scale'][e])
            w13, s13 = torch.stack(ws), torch.stack(ss)
            del ws, ss
            gate = mod.grouped_gemm_sm80(a, unpack_fp4(w13[:,0].contiguous(), s13[:,0].contiguous()), sizes, 0)
            up = mod.grouped_gemm_sm80(a, unpack_fp4(w13[:,1].contiguous(), s13[:,1].contiguous()), sizes, 0)
            del w13, s13, a
            mid = swiglu(gate, up, self.c['swiglu_limit'], probabilities[tok,ch,None])
            del gate, up
            mid = activation_fp8(mid).bfloat16().contiguous()
            ws, ss = [], []
            for e in group:
                ws.append(self.w[base+'w2.weight'][e])
                ss.append(self.w[base+'w2.scale'][e])
            wd = unpack_fp4(torch.stack(ws), torch.stack(ss))
            del ws, ss
            down = mod.grouped_gemm_sm80(mid, wd, sizes, 0)
            self.gemm_calls += 3
            out.index_add_(0,tok,down.float())
            del mid, wd, down
            offset += rows
        self.parallel.sum(out)
        return out
