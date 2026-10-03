"""Bounded serial projection scratch; output storage remains caller-owned.
Retains reference N256 ATen FP32 GEMM, CUDA quantization and weight unpack.
"""
from functools import lru_cache
from pathlib import Path
import os
import io, torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_prefill_projection_workspace',
        sources=[str(Path(__file__).parent/'cuda'/'projection_workspace.cu')],
        extra_cuda_cflags=['-O3'], verbose=False)


def dequant_e8m0_bf16(weight, scale):
    """FP8 E4M3 weight x E8M0 32x32-block scale -> BF16, exactly.

    The scale is a power of two and E4M3 carries three mantissa bits, so BF16's
    eight bits lose nothing: this is a representation change, not an
    approximation.  Done once per weight at first use, never in the hot path.
    """
    n, k = int(weight.shape[0]), int(weight.shape[1])
    s = torch.exp2(scale.to(torch.float32) - 127.0)
    s = s.repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def packed_fp8_ok(weight, scale, k):
    """Shapes the packed-FP8 tensor-core GEMM tiles exactly (N32 x K128)."""
    return (scale is not None and weight.dim() == 2 and weight.is_contiguous()
            and weight.dtype == torch.float8_e4m3fn and weight.shape[1] == k
            and scale.dtype == torch.uint8 and scale.is_contiguous()
            and weight.shape[0] % 32 == 0 and k % 128 == 0)


class ProjectionWorkspace:
    def __init__(self, capacity, max_width, device):
        if capacity <= 0 or max_width <= 0 or max_width % 32:
            raise ValueError('positive capacity and K32 width required')
        self.capacity, self.max_width = capacity, max_width
        self.a = torch.empty(capacity*max_width, device=device, dtype=torch.float32)
        self.w = torch.empty(256*max_width, device=device, dtype=torch.float32)
        self.p = torch.empty(capacity*256, device=device, dtype=torch.float32)
        self.stream = torch.cuda.current_stream(self.a.device)
        self.mod = extension()
        self.calls = 0
        self.bf16 = {}
        self.bf16_calls = 0
        self.fp8 = {}
        self.fp8_calls = 0

    def fused(self, x, parts):
        """One GEMV for the projections that read the same activation.

        A decode-width GEMV is launch-bound, not math-bound, so weights sharing
        x are concatenated once into a single BF16 matrix (exact, see
        dequant_e8m0_bf16) and issued as one mm; the caller slices the row.
        Returns None when a weight is not canonical FP8/E8M0, so callers keep
        their per-name fallback.
        """
        if self.bf16 is None or x.dim() != 2 or not x.is_contiguous():
            return None
        if x.dtype != torch.bfloat16 or x.device != self.a.device:
            return None
        key = tuple(w.data_ptr() for w, _ in parts)
        cat = self.fp8.get(key)
        if (cat is None and x.shape[0] <= 32
                and not torch.cuda.is_current_stream_capturing()
                and all(packed_fp8_ok(w, sc, x.shape[-1]) for w, sc in parts)):
            # Concatenating on N keeps every 32x32 scale block intact, so the
            # packed weights stay canonical and no dequantized copy is needed.
            cat = (torch.cat([w.view(torch.uint8) for w, _ in parts], 0).contiguous(),
                   torch.cat([sc for _, sc in parts], 0).contiguous())
            self.fp8[key] = cat
        if cat is not None and x.shape[0] <= 32:
            self.fp8_calls += 1
            return self.mod.fp8_dense(x, cat[0], cat[1])
        wb = self.bf16.get(key)
        if wb is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            for w, s in parts:
                if (s is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2
                        or not w.is_contiguous() or w.shape[1] != x.shape[-1]
                        or s.dtype != torch.uint8 or not s.is_contiguous()):
                    return None
            wb = torch.cat([dequant_e8m0_bf16(w, s) for w, s in parts], 0)
            self.bf16[key] = wb
        out = torch.empty((x.shape[0], wb.shape[0]), device=x.device, dtype=x.dtype)
        torch.mm(x, wb.t(), out=out)
        self.bf16_calls += 1
        return out

    def __call__(self, x, weight, scale):
        if (x.ndim < 2 or x.dtype != torch.bfloat16 or x.device != self.a.device
                or not x.is_contiguous() or x.shape[-1] <= 0):
            raise ValueError('rank-local contiguous BF16 activation required')
        k = x.shape[-1]
        m = x.numel() // k
        if m > self.capacity or k > self.max_width or k % 32:
            raise ValueError('projection workspace capacity exceeded')
        if (torch.cuda.current_stream(x.device) != self.stream
                and not torch.cuda.is_current_stream_capturing()):
            # Capture runs on a private side stream and replays on the caller's
            # stream, so the scratch buffers stay serialised inside the graph.
            raise RuntimeError('projection workspace requires construction stream')
        if m <= 32 and packed_fp8_ok(weight, scale, k):
            # Packed FP8 straight into the tensor cores: half the weight bytes
            # of the BF16 copy, and the K-split factor depends only on the
            # shape, so batch width never changes the reduction order.
            self.fp8_calls += 1
            self.calls += 1
            return self.mod.fp8_dense(x.view(m, k), weight, scale).view(
                *x.shape[:-1], weight.shape[0])
        out = torch.empty((m, weight.shape[0]), device=x.device, dtype=x.dtype)
        if (self.bf16 is not None and weight.dtype == torch.float8_e4m3fn
                and weight.is_contiguous() and weight.dim() == 2 and weight.shape[1] == k
                and scale.dtype == torch.uint8 and scale.is_contiguous()):
            # Every shape goes through the BF16 tensor-core GEMM off a
            # dequantized copy made once per weight (exact, see
            # dequant_e8m0_bf16), as V4 does at load.  Dequantizing per call
            # was measured host-bound on prefill: each call allocated fresh
            # FP32/BF16 temporaries outside the cached pool.  The N256 FP32
            # reference GEMM below was 16x the BF16 time for M~5K and is kept
            # only as the fallback for weights this branch cannot take.
            wb = self.bf16.get(weight.data_ptr())
            if wb is None and not torch.cuda.is_current_stream_capturing():
                wb = dequant_e8m0_bf16(weight, scale)
                self.bf16[weight.data_ptr()] = wb
            if wb is not None:
                torch.mm(x.view(m, k), wb.t(), out=out)
                self.bf16_calls += 1
                self.calls += 1
                return out.view(*x.shape[:-1], weight.shape[0])
        # A packed-FP8 GEMV branch used to sit here (ops/prefill/cuda/fp8_dense_gemv.cu,
        # deleted): re-wired and measured at decode it cost +2.8ms/step against the
        # BF16 path below, so halving the weight bytes does not pay for losing tensor
        # cores at M<=8.
        self.mod.projection(x.view(m,k), weight, scale, self.a, self.w, self.p, out)
        self.calls += 1
        return out.view(*x.shape[:-1], weight.shape[0])
