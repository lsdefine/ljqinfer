"""Frozen ljqinfer operator ABI backed by CUDA/Triton and FLA.

The historical w8a8_linear name accepts the BF16 payload supplied by the CUDA
loader. INT8 payloads are rejected, not silently dequantized in the hot path.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from . import cuda_ops as C


class KernelBackend:
    def dflash_path_select(self, scores, candidate):
        # Finite scores; ties select the lowest candidate index.
        if (candidate.ndim != 3 or candidate.shape[0] not in (1, 2, 3, 4)
                or candidate.shape[1:] != (7, 16)
                or scores.shape != (candidate.shape[0], 7, 16, 16)
                or scores.dtype not in (torch.bfloat16, torch.float32)
                or candidate.dtype != torch.int64
                or scores.device.type != 'cuda' or candidate.device != scores.device):
            raise ValueError('DFlash path requires CUDA scores [B1..4,7,16,16] and int64 candidates')
        return C.dflash_path_select(scores, candidate)

    eps: float = 1e-6

    @staticmethod
    def _torch():
        return torch

    @staticmethod
    def _bf16_cuda(*tensors):
        if not tensors or any(t.device.type != 'cuda' or t.device != tensors[0].device
                              or t.dtype != torch.bfloat16 for t in tensors):
            raise ValueError('CUDA leaves require same-device BF16 payloads')

    def logical_attention_metadata(self, positions, length):
        if (positions.ndim != 2 or positions.shape[1] != 8
                or positions.shape[0] not in (1, 2, 3, 4)
                or positions.dtype not in (torch.int32, torch.int64)
                or positions.device.type != 'cuda' or length < 8):
            raise ValueError('logical metadata requires CUDA integer [B1..4,8]')
        start = positions[:, 0].clamp(0, length - 8).contiguous()
        return start, C.logical_mask_packed(start, length)

    def logical_kv_gather(self, key, value, start):
        self._bf16_cuda(key, value)
        if (key.ndim != 4 or key.shape != value.shape
                or key.shape[0] not in (1, 2, 3, 4)
                or key.shape[1] < 8 or key.shape[2:] != (1, 256)
                or not key.is_contiguous() or not value.is_contiguous()
                or start.shape != (key.shape[0],) or start.device != key.device
                or start.dtype not in (torch.int32, torch.int64)
                or not start.is_contiguous()):
            raise ValueError('logical gather requires contiguous BF16 [B,L,1,256]')
        return C.logical_kv_gather(key, value, start)


    def dflash_qk_norm_rope(self, q, k, qw, kw, frequencies, eps=1e-6):
        # Functional B1..4 Q8 draft leaf; direct RMS weights, full RoPE.
        cos, sin = frequencies
        self._bf16_cuda(q, k, qw, kw, cos, sin)
        if (q.ndim != 3 or q.shape[0] not in (8, 16, 24, 32)
                or q.shape[1:] != (8, 128)
                or k.shape != (q.shape[0], 2, 128)
                or q.stride(-1) != 1 or k.stride(-1) != 1
                or qw.shape != (128,) or kw.shape != (128,)
                or not qw.is_contiguous() or not kw.is_contiguous()
                or cos.shape != (q.shape[0], 64) or sin.shape != cos.shape
                or not cos.is_contiguous() or not sin.is_contiguous()):
            raise ValueError('draft QK norm/RoPE requires Q8 B1..4, H8/H2 D128')
        return C.dflash_qk_norm_rope(q, k, qw, kw, frequencies, 128, eps)

    def embedding(self, input_ids, weight):
        return F.embedding(input_ids, weight)

    def rms_norm(self, x, weight, eps: float | None = None):
        return C.norm(x, weight, self.eps if eps is None else eps)

    def rms_norm_verify_packed(self, x, weight, eps: float | None = None):
        return C.norm_verify_packed(x, weight, self.eps if eps is None else eps)

    def add_rms_norm_verify_packed(self, x, residual, weight, eps: float | None = None):
        return C.norm_verify_packed(x, weight, self.eps if eps is None else eps,
                                    residual=residual)

    def residual_rms_norm(self, x, residual, weight, eps: float | None = None):
        if residual is None:
            return self.rms_norm(x, weight, eps), x
        return self.add_rms_norm(x, residual, weight, eps)

    def add_rms_norm(self, x, residual, weight, eps: float | None = None):
        return C.norm(x, weight, self.eps if eps is None else eps, residual=residual)

    def w8a8_linear(self, x, linear):
        hidden = x
        w = linear if isinstance(linear, torch.Tensor) else linear.weight
        if not w.is_floating_point():
            raise TypeError('CUDA BF16 ABI requires floating output-major weight; convert once in loader')
        return F.linear(hidden, w)

    def bf16_linear(self, x, weight):
        return F.linear(x, weight)

    def pack_verify_rows(self, x):
        self._bf16_cuda(x)
        return C.pack_verify_rows(x)

    def silu(self, x):
        return F.silu(x)

    def attention_gate_pack(self, y, gate):
        self._bf16_cuda(y, gate)
        if (y.ndim != 4 or y.shape != gate.shape
                or y.shape[0] not in (1, 2, 3, 4)
                or y.shape[1:] != (8, 6, 256)):
            raise ValueError('attention gate requires B1..4 Q8 H6 D256')
        return C.attention_gate_pack(y, gate)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def swiglu(self, gate, up):
        return C.swiglu(gate, up)

    def swiglu_verify_packed(self, gate, up):
        return C.swiglu_verify_packed(gate, up)

    def gated_norm_verify_packed(self, x, z, weight, eps=1e-6):
        return C.gated_norm_verify_packed(x, z, weight, eps)

    def swiglu_mlp(self, x, mlp, all_reduce=None):
        if mlp.gate_up is not None:
            gate, up = self.w8a8_linear(x, mlp.gate_up).chunk(2, dim=-1)
        else:
            gate = self.w8a8_linear(x, mlp.gate)
            up = self.w8a8_linear(x, mlp.up)
        out = self.w8a8_linear(self.swiglu(gate, up), mlp.down)
        return out if all_reduce is None else all_reduce(out)

    def rope_frequencies(self, positions, rotary_dim: int, theta: float, dtype):
        """Half-width NeoX cos/sin table, exactly as the frozen ABI defines."""
        if rotary_dim == 0:
            return None
        inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=positions.device,
                                            dtype=torch.float32) / rotary_dim))
        freq = positions.float().reshape(-1, 1) * inv.reshape(1, -1)
        return freq.cos().to(dtype), freq.sin().to(dtype)

    def apply_rope(self, x, frequencies, rotary_dim: int):
        if rotary_dim == 0:
            return x
        cos, sin = frequencies
        xr, tail = x[..., :rotary_dim], x[..., rotary_dim:]
        left, right = xr.chunk(2, dim=-1)
        c, s = cos[:, None, :], sin[:, None, :]
        rot = torch.cat((left * c - right * s, right * c + left * s), dim=-1)
        return torch.cat((rot, tail), dim=-1)

    def qk_rms_norm_rope_decode(self, q, k, q_weight, k_weight, frequencies,
                                rotary_dim: int, eps: float | None = None):
        eps = self.eps if eps is None else eps
        if frequencies is None or rotary_dim == 0:
            return self.rms_norm(q, q_weight, eps), self.rms_norm(k, k_weight, eps)
        self._bf16_cuda(q, k, q_weight, k_weight)
        if (q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]
                or q.shape[-1] != k.shape[-1] or q.stride(-1) != 1 or k.stride(-1) != 1
                or q_weight.shape != (q.shape[-1],) or k_weight.shape != q_weight.shape
                or not q_weight.is_contiguous() or not k_weight.is_contiguous()
                or rotary_dim < 0 or rotary_dim > q.shape[-1] or rotary_dim % 2):
            raise ValueError('QK norm/RoPE requires token-head-dim BF16 and even partial rotary dim')
        if any(f.shape != (q.shape[0], rotary_dim // 2) or f.device != q.device
               or f.dtype != q.dtype or not f.is_contiguous() for f in frequencies):
            raise ValueError('frequencies must be contiguous BF16 [T,rotary_dim/2] on the Q device')
        return C.qk_norm_rope(q, k, q_weight, k_weight, frequencies, rotary_dim, eps)

    def rope(self, q, k, positions, rotary_dim: int, theta: float):
        if rotary_dim == 0:
            return q, k
        frequencies = self.rope_frequencies(positions, rotary_dim, theta, q.dtype)
        return (self.apply_rope(q, frequencies, rotary_dim),
                self.apply_rope(k, frequencies, rotary_dim))

    def l2norm(self, x, eps: float = 1e-6):
        return C.norm(x, eps=eps, mode=1)

    def paired_l2norm_decode(self, q, k):
        return C.norm(q, eps=self.eps, mode=1), C.norm(k, eps=self.eps, mode=1)

    def dflash_grouped_conv_b1q8(self, hidden, delta, base, side, out=None):
        self._bf16_cuda(hidden, delta, base)
        if (hidden.shape != (8, 5120) or delta.shape != (8, 2, 320)
                or base.shape != (2, 2, 5120) or side not in (0, 1)
                or not hidden.is_contiguous() or not base.is_contiguous()
                or delta.stride(-1) != 1):
            raise ValueError('DFlash requires hidden[8,5120], delta[8,2,320], base[2,2,5120], side=0/1')
        if out is not None and (out.shape != hidden.shape or out.dtype != hidden.dtype
                              or out.device != hidden.device or not out.is_contiguous()):
            raise ValueError('out must match contiguous hidden')
        if out is not None and out.data_ptr() == hidden.data_ptr():
            raise ValueError('DFlash out may not alias hidden (cross-token read)')
        return C.grouped(hidden, delta, base, side, out)

    def gdn_gate_prepare(self, ab, decay, bias):
        self._bf16_cuda(ab)
        if (ab.ndim != 3 or ab.shape[0] not in (1, 2, 3, 4)
                or ab.shape[1:] != (8, 24) or not ab.is_contiguous()
                or any(t.shape != (12,) or t.dtype != torch.float32
                       or t.device != ab.device or not t.is_contiguous()
                       for t in (decay, bias))):
            raise ValueError('GDN gate requires BF16 [B1..4,8,24] and FP32 device vectors[12]')
        return C.gdn_gate_prepare(ab, decay, bias)

    def gdn_conv_norm_decode(self, x, base_state, pending_state, weight_kc):
        self._bf16_cuda(x, base_state, pending_state, weight_kc)
        b = x.shape[0]
        if (x.shape != (b, 8, 2560) or b not in (1, 2, 3, 4)
                or base_state.shape != (b, 3, 2560) or pending_state.shape != x.shape
                or weight_kc.shape != (4, 2560) or x.stride(-1) != 1
                or not base_state.is_contiguous() or not pending_state.is_contiguous()
                or not weight_kc.is_contiguous()):
            raise ValueError('GDN conv-norm requires Q8/C2560/K4 and contiguous BF16 states/weights')
        return C.gdn_conv_norm(x, base_state, weight_kc, pending_state)

    def causal_conv_decode(self, x, base_state, pending_state, weight_kc):
        self._bf16_cuda(x, base_state, pending_state, weight_kc)
        b = x.shape[0]
        if (x.shape != (b, 8, 2560) or b not in (1, 2, 3, 4)
                or base_state.shape != (b, 3, 2560) or pending_state.shape != x.shape
                or weight_kc.shape != (4, 2560) or x.stride(-1) != 1
                or not base_state.is_contiguous() or not pending_state.is_contiguous()):
            raise ValueError('conv requires B1..4,Q8,C2560,K4 and contiguous states')
        return C.conv(x, base_state, weight_kc, pending_state)

    def gated_delta_decode(self, q, k, v, g, beta_logits, state,
                           actual_seq_lengths, ssm_state_indices,
                           num_accepted_tokens, base_state=None):
        if base_state is None:
            raise ValueError('immutable BF16 base_state is required by the Q8 ABI')
        self._bf16_cuda(q, k, v, beta_logits, state, base_state)
        if g.device != q.device or g.dtype != torch.float32:
            raise ValueError('GDN decay g must be same-device FP32, as in the native ABI')
        if (any(t.stride(-1) != 1 for t in (q, k, v, g, beta_logits))
                or any(t.stride(1) != 128 for t in (q, k, v))
                or ssm_state_indices.device != q.device or not ssm_state_indices.is_contiguous()
                or ssm_state_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError('GDN needs unit inner strides and contiguous CUDA integer snapshot indices')
        rows = q.shape[0]
        b = rows // 8
        if (rows != b * 8 or b not in (1, 2, 3, 4) or q.shape != (rows, 4, 128)
                or k.shape != q.shape or v.shape != (rows, 12, 128)
                or g.shape != (rows, 12) or beta_logits.shape != g.shape
                or state.shape != (rows, 12, 128, 128)
                or base_state is None or base_state.shape != (b, 12, 128, 128)
                or not base_state.is_contiguous() or not state.is_contiguous()
                or ssm_state_indices.numel() != rows or num_accepted_tokens.numel() != b):
            raise ValueError('GDN requires immutable base[B,12,128,128], Q8 pending states and B1..4')
        # As in the base-pointer native ABI, actual lengths/accepted are reserved;
        # all Q8 tokens execute, and GPU indices select each snapshot destination.
        return C.recurrent(q, k, v, g, beta_logits, base_state, state, ssm_state_indices)

    def causal_conv_prefill(self, x, state, weight):
        self._bf16_cuda(x, state, weight)
        weight = weight.reshape(weight.shape[0], -1)
        if (x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1]
                or state.shape != (x.shape[0], weight.shape[1] - 1, x.shape[-1])
                or not state.is_contiguous() or x.stride(-1) != 1):
            raise ValueError('prefill conv requires X[B,T,C], state[B,K-1,C], weight[C,K]')
        return C.conv(x, state, weight)

    def chunk_gated_delta_sequence(self, q, k, v, g, beta, state, chunk_size: int = 64):
        # Public FLA only returns the final state. Reuse its same mature lower
        # kernels to expose chunk-entry checkpoints without recomputing prefixes.
        from fla.ops.gated_delta_rule.chunk import (
            chunk_gated_delta_rule_fwd_intra, chunk_gated_delta_rule_fwd_h, chunk_fwd_o)
        from fla.ops.utils import chunk_local_cumsum
        if (chunk_size != 64 or q.ndim != 4 or k.shape != q.shape
                or q.shape[1] <= 0 or q.shape[1] % 64 or v.shape[2] % q.shape[2]):
            raise ValueError('chunk GDN requires aligned T>0, chunk_size=64 and integral GVA')
        q, k = C.norm(q, eps=1e-6, mode=3), C.norm(k, eps=1e-6, mode=3)
        v = v.contiguous()
        beta = beta.float().contiguous()
        g = chunk_local_cumsum(g.float().contiguous(), chunk_size=64)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta)
        h, vn, final = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g,
            initial_state=state.float().contiguous(), output_final_state=True,
            transpose_state_layout=True)
        out = chunk_fwd_o(q=q, k=k, v=vn, h=h, g=g, scale=q.shape[-1] ** -0.5,
                         transpose_state_layout=True)
        state.copy_(final)
        # FLA B,NT,HV,Dv,Dk -> original B,HV,NT,Dk,Dv.
        # Consumers copy checkpoint slices; preserve strides instead of copying all chunks.
        return out, h.permute(0, 2, 1, 4, 3)

    def rmsnorm_gated(self, x, z, weight, eps: float = 1e-6):
        shape = x.shape
        return C.norm(x, weight, eps, mode=2, z=z).reshape(shape)

    def gated_delta_net(self, hidden, *, layer_idx: int, cache):
        if getattr(hidden, 'mock', False):
            return hidden
        raise RuntimeError('gated_delta_net needs loaded weights and state')

    def full_attention(self, hidden, *, layer_idx: int, cache):
        if getattr(hidden, 'mock', False):
            return hidden
        raise RuntimeError('full_attention needs loaded weights and KV cache')

    def mlp(self, hidden, *, layer_idx: int):
        if getattr(hidden, 'mock', False):
            return hidden
        raise RuntimeError('mlp needs loaded weights')


K = KernelBackend()
