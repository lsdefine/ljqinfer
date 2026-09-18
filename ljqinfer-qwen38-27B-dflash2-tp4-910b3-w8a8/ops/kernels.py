"""Simple, readable operator backend used by the first complete engine.

All methods are replaceable behind ``KernelBackend``. They favor transparent
PyTorch/Torch-NPU formulas over fused custom kernels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import ctypes
import os
import math
from pathlib import Path

_GDN_CONV_LIB = None
_GDN_L2_PAIR_LIB = None
_GDN_L2_PAIR_BATCH_LIB = None
_GDN_RECURRENT_BASEPTR_LIB = None
_GDN_RECURRENT_B1_BASEPTR_LIB = None
_DFLASH_GROUPED_CONV_LIB = None
_QK_RMS_NORM_ROPE_LIB = None
_QK_RMS_NORM_ROPE_BATCH_LIB = None
_ROPE_INV_FREQ_CACHE = {}
_RMS_EFFECTIVE_WEIGHT_CACHE = {}


def _effective_rms_weight(weight):
    key = id(weight)
    cached = _RMS_EFFECTIVE_WEIGHT_CACHE.get(key)
    if cached is None or cached[0] is not weight:
        cached = (weight, weight + 1.0)
        _RMS_EFFECTIVE_WEIGHT_CACHE[key] = cached
    return cached[1]

@dataclass
class KernelBackend:
    eps: float = 1e-6

    @staticmethod
    def _torch():
        import torch
        return torch

    def embedding(self, input_ids, weight):
        return self._torch().nn.functional.embedding(input_ids, weight)

    def rms_norm(self, x, weight, eps: float | None = None):
        """Qwen3.5 RMSNorm, whose checkpoint weight is a zero-centered delta."""
        torch = self._torch()
        eps = self.eps if eps is None else eps
        effective_weight = _effective_rms_weight(weight)
        if x.device.type == "npu":
            import torch_npu
            return torch_npu.npu_rms_norm(x, effective_weight, epsilon=eps)[0]
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        out = x.float() * torch.rsqrt(var + eps)
        return (out * effective_weight.float()).to(x.dtype)

    def residual_rms_norm(self, x, residual, weight, eps: float | None = None):
        residual = x if residual is None else residual + x
        return self.rms_norm(residual, weight, eps), residual

    def add_rms_norm(self, x, residual, weight, eps: float | None = None):
        """Fuse residual add with Qwen3.5 delta-weight RMSNorm on NPU."""
        eps = self.eps if eps is None else eps
        effective_weight = _effective_rms_weight(weight)
        if x.device.type == "npu":
            import torch_npu
            normed, _rstd, summed = torch_npu.npu_add_rms_norm(
                x, residual, effective_weight, epsilon=eps)
            return normed, summed
        summed = x + residual
        return self.rms_norm(summed, weight, eps), summed

    def w8a8_linear(self, x, linear):
        torch = self._torch()
        w, scale = linear.weight, linear.scale
        if x.device.type == "npu":
            import torch_npu
            qx, xscale = torch_npu.npu_dynamic_quant(x)
            # npu_quant_matmul consumes KxN weight.
            native_weight = linear.native_weight
            if native_weight is None:
                native_weight = w.transpose(0, 1).contiguous()
            return torch_npu.npu_quant_matmul(
                qx, native_weight, scale,
                offset=linear.offset, pertoken_scale=xscale,
                output_dtype=torch.bfloat16)
        # CPU/reference path dequantizes per output channel.
        wf = w.float() * scale.float().reshape(-1, 1)
        if linear.offset is not None:
            wf = wf + linear.offset.float().reshape(-1, 1)
        return torch.nn.functional.linear(x.float(), wf).to(x.dtype)

    def bf16_linear(self, x, weight):
        return self._torch().nn.functional.linear(x, weight)

    def silu(self, x):
        return self._torch().nn.functional.silu(x)

    def swiglu_mlp(self, x, mlp, all_reduce=None):
        torch = self._torch()
        if mlp.gate_up is not None:
            packed = self.w8a8_linear(x, mlp.gate_up)
            gate, up = packed.chunk(2, dim=-1)
        else:
            gate = self.w8a8_linear(x, mlp.gate)
            up = self.w8a8_linear(x, mlp.up)
        if x.device.type == "npu":
            import torch_npu
            if mlp.gate_up is not None and mlp.down.native_weight is not None:
                qact, act_scale = torch_npu.npu_dequant_swiglu_quant(
                    packed, activate_left=True, quant_mode=1)
                out = torch_npu.npu_quant_matmul(
                    qact, mlp.down.native_weight, mlp.down.scale,
                    offset=mlp.down.offset, pertoken_scale=act_scale,
                    output_dtype=torch.bfloat16)
                return all_reduce(out) if all_reduce is not None else out
            fused_input = packed if mlp.gate_up is not None else torch.cat((gate, up), dim=-1)
            activated = torch_npu.npu_swiglu(fused_input, dim=-1)
        else:
            activated = self.silu(gate) * up
        out = self.w8a8_linear(activated, mlp.down)
        return all_reduce(out) if all_reduce is not None else out

    def rope_frequencies(self, positions, rotary_dim: int, theta: float, dtype):
        """Build the dynamic cos/sin table shared by all attention layers."""
        torch = self._torch()
        if rotary_dim == 0:
            return None
        key = (str(positions.device), rotary_dim, float(theta))
        inv = _ROPE_INV_FREQ_CACHE.get(key)
        if inv is None:
            inv = 1.0 / (theta ** (torch.arange(
                0, rotary_dim, 2, device=positions.device,
                dtype=torch.float32) / rotary_dim))
            _ROPE_INV_FREQ_CACHE[key] = inv
        freq = positions.float().reshape(-1, 1) * inv.reshape(1, -1)
        return freq.cos().to(dtype), freq.sin().to(dtype)

    def apply_rope(self, x, frequencies, rotary_dim: int):
        """Apply NeoX-style partial RoPE while preserving the non-rotary tail."""
        if rotary_dim == 0:
            return x
        torch = self._torch()
        cos, sin = frequencies
        xr, tail = x[..., :rotary_dim], x[..., rotary_dim:]
        left, right = xr.chunk(2, dim=-1)
        c, s = cos[:, None, :], sin[:, None, :]
        rot = torch.cat((left * c - right * s,
                         right * c + left * s), dim=-1)
        return torch.cat((rot, tail), dim=-1)

    def rope(self, q, k, positions, rotary_dim: int, theta: float):
        """Convenience wrapper for call sites that cannot share frequencies."""
        if rotary_dim == 0:
            return q, k
        frequencies = self.rope_frequencies(
            positions, rotary_dim, theta, q.dtype)
        return (self.apply_rope(q, frequencies, rotary_dim),
                self.apply_rope(k, frequencies, rotary_dim))

    def qk_rms_norm_rope_decode(self, q, k, q_weight, k_weight,
                                frequencies, rotary_dim: int,
                                eps: float | None = None):
        """Fuse Q/K RMSNorm and partial NeoX RoPE for fixed B1-B4 Q8 decode."""
        torch = self._torch()
        eps = self.eps if eps is None else eps
        if frequencies is None:
            return (self.rms_norm(q, q_weight, eps),
                    self.rms_norm(k, k_weight, eps))
        cos, sin = frequencies
        token_count = q.shape[0] if q.ndim == 3 else 0
        native = (
            q.device.type == "npu" and k.device == q.device
            and q_weight.device == q.device and k_weight.device == q.device
            and cos.device == q.device and sin.device == q.device
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and q_weight.dtype == torch.bfloat16
            and k_weight.dtype == torch.bfloat16
            and cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
            and token_count in (8, 16, 24, 32)
            and q.shape == (token_count, 6, 256)
            and k.shape == (token_count, 1, 256)
            and q_weight.shape == (256,) and k_weight.shape == (256,)
            and cos.shape == (token_count, 32)
            and sin.shape == (token_count, 32)
            and q.stride(-1) == 1 and k.stride(-1) == 1
            and q_weight.is_contiguous() and k_weight.is_contiguous()
            and cos.is_contiguous() and sin.is_contiguous()
            and rotary_dim == 64 and eps == 1e-6
        )
        if not native:
            q = self.rms_norm(q, q_weight, eps)
            k = self.rms_norm(k, k_weight, eps)
            return (self.apply_rope(q, frequencies, rotary_dim),
                    self.apply_rope(k, frequencies, rotary_dim))

        global _QK_RMS_NORM_ROPE_LIB, _QK_RMS_NORM_ROPE_BATCH_LIB
        batch_native = token_count > 8
        fn = (_QK_RMS_NORM_ROPE_BATCH_LIB if batch_native
              else _QK_RMS_NORM_ROPE_LIB)
        if fn is None:
            name = ("ljq_qk_rms_norm_rope_aiv.so" if batch_native
                    else "ljq_qk_rms_norm_rope.so")
            so = Path(__file__).resolve().parent / "native" / name
            if not so.is_file():
                raise RuntimeError(f"missing native Q/K RMSNorm+RoPE kernel: {so}")
            lib = ctypes.CDLL(str(so))
            fn = lib.qk_norm_rope_launch
            fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 8
            fn.restype = ctypes.c_int
            if batch_native:
                _QK_RMS_NORM_ROPE_BATCH_LIB = fn
            else:
                _QK_RMS_NORM_ROPE_LIB = fn
        out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        rc = fn(
            torch.npu.current_stream(q.device)._as_parameter_,
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(q_weight.data_ptr()),
            ctypes.c_void_p(k_weight.data_ptr()),
            ctypes.c_void_p(cos.data_ptr()), ctypes.c_void_p(sin.data_ptr()),
            ctypes.c_void_p(out_q.data_ptr()), ctypes.c_void_p(out_k.data_ptr()),
            q.shape[0] * q.shape[1], k.shape[0] * k.shape[1],
            q.shape[1], k.shape[1], q.stride(0), q.stride(1),
            k.stride(0), k.stride(1))
        if rc:
            raise RuntimeError(f"qk_norm_rope_launch rc={rc}")
        return out_q, out_k

    def l2norm(self, x, eps: float = 1e-6):
        torch = self._torch()
        return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True).clamp_min(eps)).to(x.dtype)

    def paired_l2norm_decode(self, q, k):
        """Normalize packed BxQ8 GDN q/k rows with the native decode ABI."""
        torch = self._torch()
        valid = (
            q.device.type == "npu" and k.device.type == "npu"
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and q.ndim == 3 and q.shape == k.shape
            and q.shape[0] in (8, 16, 24, 32) and q.shape[-1] == 128
            and q.stride(-1) == 1 and k.stride(-1) == 1
            and q.stride(-2) == 128 and k.stride(-2) == 128
            and q.stride(0) == k.stride(0)
        )
        if not valid:
            raise ValueError(
                f"paired_l2norm_decode requires packed NPU bf16 [B*8,H,128], B=1..4, "
                f"got q={q.shape}/{q.stride()} k={k.shape}/{k.stride()}")

        rows = q.numel() // q.shape[-1]
        global _GDN_L2_PAIR_LIB, _GDN_L2_PAIR_BATCH_LIB
        batch_native = rows > 48
        fn = _GDN_L2_PAIR_BATCH_LIB if batch_native else _GDN_L2_PAIR_LIB
        if fn is None:
            name = ("ljq_gdn_l2_pair_aiv.so" if batch_native
                    else "ljq_gdn_l2_pair.so")
            so = Path(__file__).resolve().parent / "native" / name
            if not so.is_file():
                raise RuntimeError(f"missing native GDN L2 pair kernel: {so}")
            lib = ctypes.CDLL(str(so))
            fn = lib.gdn_l2_pair_launch
            fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 4
            fn.restype = ctypes.c_int
            if batch_native:
                _GDN_L2_PAIR_BATCH_LIB = fn
            else:
                _GDN_L2_PAIR_LIB = fn
        out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        heads = q.shape[-2]
        rc = fn(
            torch.npu.current_stream(q.device)._as_parameter_,
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(out_q.data_ptr()), ctypes.c_void_p(out_k.data_ptr()),
            rows, heads, q.stride(0), rows * 2)
        if rc:
            raise RuntimeError(f"gdn_l2_pair_launch rc={rc}")
        return out_q, out_k

    def dflash_grouped_conv_b1q8(self, hidden, delta, base, side, out=None):
        """Run the fixed Q8/C5120/tap2/group16 DFlash decode convolution."""
        torch = self._torch()
        valid = (
            hidden.device.type == "npu" and delta.device == hidden.device
            and base.device == hidden.device and hidden.dtype == torch.bfloat16
            and delta.dtype == torch.bfloat16 and base.dtype == torch.bfloat16
            and hidden.shape == (8, 5120) and hidden.stride() == (5120, 1)
            and delta.shape == (8, 2, 320)
            and delta.stride() == (1280, 320, 1)
            and base.shape == (2, 2, 5120)
            and base.stride() == (10240, 5120, 1)
            and side in (0, 1)
        )
        if not valid:
            raise ValueError(
                "dflash_grouped_conv_b1q8 requires same-device NPU bf16 "
                f"hidden[8,5120] contiguous, delta[8,2,320]/stride(1280,320,1), "
                f"base[2,2,5120] contiguous, side 0/1; got "
                f"hidden={hidden.shape}/{hidden.stride()}/{hidden.dtype}/{hidden.device} "
                f"delta={delta.shape}/{delta.stride()}/{delta.dtype}/{delta.device} "
                f"base={base.shape}/{base.stride()}/{base.dtype}/{base.device} side={side}")

        global _DFLASH_GROUPED_CONV_LIB
        if _DFLASH_GROUPED_CONV_LIB is None:
            so = (Path(__file__).resolve().parent / "ascendc" /
                  "libdflash_grouped_conv_b1q8.so")
            if not so.is_file():
                raise RuntimeError(f"missing DFlash grouped-conv kernel: {so}")
            lib = ctypes.CDLL(str(so))
            fn = lib.dflash_grouped_conv_b1q8_launch
            fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32]
            fn.restype = ctypes.c_int
            _DFLASH_GROUPED_CONV_LIB = fn
        if out is None:
            out = torch.empty_like(hidden)
        elif (out.device != hidden.device or out.dtype != hidden.dtype
              or out.shape != hidden.shape or out.stride() != hidden.stride()):
            raise ValueError(
                "dflash_grouped_conv_b1q8 out must match hidden shape/stride/dtype/device")
        rc = _DFLASH_GROUPED_CONV_LIB(
            torch.npu.current_stream(hidden.device)._as_parameter_,
            ctypes.c_void_p(hidden.data_ptr()), ctypes.c_void_p(delta.data_ptr()),
            ctypes.c_void_p(base.data_ptr()), ctypes.c_void_p(out.data_ptr()), side)
        if rc:
            raise RuntimeError(f"dflash_grouped_conv_b1q8_launch rc={rc}")
        return out

    def causal_conv_decode(self, x, base_state, pending_state, weight_kc):
        """Run the fixed-width BxQ8/C2560/K4 native convolution ABI."""
        torch = self._torch()
        batch = int(x.shape[0]) if x.ndim == 3 else 0
        if (x.device.type != "npu" or x.dtype != torch.bfloat16
                or batch not in (1, 2, 3, 4) or x.shape != (batch, 8, 2560)
                or base_state.shape != (batch, 3, 2560)
                or pending_state.shape != x.shape
                or weight_kc.shape != (4, 2560)):
            raise ValueError(
                "causal_conv_decode requires NPU bf16 x/pending=[B,8,2560], "
                "base=[B,3,2560], B=1..4, weight_kc=[4,2560]")
        if x.stride(-1) != 1:
            raise ValueError("causal_conv_decode requires unit input channel stride")
        if not base_state.is_contiguous() or not pending_state.is_contiguous():
            raise ValueError(
                "causal_conv_decode requires contiguous base and pending states")
        global _GDN_CONV_LIB
        if _GDN_CONV_LIB is None:
            so = Path(__file__).resolve().parent / "native" / "ljq_gdn_conv.so"
            if not so.is_file():
                raise RuntimeError(f"missing native GDN convolution kernel: {so}")
            lib = ctypes.CDLL(str(so))
            fn = lib.gdn_conv_vec_launch
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 9
            _GDN_CONV_LIB = fn
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        rc = _GDN_CONV_LIB(
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
            ctypes.c_void_p(base_state.data_ptr()), ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight_kc.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(pending_state.data_ptr()), 2560, x.stride(-2),
            x.stride(0), base_state.stride(0), out.stride(0),
            pending_state.stride(0), 640, 4, batch)
        if rc:
            raise RuntimeError(f"gdn_conv_vec_launch rc={rc}")
        if batch > 1:
            pending_state.copy_(x)
        return out

    def gated_delta_decode(self, q, k, v, g, beta_logits, state,
                           actual_seq_lengths, ssm_state_indices,
                           num_accepted_tokens, base_state=None):
        """Advance B independent fixed-width Q8 rows from BF16 beta logits."""
        import torch
        rows = int(q.shape[0]) if q.ndim == 3 else 0
        batch = rows // 8 if rows % 8 == 0 else 0
        if (batch not in (1, 2, 3, 4) or k.shape != q.shape or v.ndim != 3
                or v.shape[0] != rows or g.shape[0] != rows
                or beta_logits.shape[0] != rows or state.shape[0] != rows
                or ssm_state_indices.numel() != rows
                or num_accepted_tokens.numel() != batch):
            raise ValueError("gated_delta_decode requires packed B*8 rows, B=1..4")
        fixed_bq8 = (
            q.device.type == "npu" and q.dtype == torch.bfloat16
            and q.shape == (rows, 4, 128) and k.shape == q.shape
            and v.shape == (rows, 12, 128) and v.dtype == torch.bfloat16
            and beta_logits.shape == (rows, 12) and beta_logits.dtype == torch.bfloat16
            and g.shape == (rows, 12) and g.dtype == torch.float32
            and state.shape == (rows, 12, 128, 128)
            and state.dtype == torch.bfloat16
            and ssm_state_indices.shape == (rows,)
            and ssm_state_indices.dtype == torch.int32
            and num_accepted_tokens.shape == (batch,)
            and num_accepted_tokens.dtype == torch.int32
            and all(x.device == q.device for x in (
                k, v, g, beta_logits, state, ssm_state_indices, num_accepted_tokens))
            and all(x.is_contiguous() for x in (
                q, k, g, beta_logits, state, ssm_state_indices, num_accepted_tokens))
            and v.stride(-1) == 1 and v.stride(-2) == 128
            and v.stride(0) >= 12 * 128
        )
        if fixed_bq8:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
            stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
            if batch == 1:
                if (base_state is None
                        or base_state.shape != (batch, 12, 128, 128)
                        or base_state.dtype != torch.bfloat16
                        or base_state.device != q.device
                        or not base_state.is_contiguous()):
                    raise ValueError("batched fixed GDN requires contiguous base_state[B,12,128,128]")
                global _GDN_RECURRENT_B1_BASEPTR_LIB
                if _GDN_RECURRENT_B1_BASEPTR_LIB is None:
                    so = (Path(__file__).resolve().parent / "native"
                          / "ljq_gdn_recurrent_b1_baseptr_aiv.so")
                    if not so.is_file():
                        raise RuntimeError(f"missing native batched GDN base-pointer kernel: {so}")
                    lib = ctypes.CDLL(str(so))
                    fn = lib.gdn_recurrent_b1base_direct_launch
                    fn.restype = ctypes.c_int
                    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint32] * 2
                    _GDN_RECURRENT_B1_BASEPTR_LIB = fn
                rc = _GDN_RECURRENT_B1_BASEPTR_LIB(
                    stream, ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(g.data_ptr()),
                    ctypes.c_void_p(beta_logits.data_ptr()), ctypes.c_void_p(base_state.data_ptr()),
                    ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_void_p(ssm_state_indices.data_ptr()),
                    ctypes.c_void_p(num_accepted_tokens.data_ptr()), v.stride(0), batch)
                name = "gdn_recurrent_b1base_direct_launch"
            else:
                if (base_state is None
                        or base_state.shape != (batch, 12, 128, 128)
                        or base_state.dtype != torch.bfloat16
                        or base_state.device != q.device
                        or not base_state.is_contiguous()):
                    raise ValueError("batched fixed GDN requires contiguous base_state[B,12,128,128]")
                global _GDN_RECURRENT_BASEPTR_LIB
                if _GDN_RECURRENT_BASEPTR_LIB is None:
                    so = (Path(__file__).resolve().parent / "native"
                          / "ljq_gdn_recurrent_baseptr_aiv.so")
                    if not so.is_file():
                        raise RuntimeError(f"missing native batched GDN base-pointer kernel: {so}")
                    lib = ctypes.CDLL(str(so))
                    fn = lib.gdn_recurrent_baseptr_launch
                    fn.restype = ctypes.c_int
                    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint32] * 2
                    _GDN_RECURRENT_BASEPTR_LIB = fn
                rc = _GDN_RECURRENT_BASEPTR_LIB(
                    stream, ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                    ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(g.data_ptr()),
                    ctypes.c_void_p(beta_logits.data_ptr()), ctypes.c_void_p(base_state.data_ptr()),
                    ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_void_p(ssm_state_indices.data_ptr()),
                    ctypes.c_void_p(num_accepted_tokens.data_ptr()), v.stride(0), batch)
                name = "gdn_recurrent_baseptr_launch"
            if rc:
                raise RuntimeError(f"{name} rc={rc}")
            return out
        raise ValueError("gated_delta_decode requires the fixed NPU BF16 B1..B4 Q8 ABI")

    def causal_conv_prefill(self, x, state, weight):
        """Vectorized depthwise causal conv for one contiguous prefill sequence.

        ``x`` is [B,T,C] and ``state`` is the in/out [B,K-1,C] hot state.
        Unlike ``causal_conv_sequence`` this retains only the final state.
        """
        torch = self._torch()
        kernel = weight.shape[-1]
        sequence = torch.cat((state, x), dim=1)
        w = weight.reshape(weight.shape[0], kernel)
        # Avoid Ascend's expensive broadcast over an ``unfold`` view.
        # These aligned slices are the identical depthwise correlation.
        tokens = x.shape[1]
        out = sequence[:, :tokens] * w[:, 0]
        for tap in range(1, kernel):
            out = out + sequence[:, tap:tap + tokens] * w[:, tap]
        state.copy_(sequence[:, -(kernel - 1):])
        return out

    def chunk_gated_delta_sequence(self, q, k, v, g, beta, state,
                                   chunk_size: int = 64):
        """Chunked Qwen3.5 GDN for one or more aligned sequences.

        q/k=[B,T,Hk,Dk], v=[B,T,Hv,Dv], g/beta=[B,T,Hv], and the
        production hot state is [B,Hv,Dv,Dk]. T must be chunk-aligned.
        Preprocessing stays fp32; native H/O tensors use bf16.
        """
        import torch
        from pathlib import Path
        if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
            raise ValueError("chunk GDN expects q/k=[B,T,Hk,Dk], v=[B,T,Hv,Dv]")
        batch, tokens, key_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if tokens <= 0 or tokens % chunk_size:
            raise ValueError(f"chunk GDN token count must be a positive multiple of {chunk_size}")
        if value_heads % key_heads:
            raise ValueError("value heads must be divisible by key heads")
        if state.shape != (batch, value_heads, value_dim, key_dim):
            raise ValueError(f"unexpected production GDN state shape {tuple(state.shape)}")
        if not hasattr(torch.ops, "ljq_chunk") or not hasattr(torch.ops.ljq_chunk, "chunk_h"):
            so = Path(__file__).resolve().parent / "native" / "ljq_chunk_ext.so"
            if not so.is_file():
                raise RuntimeError(f"missing chunk GDN extension: {so}")
            torch.ops.load_library(str(so))

        q = self.l2norm(q)
        k = self.l2norm(k)
        qh = q.transpose(1, 2).contiguous()
        kh = k.transpose(1, 2).contiguous()
        vh = v.transpose(1, 2).contiguous()
        gh = g.transpose(1, 2).float().contiguous()
        bh = beta.transpose(1, 2).float().contiguous()
        chunks = tokens // chunk_size
        repeat = value_heads // key_heads
        kg = kh.repeat_interleave(repeat, dim=1).reshape(
            batch, value_heads, chunks, chunk_size, key_dim).float()
        gc = gh.reshape(batch, value_heads, chunks, chunk_size).cumsum(-1)
        bc = bh.reshape(batch, value_heads, chunks, chunk_size)
        vc = vh.reshape(batch, value_heads, chunks, chunk_size, value_dim).float()
        prepare_backend = os.environ.get(
            "LJQ_GDN_PREPARE_BACKEND", "aot").strip().lower()
        if prepare_backend not in ("aot", "torch"):
            raise ValueError(
                f"unknown LJQ_GDN_PREPARE_BACKEND={prepare_backend!r}")
        aot_shape = (
            q.device.type == "npu" and chunk_size == 64 and batch == 1 and
            value_heads == 12 and key_dim == 128 and value_dim == 128 and
            tokens <= 12288)
        if prepare_backend == "aot" and aot_shape:
            from .gdn_aot import prepare_wy
            u, w = prepare_wy(kg, vc, bc, gc)
            u, w = u.to(v.dtype), w.to(k.dtype)
        else:
            gram = torch.matmul(kg, kg.transpose(-1, -2))
            decay = torch.exp(gc[..., :, None] - gc[..., None, :])
            lower = torch.tril(
                bc[..., :, None] * gram * decay, diagonal=-1)
            eye = torch.eye(
                chunk_size, dtype=torch.float32, device=q.device)
            eye = eye.reshape(
                1, 1, 1, chunk_size, chunk_size).expand_as(lower)
            system = eye + lower
            # Recursively invert the unit lower-triangular 64x64 matrix.
            inv = eye.clone()
            width = 1
            while width < chunk_size:
                blocks = chunk_size // (2 * width)
                size = system.shape[:-2] + (
                    blocks, 2 * width, 2 * width)
                system_stride = system.stride()[:-2] + (
                    2 * width * (chunk_size + 1), chunk_size, 1)
                inverse_stride = inv.stride()[:-2] + (
                    2 * width * (chunk_size + 1), chunk_size, 1)
                system_blocks = torch.as_strided(
                    system, size=size, stride=system_stride)
                inverse_blocks = torch.as_strided(
                    inv, size=size, stride=inverse_stride)
                a21 = system_blocks[..., width:, :width]
                b11 = inverse_blocks[..., :width, :width]
                b22 = inverse_blocks[..., width:, width:]
                inverse_blocks[..., width:, :width].copy_(
                    -torch.matmul(torch.matmul(b22, a21), b11))
                width *= 2
            u = torch.matmul(inv, bc[..., None] * vc)
            w = torch.matmul(
                inv, (bc * torch.exp(gc))[..., None] * kg)
            u = u.reshape(
                batch, value_heads, tokens, value_dim).to(v.dtype)
            w = w.reshape(
                batch, value_heads, tokens, key_dim).to(k.dtype)
        gc_flat = gc.reshape(batch, value_heads, tokens)
        initial = state.transpose(-1, -2).contiguous()
        h, v_new, final_state = torch.ops.ljq_chunk.chunk_h(
            kh, w, u, gc_flat, initial, True, chunk_size, None, None)
        out = torch.ops.ljq_chunk.chunk_o(
            qh, kh, v_new, h, key_dim ** -0.5, gc_flat, None, None, chunk_size)
        state.copy_(final_state.transpose(-1, -2))
        output = out.transpose(1, 2).contiguous()
        return output, h


    def rmsnorm_gated(self, x, z, weight, eps: float = 1e-6):
        """GDN gated RMSNorm uses a direct, one-centered weight (no +1)."""
        torch = self._torch()
        if x.device.type == "npu":
            import torch_npu
            normed = torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
            return torch_npu.npu_swiglu(
                torch.cat((z, normed), dim=-1), dim=-1)
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)
        gate = torch.nn.functional.silu(z.float())
        return (normed.float() * gate).to(x.dtype)

K = KernelBackend()
