"""decode 叶子块: attn / route / FFN(dense·MoE·special) — 全部 graph-safe 写入 ws_* 预分配缓冲。
K 由调用方传入 (ops.kernels.K)。
"""

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

from ops.kernels import K

from model.config import *  # noqa: F401,F403 — 本机固定常量, 全部写死


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * w.float()).to(dtype=x.dtype)


def _route_local(K, h: torch.Tensor, moe, r: int, ws):
    """路由: FP16 Tensor-Core projection -> FP32 logits -> grouped top-k [Q,8]."""
    logits = ws.ws_logits_r[r]
    # Keep a contiguous FP16 transpose per rank. torch.mm accumulates and writes
    # FP32 (out_dtype), avoiding the FP32 input copy and FP32 SGEMM path.
    # The cache is populated during warm-run, before CUDA Graph capture.
    cache = getattr(moe, '_router_f16t', None)
    if cache is None:
        cache = [None] * len(moe.router)
        object.__setattr__(moe, '_router_f16t', cache)
    w = cache[r]
    if w is None:
        w = moe.router[r].t().contiguous()
        cache[r] = w
    torch.mm(h, w, out=logits, out_dtype=torch.float32)
    bias = moe.bias[r]
    b = bias if bias.dtype == torch.float32 else bias.float()
    K.route.moe_route_t1_cuda(logits, b.view(-1), ws.ws_ei[r], ws.ws_ew[r])
    return ws.ws_ei[r], ws.ws_ew[r]


def decode_ffn_rank(K, layer_w, h: torch.Tensor, layer: int, r: int, ws) -> torch.Tensor:
    from model.weights import DenseFFN, MoE  # local import: weights is stable

    Q = h.shape[0]
    if isinstance(layer_w, DenseFFN):
        hc = h.contiguous()
        g = K.q8_matmul(hc, layer_w.gate[r], LOCAL_FF)
        u = K.q8_matmul(hc, layer_w.up[r], LOCAL_FF)
        act = (F.silu(g) * u).contiguous()
        return K.q8_matmul(act, layer_w.down[r], D)

    raise RuntimeError("Legacy GLM52 MoE retired; GLM53 weight/engine integration pending. Use ops.moe_decode or ops.moe_prefill.")


def decode_special_moe_rank_dualstream(
    K, moe, h: torch.Tensor, layer: int, r: int, ws, root_stream,
    aux_stream, fork_event, done_event,
) -> torch.Tensor:
    raise RuntimeError("Legacy GLM52 MoE retired; GLM53 weight/engine integration pending. Use ops.moe_decode or ops.moe_prefill.")


def decode_moe_rank_dualstream(
    K, moe, h: torch.Tensor, r: int, ws, root_stream,
    aux_stream, fork_event, done_event,
) -> torch.Tensor:
    raise RuntimeError("Legacy GLM52 MoE retired; GLM53 weight/engine integration pending. Use ops.moe_decode or ops.moe_prefill.")


def _capture_one(stream, body, *, keep_graph=True, pool=None):
    """accept4e-style: capture body() on stream without stock graph ctx GC tax."""
    graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    began = False
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin(pool=pool, capture_error_mode="thread_local")
            began = True
            body()
            graph.capture_end()
            began = False
        return graph
    except Exception:
        if began:
            try:
                graph.capture_end()
            except Exception:
                pass
        try:
            stream.synchronize()
        except Exception:
            pass
        try:
            graph.reset()
        except Exception:
            pass
        del graph
        gc.collect()
        raise
