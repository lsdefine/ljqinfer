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
    """FFN 相位的 rank 本地部分，返回 pre-AllReduce partial [Q,D]。
    三种层型: 稠密FFN(层0-2) / 普通MoE(fused GEMV) / special层MoE(packed quantized leaf)。
    ws = DecodeGraph: 一切输出写入预分配 ws_* 缓冲(graph-safe, 无新分配)。
    """
    from model.weights import DenseFFN, MoE  # local import: weights is stable

    Q = h.shape[0]
    if isinstance(layer_w, DenseFFN):
        hc = h.contiguous()
        g = K.q8_matmul(hc, layer_w.gate[r], LOCAL_FF)
        u = K.q8_matmul(hc, layer_w.up[r], LOCAL_FF)
        act = (F.silu(g) * u).contiguous()
        return K.q8_matmul(act, layer_w.down[r], D)

    moe: MoE = layer_w

    # 共享专家: shared_decode_q8_inplace (TN weight-reuse, T=1..32)
    #   x half [Q,D] -> h_ws half [Q,H_shared] -> yf fp32 [Q,D/TP局部]
    h_ws, yf = ws.ws_shared_h[r], ws.ws_shared_y[r]
    K.iq.shared_decode_q8_inplace(
        moe.gate_shexp[r], moe.up_shexp[r], moe.down_shexp[r],
        h.contiguous(), h_ws, yf)

    eid, ew = _route_local(K, h, moe, r, ws=ws)   # int64/fp32 [Q,8]

    routed = ws.ws_xf[r]                                  # fp32 [Q,D] leaf output
    if layer in SPECIAL_LAYERS:
        x32 = ws.ws_x32[r]
        x32.copy_(h)
        # Packed special MoE keeps its own explicit-out leaf.
        cfg = SPECIAL_CFG[layer]
        if cfg == (2, 5):
            op = K.special_quant.special_iq4xs_q5k
        elif cfg == (1, 6):
            op = K.special_quant.special_iq3xxs_q6k
        elif cfg == (3, 4):
            op = K.special_quant.special_q3k_q4k
        else:
            raise RuntimeError(f"unsupported special quant config layer={layer} cfg={cfg}")
        op(moe.gate_exps[r], moe.up_exps[r], moe.down_exps[r],
           x32, eid, ew, routed)
    else:
        # Ordinary MoE writes directly into caller-owned graph workspaces.
        # Startup attaches persistent selected-expert Down rows before capture.
        down_cache = getattr(moe, "down_cache", None)
        down_cache_map = getattr(moe, "down_cache_map", None)
        if down_cache is None or down_cache_map is None:
            raise RuntimeError(f"routed Down cache missing before decode capture: layer={layer}")
        K.down_cache.decode(
            moe.gate_exps[r], moe.up_exps[r], moe.down_exps[r],
            h, eid, ew, ws.ws_routed_h[r], routed,
            down_cache[r], down_cache_map[r])
    out = ws.partial[r]                                    # fp16 [Q,D] collective partial
    K.residual_add_copy.combine_moe_f32_to_f16(routed, yf, out)
    return out


def decode_special_moe_rank_dualstream(
    K, moe, h: torch.Tensor, layer: int, r: int, ws, root_stream,
    aux_stream, fork_event, done_event,
) -> torch.Tensor:
    """Special routed MoE with shared/routed branches overlapped."""
    cfg = SPECIAL_CFG[layer]
    if cfg == (2, 5):
        op = K.special_quant.special_iq4xs_q5k
    elif cfg == (1, 6):
        op = K.special_quant.special_iq3xxs_q6k
    elif cfg == (3, 4):
        op = K.special_quant.special_q3k_q4k
    else:
        raise RuntimeError(
            f"unsupported special quant config layer={layer} cfg={cfg}")

    fork_event.record(root_stream)
    aux_stream.wait_event(fork_event)
    with torch.cuda.stream(aux_stream):
        h_ws, yf = ws.ws_shared_h[r], ws.ws_shared_y[r]
        K.iq.shared_decode_q8_inplace(
            moe.gate_shexp[r], moe.up_shexp[r], moe.down_shexp[r],
            h.contiguous(), h_ws, yf)
        done_event.record(aux_stream)

    eid, ew = _route_local(K, h, moe, r, ws=ws)
    x32 = ws.ws_x32[r]
    x32.copy_(h)
    routed = ws.ws_xf[r]
    op(moe.gate_exps[r], moe.up_exps[r], moe.down_exps[r],
       x32, eid, ew, routed)

    root_stream.wait_event(done_event)
    out = ws.partial[r]
    K.residual_add_copy.combine_moe_f32_to_f16(routed, yf, out)
    return out


def decode_moe_rank_dualstream(
    K, moe, h: torch.Tensor, r: int, ws, root_stream,
    aux_stream, fork_event, done_event,
) -> torch.Tensor:
    """Ordinary routed MoE with shared/routed branches overlapped across streams."""
    fork_event.record(root_stream)
    aux_stream.wait_event(fork_event)
    with torch.cuda.stream(aux_stream):
        h_ws, yf = ws.ws_shared_h[r], ws.ws_shared_y[r]
        K.iq.shared_decode_q8_inplace(
            moe.gate_shexp[r], moe.up_shexp[r], moe.down_shexp[r],
            h.contiguous(), h_ws, yf)
        done_event.record(aux_stream)

    eid, ew = _route_local(K, h, moe, r, ws=ws)
    routed = ws.ws_xf[r]
    down_cache = getattr(moe, "down_cache", None)
    down_cache_map = getattr(moe, "down_cache_map", None)
    if down_cache is None or down_cache_map is None:
        raise RuntimeError("routed Down cache missing before decode capture")
    K.down_cache.decode(
        moe.gate_exps[r], moe.up_exps[r], moe.down_exps[r],
        h, eid, ew, ws.ws_routed_h[r], routed,
        down_cache[r], down_cache_map[r])

    root_stream.wait_event(done_event)
    out = ws.partial[r]
    K.residual_add_copy.combine_moe_f32_to_f16(routed, yf, out)
    return out


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
