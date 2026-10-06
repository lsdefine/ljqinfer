"""Eager attention semantic operators, independent of Past and slot allocation.

Inputs/weights are read-only except compressor carries. Results are owned
current-stream tensors; no cached request addresses. `linear` is a bound
weight projection callable, `norms` only contains this attention's RMS weights.
"""
import torch
from . import attention as a, residual as r


def attention_prepare(x, freqs, *, linear, norms, eps, heads, head_dim,
                      index_heads, index_dim, needs_index, library=None):
    """BF16[T,D] -> q BF16[T,H,Dh], kv BF16[T,Dh], iq/iw or None.

    Index Q uses released RoPE/QDQ; index weights accumulate as FP32.
    """
    qr = r.rms(linear('wq_a', x), norms['q'], eps)
    q = linear('wq_b', qr).view(len(x), heads, head_dim)
    q = a.rope(q, freqs, library=library)
    kv = a.rope(r.rms(linear('wkv', x), norms['kv'], eps), freqs, library=library)
    kv = a.qdq(kv, 'local', library=library)
    iq = iw = None
    if needs_index:
        iq = linear('i_wq_b', qr).view(len(x), index_heads, index_dim)
        iq = a.qdq(a.rope(iq, freqs, library=library), 'index', library=library)
        iw = linear('i_weights', x).float()
    return q, kv, iq, iw


def source_append(x, *, linear, norm_weight, index_norm_weight, eps, ratio,
                  carry_kv, carry_score, start, frequencies, library=None):
    """Append-only source transform -> (compressed KV, index keys) or None.

    x contains ONLY newly appended BF16 rows. carry_{kv,score} are one slot's
    FP32 compressor views, mutated for ratio2 including odd tails. `start` is
    absolute input position. Model writes returned rows into reserved pages,
    selects slot and commits position. No page allocation occurs here.
    """
    values = linear('c_wkv', x).float()
    if ratio == 2:
        scores = linear('c_wgate', x).float()
        raw = a.compress(values, scores, carry_kv, carry_score, start, library=library)
    else:
        raw = values
    if not len(raw):
        return None
    pos = torch.arange(start//ratio, (start+len(x))//ratio,
                       device=x.device, dtype=torch.int64) * ratio
    freqs = frequencies(pos)
    latent = r.rms(raw.to(torch.bfloat16), norm_weight, eps).to(torch.bfloat16)
    index = r.rms(linear('i_wk', latent), index_norm_weight, eps)
    latent = a.qdq(a.rope(latent, freqs, library=library), 'compressed', library=library)
    index = a.qdq(a.rope(index, freqs, library=library), 'index', library=library)
    return latent, index


def attention_finish(out, freqs, *, linear, comm, dtype, library=None):
    """Inverse RoPE, low-rank output projection, FP32 TP sum, output rounding."""
    out = a.rope(out, freqs, inverse=True, library=library)
    projected = linear('wo_a', out.flatten(1))
    result = linear('wo_b', projected).float()
    comm.sum(result)
    return result.to(dtype)


# These semantic entry points already encapsulate score tiling/TP and pack/
# correction. Keep a single implementation, including CED's existing callers.
index_select = a.select
sparse_attention = a.attend
