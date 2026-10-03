"""V4.1 canonical weight ABI: TP8 experts, TP8 projections, host Engram.

No dequant/requant. Kernel swizzles must use a new pack ABI; cache their FINAL
outputs, so engine restarts never repeat slicing, stacking or prepacking.
"""
from dataclasses import dataclass
import re
import torch

LAYOUT_ABI = 'v41-tp8moe-host-engram-v1'
WORLD = 8


@dataclass(frozen=True)
class Placement:
    kind: str
    axis: int | None = None
    owner: int | None = None
    local_expert: int | None = None


def placement(name: str) -> Placement:
    """Unknown parameter names fail closed instead of silently replicating."""
    if name in {'embed.weight', 'head.weight'} or re.fullmatch(
            r'mtp\.2\.markov_head\.(embed|head)\.weight', name):
        return Placement('tp', 0)
    if name in {'norm.weight', 'image_start', 'image_end', 'image_newline'}:
        return Placement('replicated')
    # Explicit vision BF16 baseline; not silently inherited text TP rules.
    if re.fullmatch(r'(vision\.(patch_embed\.proj\.(weight|bias)|norm\.weight|'
                    r'blocks\.\d+\.(norm[12]\.weight|attn\.(wqkv|wo)\.(weight|bias)|'
                    r'mlp\.w[12]\.weight))|aligner\.w[12]\.(weight|bias))', name):
        return Placement('replicated')
    m = re.fullmatch(r'(layers|mtp)\.(\d+)\.(.+)', name)
    if not m:
        raise ValueError(f'unknown weight: {name}')
    group, layer, tail = m[1], int(m[2]), m[3]
    if layer >= (40 if group == 'layers' else 3):
        raise ValueError(f'layer out of range: {name}')
    expert = re.fullmatch(r'ffn\.experts\.(\d+)\.w([123])\.(weight|scale)', tail)
    if expert:
        count = 384 if group == 'layers' else 128
        eid = int(expert[1])
        if eid >= count:
            raise ValueError(f'expert out of range: {name}')
        # TP8 over the expert intermediate dim: gate/up keep whole rows, down
        # splits its reduction columns. Every rank owns every expert, so token
        # routing can never skew per-rank work; the cost is one extra all-reduce
        # of the down projection, which EP paid as a rank reduction anyway.
        return Placement('tp', 0 if expert[2] in ('1', '3') else 1,
                         local_expert=eid)
    if tail.startswith('engram.'):
        if group != 'layers' or layer not in (1, 14):
            raise ValueError(f'Engram on invalid layer: {name}')
        if tail in {'engram.embed.weight', 'engram.embed.scale'}:
            return Placement('host')
        if tail in {'engram.wkv.weight', 'engram.wkv.scale'}:
            return Placement('tp', 1)  # 24 hash rows -> 3 rows/rank; sum projections
        if tail in {'engram.q_weight', 'engram.k_weight'}:
            return Placement('replicated')
    if re.fullmatch(r'attn\.(wq_b|wo_a|indexer\.(wq_b|weights_proj))\.(weight|scale)', tail):
        return Placement('tp', 0)
    if tail == 'attn.attn_sink':
        return Placement('tp', 0)
    if re.fullmatch(r'attn\.wo_b\.(weight|scale)', tail):
        return Placement('tp', 1)
    if re.fullmatch(r'ffn\.shared_experts\.w[123]\.(weight|scale)', tail):
        return Placement('tp', 1 if '.w2.' in tail else 0)
    if re.fullmatch(r'(hc_(attn|ffn)_(base|fn|scale)|'
                    r'(attn_norm|ffn_norm)\.weight|ffn\.gate\.(weight|bias|bias_vl)|'
                    r'attn\.((wq_a|wkv)\.(weight|scale)|(q_norm|kv_norm)\.weight|'
                    r'compressor\.(wkv|wgate|norm)\.weight|indexer\.(wk|k_norm)\.weight))', tail):
        return Placement('replicated')
    if group == 'mtp' and ((layer == 0 and tail in {
            'main_proj.weight', 'main_proj.scale', 'main_norm.weight'}) or
            (layer == 2 and tail in {'norm.weight', 'confidence_head.proj.weight'})):
        return Placement('replicated')
    raise ValueError(f'unknown weight: {name}')


def shard(name: str, tensor: torch.Tensor, rank: int) -> torch.Tensor | None:
    """Byte-preserving base transform. Host table never enters rank payloads."""
    if not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    p = placement(name)
    if p.kind == 'host' or (p.kind == 'ep' and p.owner != rank):
        return None
    if p.kind == 'tp':
        if tensor.ndim <= p.axis or tensor.shape[p.axis] % WORLD:
            raise ValueError(f'non-divisible TP dimension: {name} {tuple(tensor.shape)}')
        n = tensor.shape[p.axis] // WORLD
        return tensor.narrow(p.axis, rank * n, n).contiguous()
    return tensor.contiguous()


def quant_pair(prefix: str, weight: torch.Tensor, scale: torch.Tensor,
               rank: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Validate FP4 K/32 or FP8 32x32 geometry before slicing either tensor.

    Scale bytes retain the checkpoint's E8M0 encoding; FP4 nibble order remains
    low-K then high-K. Host Engram uses a separate per-row scale layout.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError('quant pair must be matrices')
    n, k = weight.shape
    if weight.dtype == torch.int8:
        logical_k, block_n = k * 2, 1
    elif weight.dtype == torch.float8_e4m3fn:
        logical_k, block_n = k, 32
    else:
        raise ValueError(f'unsupported quantized dtype: {weight.dtype}')
    expected = ((n + block_n - 1) // block_n, (logical_k + 31) // 32)
    if tuple(scale.shape) != expected or scale.dtype != torch.uint8:
        raise ValueError(f'bad scale geometry/dtype: {prefix}: {tuple(scale.shape)} != {expected}')
    p = placement(prefix + '.weight')
    if p.kind == 'host':
        raise ValueError('host table is mapped once, not rank-sharded')
    if p.kind == 'tp':
        extent, block = (n, block_n) if p.axis == 0 else (logical_k, 32)
        if extent % (WORLD * block):
            raise ValueError(f'TP boundary cuts quantization block: {prefix}')
    w, s = shard(prefix + '.weight', weight, rank), shard(prefix + '.scale', scale, rank)
    return None if w is None else (w, s)


def pack_experts(tensors: dict[str, torch.Tensor], prefix: str, rank: int):
    """Complete TP bank: every expert, this rank's slice of the intermediate.

    w13[E,2(gate,up),I/8,K/2], w2[E,K,I/16], s13[E,2,I/8,K/32], s2[E,K,I/256].
    Canonical FP4 bank; no kernel swizzle.
    Incomplete banks fail instead of filling missing experts with zeros.
    """
    if not 0 <= rank < WORLD or not re.fullmatch(r'(layers\.\d+|mtp\.[0-2])\.ffn', prefix):
        raise ValueError('invalid bank/rank')
    count = 128 if prefix.startswith('mtp.') else 384
    pieces = {c: [] for c in ('w1', 'w3', 'w2')}
    for eid in range(count):
        for c in pieces:
            key = f'{prefix}.experts.{eid}.{c}'
            w, s = tensors[key + '.weight'], tensors[key + '.scale']
            if w.dtype != torch.int8:
                raise ValueError('expert bank requires packed FP4 int8 bytes')
            pieces[c].append(quant_pair(key, w, s, rank))
        w1, w3, w2 = [pieces[c][-1][0] for c in ('w1', 'w3', 'w2')]
        if w1.shape != w3.shape or w2.shape != (w1.shape[1] * 2, w1.shape[0] // 2):
            raise ValueError('incompatible expert gate/up/down shapes')
    result = {}
    for index, suffix in enumerate(('weight', 'scale')):
        result['w13.' + suffix] = torch.stack([
            torch.stack([pieces['w1'][e][index], pieces['w3'][e][index]]) for e in range(count)])
        result['w2.' + suffix] = torch.stack([p[index] for p in pieces['w2']])
    return result


def prepare_rank(tensors: dict[str, torch.Tensor], rank: int):
    """Prepare a complete unit, not necessarily the entire checkpoint.

    Layer/expert bank/embedding units enable incremental build during download.
    Output is directly cacheable; warm startup need not revisit source weights.
    """
    if not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    result, banks = {}, set()
    for name, tensor in tensors.items():
        p = placement(name)
        if p.kind == 'host':
            continue
        if p.local_expert is not None:
            banks.add(name.split('.experts.')[0])
            continue
        if name.endswith('.scale'):
            if name[:-5] + 'weight' not in tensors:
                raise ValueError(f'orphan scale: {name}')
            continue
        if name.endswith('.weight') and tensor.dtype in (torch.int8, torch.float8_e4m3fn):
            prefix = name[:-7]
            pair = quant_pair(prefix, tensor, tensors[prefix + '.scale'], rank)
            result[name], result[prefix + '.scale'] = pair
        else:
            result[name] = shard(name, tensor, rank)
    for prefix in sorted(banks):
        for name, tensor in pack_experts(tensors, prefix, rank).items():
            result[prefix + '.local_experts.' + name] = tensor
    return result
