"""V4.1 canonical weight ABI: TP8 everywhere, including routed experts.

The released checkpoint is W4A8. Routed experts pack two int4 output channels
into one byte along rows; every other quantized tensor stays plain int8. The
float32 companions are indexed by output channel, so splitting the contraction
axis replicates them instead of slicing them. They are cached under the engine's
own suffixes (.scale), not the checkpoint's, so kernels keep one vocabulary. Nothing
is dequantized here: kernels consume the packed bytes, and this module only
slices, stacks and prepackages routed INT4 into NZ. Cache the FINAL outputs, so engine restarts never repeat
slicing, stacking or prepacking.
"""
from dataclasses import dataclass
import re

import torch

from .routed_layout import pack_projection

LAYOUT_ABI = 'v41-tp8-w4a8-routed-nz-k64-host-engram-v3'
WORLD = 8
COMPANIONS = {'weight_scale': 'scale', 'scale_bias': 'scale_bias'}
_Q = r'weight'


@dataclass(frozen=True)
class Placement:
    kind: str
    axis: int | None = None


def placement(name: str) -> Placement:
    """Unknown parameter names fail closed instead of silently replicating."""
    if name.endswith('.weight_offset'):
        # Symmetric-checkpoint zero points: quant_group verifies they are zero and
        # drops them, so they never reach a device. Names are still validated.
        placement(name[:-len('.weight_offset')] + '.weight')
        return Placement('dropped')
    for suffix in COMPANIONS:
        if name.endswith('.' + suffix):
            base = placement(name[:-len(suffix) - 1] + '.weight')
            # Companions are indexed by output channel, so a row-parallel group,
            # which is split along K, keeps them whole on every rank.
            if base.kind == 'tp' and base.axis == 1:
                return Placement('replicated')
            return base
    if name in {'embed.weight', 'head.weight'}:
        return Placement('tp', 0)
    if name == 'norm.weight' or name.startswith(('vision.', 'aligner.', 'image_')):
        return Placement('replicated')
    block = re.fullmatch(r'(layers|mtp)\.(\d+)\.(.+)', name)
    if not block:
        raise ValueError(f'unknown weight: {name}')
    group, tail = block.group(1), block.group(3)
    if group == 'mtp':
        if re.fullmatch(r'(embed|head|markov_head\.(embed|head))\.weight', tail):
            return Placement('tp', 0)
        if re.fullmatch(rf'(norm|main_norm)\.weight|(main_proj|confidence_head\.proj)\.{_Q}', tail):
            return Placement('replicated')
    if group == 'layers':
        if re.fullmatch(r'engram\.embed\.(weight|scale)', tail):
            return Placement('host')
        if tail == 'engram.wkv.weight':
            return Placement('tp', 1)
        if re.fullmatch(r'engram\.[qk]_weight', tail):
            return Placement('replicated')
    if tail == 'attn.attn_sink' or re.fullmatch(
            rf'(attn\.(wq_b|wo_a|indexer\.(wq_b|weights_proj))'
            rf'|ffn\.(shared_experts|experts\.\d+)\.(w1|w3))\.{_Q}', tail):
        return Placement('tp', 0)
    if re.fullmatch(rf'(attn\.wo_b|ffn\.(shared_experts|experts\.\d+)\.w2)\.{_Q}', tail):
        return Placement('tp', 1)
    if re.fullmatch(rf'attn\.(wq_a|wkv)\.{_Q}', tail):
        return Placement('replicated')
    if re.fullmatch(r'attn\.(compressor\.(wkv|wgate|norm)|indexer\.(wk|k_norm))\.weight'
                    r'|(attn_norm|ffn_norm|attn\.(q_norm|kv_norm))\.weight'
                    r'|ffn\.gate\.(weight|bias|bias_vl)|hc_(attn|ffn)_(fn|base|scale)', tail):
        return Placement('replicated')
    raise ValueError(f'unknown weight: {name}')


def shard(name: str, tensor: torch.Tensor, rank: int) -> torch.Tensor | None:
    """Byte-preserving slice; host and dropped tensors report None, not a piece."""
    if not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    p = placement(name)
    if p.kind in ('host', 'dropped'):
        return None
    if p.kind == 'tp':
        if tensor.ndim <= p.axis or tensor.shape[p.axis] % WORLD:
            raise ValueError(f'non-divisible TP dimension: {name} {tuple(tensor.shape)}')
        n = tensor.shape[p.axis] // WORLD
        return tensor.narrow(p.axis, rank * n, n).contiguous()
    return tensor.contiguous()


def quant_group(prefix: str, tensors: dict[str, torch.Tensor], rank: int):
    """Slice a quantized weight together with its per-output-channel companions.

    int4 packing is proven by geometry rather than by name: two output channels
    share one byte, so the companions are exactly twice as tall as the stored
    weight. Symmetric quantization is mandatory; a nonzero offset would silently
    change results, so it is verified and dropped rather than carried forever.
    """
    weight = tensors[prefix + '.weight']
    scale = tensors[prefix + '.weight_scale']
    if weight.dtype != torch.int8 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f'quantized group must be int8 matrix plus matrices: {prefix}')
    rows, channels = weight.shape[0], scale.shape[0]
    if channels not in (rows, rows * 2):
        raise ValueError(f'scale does not index output channels: {prefix}')
    offset = tensors.get(prefix + '.weight_offset')
    if offset is not None and bool(offset.any()):
        raise ValueError(f'asymmetric quantization is unsupported: {prefix}')
    p = placement(prefix + '.weight')
    if p.kind == 'host':
        raise ValueError('host table is mapped once, not rank-sharded')
    if p.kind == 'tp' and p.axis == 0 and channels % WORLD:
        raise ValueError(f'TP boundary cuts an int4 channel pair: {prefix}')
    result = {prefix + '.weight': shard(prefix + '.weight', weight, rank)}
    for suffix in COMPANIONS:
        companion = tensors.get(prefix + '.' + suffix)
        if companion is None:
            continue
        if companion.dtype != torch.float32 or companion.shape[0] != channels:
            raise ValueError(f'companion is not per output channel: {prefix}.{suffix}')
        piece = channels // WORLD
        result[prefix + '.' + COMPANIONS[suffix]] = (
            companion.narrow(0, rank * piece, piece).contiguous()
            if p.kind == 'tp' and p.axis == 0 else companion.contiguous())
    return result


def pack_experts(tensors: dict[str, torch.Tensor], prefix: str, rank: int):
    """Complete TP bank of ALL experts, persisted in decode-compatible INT4 NZ.

    Weight rows carry two int4 output channels each while the float32 companions
    stay per channel, so w13 companions are sliced and w2 companions replicated.
    Every rank holds every expert and one intermediate slice; partial outputs are
    summed across ranks at run time. Incomplete banks fail instead of filling
    missing experts with zeros.
    """
    if not 0 <= rank < WORLD or not re.fullmatch(r'(layers\.\d+|mtp\.\d+)\.ffn', prefix):
        raise ValueError('invalid bank/rank')
    count = 128 if prefix.startswith('mtp.') else 384
    pieces = {c: [] for c in ('w1', 'w3', 'w2')}
    for expert in range(count):
        base = f'{prefix}.experts.{expert}.'
        for column in pieces:
            pieces[column].append(quant_group(base + column, tensors, rank))
        w1, w3, w2 = (pieces[c][-1][f'{base}{c}.weight'] for c in ('w1', 'w3', 'w2'))
        if w1.shape != w3.shape or w2.shape != (w1.shape[1] // 2, w1.shape[0] * 2):
            raise ValueError('incompatible expert gate/up/down shapes')
    result = {}
    # CANN uses scale plus computed hp_bias, never checkpoint scale_bias.
    for suffix in ('weight', 'scale'):
        result['w13.' + suffix] = torch.stack([
            torch.stack([pieces[c][e][f'{prefix}.experts.{e}.{c}.{suffix}']
                         for c in ('w1', 'w3')]) for e in range(count)])
        result['w2.' + suffix] = torch.stack([
            pieces['w2'][e][f'{prefix}.experts.{e}.w2.{suffix}'] for e in range(count)])
    # One persisted INT4 bank serves separate prefill/decode execution modules.
    # Conversion and correction-bias computation happen only at cache build.
    for projection in ('w13', 'w2'):
        weight, scale, bias = pack_projection(
            result[projection + '.weight'], result[projection + '.scale'])
        result[projection + '.weight'] = weight
        result[projection + '.scale'] = scale
        result[projection + '.hp_bias'] = bias
    return result


def validate_routed_cache(tensors):
    """Check the persisted NZ ABI using metadata only; never repack on load."""
    stems = set()
    for name in tensors:
        match = re.fullmatch(r'((?:layers|mtp)\.\d+\.ffn\.w(?:13|2))\..+', name)
        if match:
            if name.endswith('.scale_bias'):
                raise ValueError('obsolete routed scale_bias in NZ cache: ' + name)
            stems.add(match[1])
        if re.match(r'(?:layers|mtp)\.\d+\.ffn\.experts\.', name):
            raise ValueError('unpacked expert weights cannot enter the NZ cache')
    for stem in stems:
        experts = 128 if stem.startswith('mtp.') else 384
        k, n = (5120, 576) if stem.endswith('.w13') else (320, 5120)
        for suffix, shape, dtype in (
                ('weight', (experts, k, n//8), torch.int32),
                ('scale', (experts, n), torch.float32),
                ('hp_bias', (experts, n), torch.float32)):
            tensor = tensors.get(stem+'.'+suffix)
            if (tensor is None or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or not tensor.is_contiguous()):
                raise ValueError('invalid canonical NZ cache tensor: '+stem+'.'+suffix)


def prepare_rank(tensors: dict[str, torch.Tensor], rank: int):
    """Prepare a complete unit, not necessarily the entire checkpoint.

    Layer/expert bank/embedding units enable incremental build during download.
    Output is directly cacheable; warm startup need not revisit source weights.
    """
    if not 0 <= rank < WORLD:
        raise ValueError('rank must be 0..7')
    result, banks = {}, set()
    for name, tensor in tensors.items():
        if placement(name).kind in ('host', 'dropped'):
            continue
        if '.experts.' in name:
            banks.add(name.split('.experts.')[0])
            continue
        stem, _, suffix = name.rpartition('.')
        if suffix in COMPANIONS:
            if stem + '.weight' not in tensors:
                raise ValueError(f'companion without its weight: {name}')
            continue
        if suffix == 'weight' and stem + '.weight_scale' in tensors:
            result.update(quant_group(stem, tensors, rank))
            continue
        result[name] = shard(name, tensor, rank)
    for bank in sorted(banks):
        for suffix, value in pack_experts(tensors, bank, rank).items():
            result[f'{bank}.{suffix}'] = value
    return result