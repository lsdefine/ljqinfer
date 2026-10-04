"""Released V4.1 text backbone: the released constants plus the frozen tables
the prefill assembly specialises on.

Rotary tables live in ops/prefill/rope.table; this module only states what the
released config is, so every role and shape baked into the assembly has one
checkable source.
"""
import json
from dataclasses import dataclass
from pathlib import Path

PREFILL_LAYERS = 20
CHUNK = 12288
WINDOW = 128


def released_config():
    return json.loads(Path(__file__).with_name('v41_config.json').read_text())


def validate_config(config):
    expected = released_config()
    for name, value in expected.items():
        if name not in config or config[name] != value:
            raise ValueError(f'released V4.1 config mismatch: {name}')


@dataclass(frozen=True)
class LayerRole:
    """What a prefill layer is, decided once from the released constants."""
    layer: int
    role: str          # swa | source | reuse
    source: int        # layer owning the compressed history (-1 for swa)
    ratio: int         # compressor fold ratio (0 for swa)


def layer_roles(config):
    """Freeze the role of every prefill layer.

    Released V4.1: compress_ratios is 0 on layers 0-1 (pure sliding window),
    2 on layers 2-19, and kv_source_layers/index_source_layers agree on
    2, 8, 14 inside the prefill half.  Every index source below layer 20 is
    also a KV source, so no layer here reindexes, and candidate_source_layer
    is 20, so no prefill layer builds candidates.
    """
    ratios, kv = config['compress_ratios'], config['kv_source_layers']
    index = config['index_source_layers']
    roles = []
    for layer in range(PREFILL_LAYERS):
        ratio = ratios[layer]
        if ratio == 0:
            roles.append(LayerRole(layer, 'swa', -1, 0))
            continue
        if ratio != 2:
            raise ValueError(f'prefill layer {layer} is not a ratio-2 layer')
        source = max(s for s in kv if s <= layer)
        if layer in kv:
            roles.append(LayerRole(layer, 'source', layer, ratio))
        elif layer in index:
            raise ValueError(f'layer {layer} reindexes inside the prefill half')
        else:
            roles.append(LayerRole(layer, 'reuse', source, ratio))
    if config['candidate_source_layer'] < PREFILL_LAYERS:
        raise ValueError('candidates are built inside the prefill half')
    return tuple(roles)


def rotary_frequencies(config, layer, length, *, device, positions=None):
    """Released YaRN reference table, kept for the released-math tests.

    The assembly itself builds its tables through ops/prefill/rope.table; this
    stays because tests/test_prefill_released.py and tests/released_random.py
    state the released rule here and compare against it.
    """
    import math
    import torch
    d = config['rope_head_dim']
    compressed = config['compress_ratios'][layer] != 0
    base = config['compress_rope_theta'] if compressed else config['rope_theta']
    original = config['original_seq_len'] if compressed else 0
    f = 1 / (base ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    if original > 0:
        def correction(rotations):
            return d * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(config['beta_fast'])), 0)
        high = min(math.ceil(correction(config['beta_slow'])), d-1)
        ramp = ((torch.arange(d//2, device=device, dtype=torch.float32)-low) / max(high-low, 1e-3)).clamp(0, 1)
        f = f / config['rope_factor'] * ramp + f * (1-ramp)
    positions = torch.arange(length, device=device) if positions is None else positions.clamp_min(0)
    angles = torch.outer(positions, f)
    return torch.polar(torch.ones_like(angles), angles)
