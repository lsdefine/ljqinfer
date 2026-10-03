"""Released V4.1 text-backbone configuration; runtime capacities are separate."""
import json
import math
from pathlib import Path
import torch


def released_config():
    return json.loads(Path(__file__).with_name('v41_config.json').read_text())


def validate_config(config):
    expected = released_config()
    for name, value in expected.items():
        if name not in config or config[name] != value:
            raise ValueError(f'released V4.1 config mismatch: {name}')


def rotary_frequencies(config, layer, length, *, device):
    """Official YaRN rule; pure SWA disables extrapolation and uses its own theta."""
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
    angles = torch.outer(torch.arange(length, device=device), f)
    return torch.polar(torch.ones_like(angles), angles)
