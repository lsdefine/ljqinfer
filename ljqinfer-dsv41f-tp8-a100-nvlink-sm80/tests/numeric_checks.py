"""Comparisons that survive the SWA quantizer's discontinuous index rounding.

Measured on the released-shape random model (length=129, float32 diagnostic
chain), fused sparse attention vs the float32 sparse reference:

    quantizer removed : max|diff| ~7e-7, 0% of elements outside 2e-4+2e-3|ref|
    quantizer active  : logits relL2 8.4e-4 (max 7.6e-4), hidden relL2 6.6e-3
                        (max 1.6e-2), 3.8%/19.9% of elements outside that band

So the kernels agree to fp32 noise; the spread appears only once window scores
are quantized to fp8, where near-ties let the two implementations keep
different tokens. Element-wise assert_close therefore either fails or needs an
atol so wide it stops catching wiring errors. Bound instead the ENERGY of the
disagreement (relative L2) and its worst single element, and require the
argmax to be untouched - a real mis-wiring moves all three at once.
"""
import torch


def relative_l2(actual, reference):
    a, r = actual.float(), reference.float()
    return ((a - r).pow(2).sum().sqrt() / r.pow(2).sum().sqrt().clamp_min(1e-12)).item()


def assert_close_modulo_index_rounding(actual, reference, *, rel_l2, max_abs):
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    a, r = actual.float(), reference.float()
    assert torch.isfinite(a).all(), 'non-finite values in actual'
    got, worst = relative_l2(a, r), (a - r).abs().max().item()
    assert got <= rel_l2, f'relative L2 {got:.3g} exceeds index-rounding budget {rel_l2:g}'
    assert worst <= max_abs, f'worst deviation {worst:.3g} exceeds {max_abs:g}'
    return got, worst


def assert_parity_modulo_index_rounding(actual, reference):
    """Full acceptance check for a replay/finish result against its reference."""
    assert torch.equal(actual.logits.argmax(-1), reference.logits.argmax(-1)), \
        'index rounding must not change the selected token'
    assert_close_modulo_index_rounding(actual.logits, reference.logits,
                                       rel_l2=5e-3, max_abs=1e-2)
    assert_close_modulo_index_rounding(actual.main_hidden, reference.main_hidden,
                                       rel_l2=3e-2, max_abs=5e-2)
