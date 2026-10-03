"""How far apart, in units of the last bit, do two batch widths land?

    torchrun --standalone --nproc-per-node=8 -m tests.ulp_probe

Texts diverge when a batch is wider, but a diverged text says nothing about
how large the numeric gap is: one bit at the bottom of a bf16 is enough to
swap two nearly tied candidates.  This measures the gap itself.  The same
prompt is stepped once alone and once beside three copies of itself, and
every float tensor the step hands back is compared bit by bit: how many
entries moved, and by how many representable steps.
"""
import torch

from strategy.decode_worker import MAX_BATCH, bootstrap
from tests.batch_probe import PROMPT, _prefill


def _fields(state):
    out = {}
    for name in dir(state):
        if name.startswith('_'):
            continue
        try:
            v = getattr(state, name)
        except Exception:
            continue
        if torch.is_tensor(v) and v.is_floating_point():
            out[name] = v
    return out


def _ulps(a, b):
    """Distance in representable steps of the local exponent."""
    x, y = a.float(), b.float()
    big = torch.maximum(x.abs(), y.abs())
    # spacing of the dtype at that magnitude: 2**exponent * 2**-mantissa_bits
    mant = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}[a.dtype]
    step = torch.ldexp(torch.ones_like(big), torch.floor(torch.log2(
        big.clamp_min(torch.finfo(a.dtype).tiny))).int() - (mant - 1))
    return (x - y).abs() / step


def once(eng, width):
    past = eng.past
    slots = [past.alloc() for _ in range(width)]
    states = []
    for s in slots:
        st, _ = _prefill(eng, s, PROMPT)
        states.append(st)
    states, ems = eng.spec.step_gb(states, past=past, slots=tuple(slots))
    kept = {k: v[:1].clone() if v.dim() and v.shape[0] >= width else v.clone()
            for k, v in _fields(states[0]).items()}
    tok = [int(t) for t in ems[0]]
    for s in slots:
        past.release(s)
    return kept, tok


def main():
    with torch.inference_mode():
        eng = bootstrap()
        ref, rtok = once(eng, 1)
        if torch.distributed.get_rank() == 0:
            print('ULP fields: %s' % sorted(ref), flush=True)
        for width in (2, MAX_BATCH):
            got, gtok = once(eng, width)
            if torch.distributed.get_rank() != 0:
                continue
            print('ULP width=%d tokens_equal=%s' % (width, gtok == rtok), flush=True)
            for name in sorted(ref):
                a, b = ref[name], got[name]
                if a.shape != b.shape:
                    print('ULP   %-16s shape %s vs %s' % (name, tuple(a.shape), tuple(b.shape)))
                    continue
                d = _ulps(a, b)
                n = int((d > 0).sum())
                print('ULP   %-16s %s n=%d/%d moved  max=%.2f ulp  worst_abs=%.3e  scale=%.3e'
                      % (name, a.dtype, n, d.numel(), float(d.max()),
                         float((a.float() - b.float()).abs().max()),
                         float(a.float().abs().max())), flush=True)


if __name__ == '__main__':
    main()
