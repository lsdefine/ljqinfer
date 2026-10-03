"""Publish a decode window's KV rows into the pools with one launch.

The rows go to two places: the head of each slot's band (``front``, so that
logical row ids stay contiguous for the kernel) and the sliding ring
(``pos % ring``).  Expressed in torch that is nine launches -- a long multiply
for the band base, an add and a modulo for the ring, an ``arange`` rebuilt
every layer, a ``cat`` to prepend the staged rows and two ``index_copy_`` --
of which only the two copies move data.  Inside a captured decode graph each
launch costs ~2us of pure latency, so the addressing is done in the kernel
instead: one program per (request, row, column block) computes both
destinations itself.
"""
import triton
import triton.language as tl


@triton.jit
def _publish_rows(FLAT, ROWS, POS, KV, ST, rps, pad, ring,
                  NS: tl.constexpr, QWIN: tl.constexpr,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    req, j = tl.program_id(0), tl.program_id(1)
    off = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = off < D
    base = tl.load(ROWS + req).to(tl.int64) * rps
    if j < NS:
        src = tl.load(ST + (req * NS + j) * D + off, mask=mask)
    else:
        src = tl.load(KV + (req * QWIN + (j - NS)) * D + off, mask=mask)
    src = src.to(FLAT.dtype.element_ty)
    tl.store(FLAT + (base + j) * D + off, src, mask=mask)
    if j >= NS:
        pos = tl.load(POS + req * QWIN + (j - NS)).to(tl.int64)
        tl.store(FLAT + (base + pad + pos % ring) * D + off, src, mask=mask)


def publish(window, rows, pos, kv, staged, nreq, qwin, ns):
    """Write ``kv`` (and any ``staged`` rows before it) into ``window``."""
    flat = window.main_kv.view(-1, window.main_kv.shape[-1])
    d = flat.shape[-1]
    kv = kv.contiguous()
    if ns:
        staged = staged.contiguous()
    block = min(triton.next_power_of_2(d), 1024)
    _publish_rows[(nreq, ns + qwin, triton.cdiv(d, block))](
        flat, rows, pos, kv, staged if ns else kv,
        window.main_kv.shape[1], window.pad, window.ring,
        ns, qwin, d, block, num_warps=4)
