# CPU unit tests for model/past.py (SlotPool): per-slot pos, cold export/import
# round-trip with tail state, step-wise vs chunk residue equivalence.
# Synced 2026-09-04 with the current past.py API: residue is a 4*ratio ring
# addressed by absolute token id (write_res_t), and res() takes [t0, t1).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.past import SlotPool, IndexedPast, RATIO_COMPRESSED, PAGE_TOKENS

torch.manual_seed(0)
S, CAP, KD, D = 2, 4096, 512, 4096


def mk():
    pool = SlotPool(S, CAP)
    pool.add_layer(0, 0, kv_dim=KD)
    pool.add_layer(1, RATIO_COMPRESSED, kv_dim=KD, dim=D)
    pool.add_layer(2, IndexedPast.ratio, kv_dim=KD, dim=D)
    return pool


def fill(pool, slot, n):
    """Write n tokens into every layer of `slot` (all derived state included)."""
    r4, rc = IndexedPast.ratio, RATIO_COMPRESSED
    pool.ensure(slot, n)
    kv = torch.randn(n, KD)
    for p in pool.layers.values():
        p.write_kv(slot, 0, kv)
    pool.layers[1].write_ckv(slot, 0, torch.randn(n // rc, KD))
    pool.layers[2].write_ckv(slot, 0, torch.randn(n // r4, KD))
    pool.layers[2].write_ickv(slot, 0, torch.randn(n // r4, 128))
    for j in range(n):  # r=128 layer keeps no residue (open window = compressor carry)
        pool.layers[2].write_res_t(slot, torch.tensor([j]), torch.randn(1, D))
    pool.advance(slot, n)
    return kv


def same_state(a, sa, b, sb, n):
    for l in a.layers:
        pa, pb = a.layers[l], b.layers[l]
        assert torch.equal(pa.kv(sa, 0, n), pb.kv(sb, 0, n)), f"kv layer {l}"
        if l:
            assert torch.equal(pa.ckv(sa, n), pb.ckv(sb, n)), f"ckv layer {l}"
    assert torch.equal(a.layers[2].ickv(sa, n), b.layers[2].ickv(sb, n))
    m = a.layers[2].res_len(n)  # resumable residue window only
    assert torch.equal(a.layers[2].res(sa, n - m, n), b.layers[2].res(sb, n - m, n))


def test_per_slot_pos_independent():
    pool = mk()
    kv0 = fill(pool, 0, 300)
    kv1 = fill(pool, 1, 130)
    assert pool.pos[0] == 300 and pool.pos[1] == 130
    assert torch.equal(pool.layers[0].kv(0, 0, 300), kv0)
    assert torch.equal(pool.layers[0].kv(1, 0, 130), kv1)


def test_cold_roundtrip_single_segment():
    n = 2 * RATIO_COMPRESSED  # cold boundaries are 128-aligned by design
    a = mk(); fill(a, 0, n)
    seg = a.export_cold(0, 0)  # t1 defaults to pos -> carries tail
    b = mk()
    assert b.import_cold(1, [seg]) == n
    assert b.pos[1] == n
    same_state(a, 0, b, 1, n)


def test_cold_roundtrip_chunked_segments():
    n = 3 * RATIO_COMPRESSED
    a = mk(); fill(a, 0, n)
    segs = [a.export_cold(0, 0, RATIO_COMPRESSED),
            a.export_cold(0, RATIO_COMPRESSED, 2 * RATIO_COMPRESSED),
            a.export_cold(0, 2 * RATIO_COMPRESSED)]
    assert "tail" not in segs[0] and "tail" in segs[-1]
    b = mk()
    assert b.import_cold(0, segs) == n
    same_state(a, 0, b, 0, n)


def test_indexed_res_step_vs_chunk():
    r = IndexedPast.ratio
    xs = torch.randn(203, D)
    a = mk().layers[2]; b = mk().layers[2]
    for j in range(203):
        a.write_res_t(0, torch.tensor([j]), xs[j:j + 1])
    n = a.res_len(203)
    assert n == r + 203 % r
    b.write_res(0, 203 - n, xs[203 - n:])
    assert torch.equal(a.res(0, 203 - n, 203), b.res(0, 203 - n, 203))


def test_indexed_res_step_window_shift():
    r = IndexedPast.ratio
    xs = torch.randn(2 * r + 1, D)
    p = mk().layers[2]
    for j in range(2 * r + 1):
        p.write_res_t(0, torch.tensor([j]), xs[j:j + 1])
    pos = 2 * r + 1
    assert p.res_len(pos) == r + 1
    # ring of 4r rows indexed by absolute token: no wrap for 2r+1 tokens
    assert torch.equal(p.res_x[0, :2 * r + 1], xs)
    # resumable view = closed window (tokens r..2r) + open tail (token 2r)
    assert torch.equal(p.res(0, pos - (r + 1), pos), xs[r:2 * r + 1])


def test_alloc_release():
    pool = mk()
    s0, s1 = pool.alloc(), pool.alloc()
    assert {s0, s1} == {0, 1}
    pool.release(s0)
    assert pool.alloc() == s0
    assert pool.pos[s0] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(fns)} PASS")
