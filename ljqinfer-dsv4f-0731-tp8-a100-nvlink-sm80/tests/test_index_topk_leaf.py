
import torch, sys
sys.path.insert(0, ".")
from ops import _mod
torch.manual_seed(0)
dev = "cuda"
def ref(q, kv, w, ratio, start_pos, offset, K):
    b, s = q.shape[:2]; end = start_pos + s
    sc = torch.einsum("bshd,btd->bsht", q, kv)
    sc = (sc.relu_() * w.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        mask = torch.arange(s // ratio, device=dev).repeat(s, 1) >= torch.arange(1, s + 1, device=dev).unsqueeze(1) // ratio
        sc += torch.where(mask, float("-inf"), 0)
    idx = sc.topk(min(K, end // ratio), dim=-1)[1]
    if start_pos == 0:
        mask = idx >= torch.arange(1, s + 1, device=dev).unsqueeze(1) // ratio
        idx = torch.where(mask, -1, idx + offset)
    else:
        idx += offset
    return sc, idx.int()
def leaf(q, kv, w, ratio, start_pos, offset, K):
    sc = torch.einsum("bshd,btd->bsht", q, kv)
    sc = _mod().index_score_reduce(sc, w, ratio, start_pos)
    idx = _mod().topk_select_post(sc, min(K, kv.shape[1]), ratio, offset, start_pos)
    return sc, idx
for (b, s, h, start, K) in [(1, 411, 64, 0, 2048), (1, 1, 64, 411, 2048), (1, 1, 8, 4096, 2048), (1, 300, 8, 0, 64), (2, 5, 8, 4000, 2048)]:
    ratio, d, off = 4, 128, 12345
    N = (start + s) // ratio
    q = torch.randn(b, s, h, d, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(b, N, d, device=dev, dtype=torch.bfloat16)
    w = torch.randn(b, s, h, device=dev, dtype=torch.bfloat16)
    sr, ir = ref(q, kv, w, ratio, start, off, K)
    sl, il = leaf(q, kv, w, ratio, start, off, K)
    fin = torch.isfinite(sr)
    sdiff = ((sl - sr.float()).abs() / (sr.float().abs() + 1)).masked_fill(~fin, 0).max().item()
    mismatch = 0; bad_rows = []
    for bi in range(b):
        for si in range(s):
            a = set(ir[bi, si].tolist()); c = set(il[bi, si].tolist())
            if a != c:
                mismatch += 1
                if len(bad_rows) < 3: bad_rows.append((bi, si, len(a - c), len(c - a), sorted(a - c)[:5], sorted(c - a)[:5], (ir[bi,si]==-1).sum().item(), (il[bi,si]==-1).sum().item()))
    print(f"b={b} s={s} h={h} start={start} K={K} N={N}: score_reldiff={sdiff:.2e} set_mismatch_rows={mismatch}/{b*s} shape ref={tuple(ir.shape)} leaf={tuple(il.shape)}", bad_rows)
