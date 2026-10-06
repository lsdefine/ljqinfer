"""Does the wiring select exactly what the chain is supposed to select?

Checks three things that decide whether this can replace the hand chain:
the selected set equals the exact causal top-k over committed rows plus the
six uncommitted ones, the returned ids follow the id>=start pending
convention, and the canonical bank outside the reserved page is untouched.
"""
import sys, torch, torch_npu

sys.path.insert(0, '/data/ljqinfer_dsv41f_tp8')
from ops.decode.vendor_select import VendorSelect

B, R, H, D, RPP, BLK, SC = 2, 6, 32, 128, 2048, 128, 512
dev = torch.device('npu:0')
torch.manual_seed(0)


class Pool:
    ratio = 1

    def __init__(self, data):
        self.data = data


pages = 5                       # four usable pages, one reserved scratch page
data = torch.randn(pages, RPP, D, dtype=torch.bfloat16, device=dev)
table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device=dev)
slots = torch.tensor([0, 1], dtype=torch.int64, device=dev)
start = torch.tensor([300, 4000], dtype=torch.int64, device=dev)
pending = torch.randn(B, R, D, dtype=torch.bfloat16, device=dev)
iq = torch.randn(B * R, H, D, dtype=torch.bfloat16, device=dev)
hw = torch.randn(B * R, H, dtype=torch.bfloat16, device=dev)
selected = torch.zeros(B, R, SC, dtype=torch.int64, device=dev)

before = data[:pages - 1].clone()
sel = VendorSelect(bank=data, table=table, slots=slots, start=start,
                   pending=pending, batch=B, heads_local=H, world=1,
                   sparse_count=SC).bind()
sel.prepare()
sel.dispatch(iq, hw, selected)
torch.npu.synchronize()

print('bank untouched:', bool(torch.equal(before, data[:pages - 1])))
flat = data.view(-1, D)
for b in range(B):
    n, blk = int(start[b]), int(start[b]) // 128
    sb = sel.scratch + 2 * b
    phys = int(table[int(slots[b]), blk // 16]) * 16 + blk % 16
    off = n - blk * 128
    got = flat[sb * 128: sb * 128 + 128]
    ref = flat[phys * 128: phys * 128 + 128].clone()
    ref[off:off + 6] = pending[b, :min(6, 128 - off)] if off + 6 <= 128 else ref[off:off + 6]
    print(f'scratch b{b} off={off} copy_ok={bool(torch.equal(got, ref))} '
          f'pending_ok={bool(torch.equal(flat[sb * 128 + off: sb * 128 + off + 6], pending[b]))}')

# Which query row does the op use when it scores the uncommitted rows?
b = 1
n = int(start[b])
q = iq[b * R:(b + 1) * R].float()
wt = hw[b * R:(b + 1) * R].float()
keys = torch.cat([flat[int(table[int(slots[b]), k // 16]) * 16 * 128 + (k % 16) * 128:
                       int(table[int(slots[b]), k // 16]) * 16 * 128 + (k % 16) * 128 + 128]
                  for k in range(n // 128 + 1)]).float()[:n]
keys = torch.cat([keys, pending[b].float()])
S = torch.einsum('rhd,kd->rhk', q, keys).relu().mul(wt[:, :, None]).sum(1)   # [R, n+6]
for j in (3, 4, 5):
    ids = selected[b, j].tolist()
    for p in (n, n + 2, n + 3, n + 5):
        if p in ids:
            r = ids.index(p)
            near = S[j, ids[max(r - 1, 0)]].item(), S[j, ids[min(r + 1, len(ids) - 1)]].item()
            print(f'row{j} id{p} rank={r} neighbour_scores={near[0]:.2f}/{near[1]:.2f} '
                  f'score_by_row={[round(S[t, p].item(), 2) for t in range(R)]}')

ok = True
for b in range(B):
    n = int(start[b])
    rows = torch.cat([data[int(table[slots[b], p])] for p in range(table.shape[1])])
    keys = torch.cat([rows[:n], pending[b]]).float()          # [n+6, D]
    for j in range(R):
        s = (torch.einsum('hd,kd->hk', iq.view(B, R, H, D)[b, j].float(), keys)
             .relu() * hw.view(B, R, H)[b, j].float()[:, None]).sum(0)
        s = s[:n + j + 1]                                      # causal for this row
        k = min(SC, s.numel())
        want = s.topk(k).indices.sort().values
        got = selected[b, j, :k].sort().values
        same = bool(torch.equal(want, got))
        ok &= same
        if not same:
            miss = sorted(set(want.tolist()) - set(got.tolist()))[:6]
            extra = sorted(set(got.tolist()) - set(want.tolist()))[:6]
            print(f'  b{b} row{j} miss={miss} {[round(float(s[i]),3) for i in miss]} '
                  f'extra={extra} {[round(float(s[i]),3) for i in extra]} '
                  f'cut={round(float(s.topk(k).values[-1]),3)}')
        if j in (0, R - 1):
            tail = selected[b, j, k:]
            print(f'b{b} row{j} k={k} exact={same} '
                  f'pending={int((selected[b, j, :k] >= n).sum())} '
                  f'tail={tail[:4].tolist() if tail.numel() else []}')
print('ALL EXACT:', ok)
print('DONE')
