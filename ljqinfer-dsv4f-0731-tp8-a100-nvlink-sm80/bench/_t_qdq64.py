import torch, sys
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
from ops import _mod
import model.kernels_torch as KT
m = _mod()
torch.manual_seed(0)
for shape, cut in [((8, 1, 576), 512), ((1, 8, 576), 512), ((3, 7, 640), 640), ((2, 2, 64), 64)]:
    x = (torch.randn(shape, device='cuda') * 3).to(torch.bfloat16)
    x[0, 0, :64] = 0
    a = x.clone(); b = x.clone()
    KT.act_quant(a[..., :cut], 64, 'ue8m0', torch.float8_e8m0fnu, True)
    m.fp8_qdq64_(b[..., :cut])
    print(shape, cut, 'bitwise' if torch.equal(a, b) else 'MISMATCH max=%g' % (a.float()-b.float()).abs().max().item(), 'tail-ok' if torch.equal(a[..., cut:], x[..., cut:]) else 'tail-changed')
print('DONE')
