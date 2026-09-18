import torch, sys
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
from ops import _mod
m = _mod(); torch.manual_seed(0); eps=1e-6
for shape in [(8,1,64,512),(1,8,64,512),(3,64,512)]:
    x=(torch.randn(shape,device='cuda')*2).to(torch.bfloat16)
    a=x.clone(); a*=torch.rsqrt(a.square().mean(-1,keepdim=True)+eps)
    b=x.clone(); m.rms_scale_(b, eps)
    d=(a.float()-b.float()).abs(); print(shape, 'bitwise' if torch.equal(a,b) else f'maxdiff={d.max().item()} n={(d>0).sum().item()}/{d.numel()}')
print('DONE')
