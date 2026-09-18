
import torch, sys
sys.path.insert(0,'.')
from ops import _mod
from model.qmath import rotate_activation, fp4_act_quant_sim
torch.manual_seed(0)
ok=True
for shp in [(8,8,128),(64,128),(3,5,128),(16,64),(2,256)]:
    for sc in [1.0, 0.01, 30.0, 1e-30]:
        x=(torch.randn(*shp,device='cuda')*sc).bfloat16()
        ref=rotate_activation(x.clone()); fp4_act_quant_sim(ref,32)
        out=x.clone(); _mod().had_fp4_qdq_(out)
        same=torch.equal(out.view(torch.int16),ref.view(torch.int16))
        nd=(out.view(torch.int16)!=ref.view(torch.int16)).sum().item()
        print(shp,sc,'bitwise' if same else 'DIFF %d/%d maxabs %g'%(nd,out.numel(),(out.float()-ref.float()).abs().max().item()))
        ok&=same
print('ALL_BITWISE' if ok else 'NOT_BITWISE')
