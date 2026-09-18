import json
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_context_chunked_prefill_workspace():
    from model.cuda_attention import paged_attention
    from model.config import KV_PAGE_SIZE
    torch.manual_seed(731)
    ps, total, h, dim = KV_PAGE_SIZE, 262144, 6, 256
    pages = (total + ps - 1) // ps
    k = torch.randn(1, pages, ps, 1, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache = SimpleNamespace(spec=SimpleNamespace(page_size=ps), host_page_table=[list(range(pages))], k=k, v=v)
    q = torch.randn(12288, h, dim, device="cuda", dtype=k.dtype)
    cases = [(min(12288,total-start),min(start+12288,total)) for start in range(0,total,12288)]
    cases += [(n,total) for n in (1,2,17,256,512,1024,2048,4096,12288)]
    for nq, nk in cases:
        y = paged_attention(q[:nq],cache,0,0,nk)
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        for row in sorted(set((0,nq//2,nq-1))):
            end = nk-nq+row+1
            keys = k[0].flatten(0,1)[:end,0].float()
            vals = v[0].flatten(0,1)[:end,0].float()
            prob = torch.softmax(q[row].float() @ keys.T / dim**0.5,dim=-1)
            ref = prob @ vals
            torch.testing.assert_close(y[row].float(),ref,atol=0.008,rtol=0.01)
        wrapper = cache._cuda_prefill_attention[1]
        print(json.dumps({"query":nq,"kv":nk,"workspace_bytes":cache._cuda_prefill_attention[0].numel(), "plan":list(wrapper._plan_info)}),flush=True)
