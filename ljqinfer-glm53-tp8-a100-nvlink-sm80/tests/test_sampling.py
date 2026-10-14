"""Independent nucleus tests; set SAMPLING_DEVICE=cuda for the real CUDA path."""
import os
import pytest
import torch
from transformers.generation.logits_process import TopPLogitsWarper
from model.sampling import nucleus_cdf, inverse_cdf, pick_tokens

DEVICE = os.environ.get('SAMPLING_DEVICE', 'cpu')

@pytest.mark.parametrize('vocab', [1, 7, 127, 154880])
@pytest.mark.parametrize('p', [.01, .5, .95, 1.])
def test_hf_support_and_inverse_quantile(vocab, p):
    torch.manual_seed(814)
    scores = torch.randn(8, vocab, device=DEVICE) * 3
    if vocab > 1:
        scores[0].zero_()  # ties
        scores[1].fill_(-float('inf')); scores[1, 0] = 0  # masked point mass
        scores[2] *= 100  # concentrated distribution
    cdf, ids = nucleus_cdf(scores)
    # HF is the independent definition of the nucleus, not the old sampler.
    ref = TopPLogitsWarper(p)(None, scores)
    keep = (cdf > 1-p); keep[:, -1] = True
    actual_keep = torch.zeros_like(keep).scatter(1, ids, keep)
    assert torch.equal(actual_keep, torch.isfinite(ref)) or torch.equal(
        actual_keep & torch.isfinite(scores), torch.isfinite(ref))
    # Build a separate FP64 categorical distribution after HF truncation.
    probs = ref.double().softmax(-1).gather(1, ids)
    ref_cdf = probs.cumsum(-1).cpu()
    for u in [0., .001234, .123456, .612345, .99999]:
        uniforms = torch.full((8,), u, device=DEVICE)
        got = inverse_cdf(cdf, ids, uniforms, p).cpu()
        positions = torch.searchsorted(ref_cdf, torch.full((8,1), u, dtype=torch.float64), right=True).clamp_max(vocab-1)
        expected = ids.cpu().gather(1, positions).flatten()
        assert torch.equal(got, expected), (vocab, p, u, got, expected)

@pytest.mark.parametrize('p', [.01, .5, .95, 1.])
def test_distribution_against_hf(p):
    torch.manual_seed(42)
    logits = torch.tensor([.80,.16,.03,.01],device=DEVICE).log()
    tokens = pick_tokens(logits.expand(1, 40000, 4), [1], top_p=p).flatten()
    expected = TopPLogitsWarper(p)(None, logits[None]).softmax(-1).cpu()[0]
    freq = tokens.cpu().bincount(minlength=4).float()/tokens.numel()
    torch.testing.assert_close(freq, expected, atol=.012, rtol=0)
    assert (freq[expected==0] == 0).all()

@pytest.mark.parametrize('temps', [[0], [1], [0,1], [1,0,1,0], [1,1,1,1]])
def test_batch_and_no_mutation(temps):
    torch.manual_seed(18)
    x = torch.randn(len(temps),8,113,device=DEVICE)
    before = x.clone()
    got = pick_tokens(x, temps)
    assert got.shape == x.shape[:-1]
    assert torch.equal(x, before)
    for i,t in enumerate(temps):
        if t == 0: assert torch.equal(got[i], x[i].argmax(-1))
        else:
            ref = TopPLogitsWarper(.95)(None, x[i])
            assert torch.isfinite(ref.gather(1, got[i,:,None])).all()


def test_greedy_ties_no_rng_consumption():
    torch.manual_seed(15)
    state = torch.cuda.get_rng_state() if DEVICE == 'cuda' else torch.get_rng_state()
    x = torch.zeros(4,8,100,device=DEVICE)
    assert (pick_tokens(x,[0]*4)==0).all()
    after = torch.cuda.get_rng_state() if DEVICE == 'cuda' else torch.get_rng_state()
    assert torch.equal(state,after)


def test_rng_advances_and_reseeds():
    x=torch.zeros(1,1024,17,device=DEVICE)
    torch.manual_seed(53); a=pick_tokens(x,[1]); b=pick_tokens(x,[1])
    assert not torch.equal(a,b)
    torch.manual_seed(53); assert torch.equal(a,pick_tokens(x,[1]))

@pytest.mark.parametrize('p', [0,-1,1.1,float('nan'),True])
def test_invalid_p(p):
    with pytest.raises(ValueError): pick_tokens(torch.zeros(1,7,device=DEVICE),[1],top_p=p)

@pytest.mark.skipif(DEVICE != 'cpu', reason='CUDA bad distributions assert asynchronously and poison the context')
@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_invalid_distribution(bad):
    with pytest.raises(ValueError): pick_tokens(torch.full((1,5),bad),[1])

@pytest.mark.skipif(DEVICE != 'cuda', reason='CUDA graph only')
def test_cuda_graph_rng_advances():
    x=torch.zeros(1,8,127,device='cuda')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): pick_tokens(x,[1])
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): y=pick_tokens(x,[1])
    graph.replay(); a=y.clone()
    graph.replay(); b=y.clone()
    torch.cuda.synchronize()
    assert not torch.equal(a,b)
