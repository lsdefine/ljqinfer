"""DSpark Markov chaining and confidence: parity with the released loop."""
import torch
from ops.decode.dspark import confidence, markov_refine, sample


class Markov:
    """Stand-in for the checkpoint's low-rank markov head."""

    def __init__(self, vocab, rank, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, rank, generator=g).cuda()
        self.head = (torch.randn(vocab, rank, generator=g) / rank ** 0.5).cuda()

    def __call__(self, ids):
        embed = self.emb[ids]
        return embed @ self.head.T, embed


def reference(logits, token, markov, temperature=0.0):
    """Direct transcription of DSparkBlock.forward_head's loop."""
    steps = logits.shape[1]
    out = token.new_empty(token.shape[0], steps + 1)
    out[:, 0] = token
    embeds = []
    for i in range(steps):
        bias, embed = markov(out[:, i])
        logits[:, i].add_(bias)
        embeds.append(embed)
        out[:, i + 1] = logits[:, i].argmax(-1) if temperature == 0 else None
    return out, torch.stack(embeds, dim=1)


def test_matches_released_loop():
    torch.manual_seed(0)
    markov = Markov(64, 8)
    token = torch.tensor([3, 11], device='cuda')
    logits = torch.randn(2, 5, 64, device='cuda')
    got_ids, _, got_emb = markov_refine(logits.clone(), token, markov)
    want_ids, want_emb = reference(logits.clone(), token, markov)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_emb, want_emb)
    assert got_ids.shape == (2, 6)


def test_chain_is_conditioned_on_previous_draft():
    """Without the Markov bias every position would be independent; changing
    the opening token must move later draft positions."""
    torch.manual_seed(1)
    markov = Markov(64, 8)
    logits = torch.randn(1, 5, 64, device='cuda')
    a, _, _ = markov_refine(logits.clone(), torch.tensor([3], device='cuda'), markov)
    b, _, _ = markov_refine(logits.clone(), torch.tensor([40], device='cuda'), markov)
    assert a[0, 1] != b[0, 1] or not torch.equal(a, b)


def test_greedy_is_deterministic():
    torch.manual_seed(2)
    markov = Markov(64, 8)
    logits = torch.randn(3, 5, 64, device='cuda')
    first, _, _ = markov_refine(logits.clone(), torch.tensor([1, 2, 3], device='cuda'), markov)
    again, _, _ = markov_refine(logits.clone(), torch.tensor([1, 2, 3], device='cuda'), markov)
    assert torch.equal(first, again)


def test_sampling_respects_temperature_zero_split():
    torch.manual_seed(3)
    logits = torch.randn(4, 32, device='cuda')
    assert torch.equal(sample(logits, 0.0), logits.argmax(-1))
    hot = torch.stack([sample(logits, 1.0) for _ in range(32)])
    assert hot.float().std(0).max() > 0


def test_confidence_is_fp32_per_position():
    torch.manual_seed(4)
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16, device='cuda')
    embeds = torch.randn(2, 5, 8, dtype=torch.bfloat16, device='cuda')
    proj = torch.randn(1, 24, dtype=torch.bfloat16, device='cuda')
    got = confidence(x, embeds, proj)
    assert got.shape == (2, 5) and got.dtype == torch.float32
    want = (torch.cat((x, embeds), -1).float() @ proj.float().T).squeeze(-1)
    assert torch.allclose(got, want)
