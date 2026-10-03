"""DSpark draft attention: semi-autoregressive visibility, reference-gather parity."""
import torch
from ops.decode.dspark import draft_attend


def reference(q, window_kv, draft_kv, sink, scale, start):
    """Direct transcription of the released implementation: a shared topk_idxs
    row per query, gathered out of cat(ring, draft)."""
    win = window_kv.shape[0]
    t = q.shape[0]
    bank = torch.cat((window_kv, draft_kv), dim=0)
    ix = torch.cat([torch.arange(min(win, start + 1)), win + torch.arange(t)])
    ix = ix.int().view(1, -1).expand(t, -1).to(q.device).long()
    rows = bank[ix].float()
    logits = torch.einsum("thd,tkd->thk", q.float(), rows) * scale
    logits = torch.cat((logits, sink.float().expand(t, q.shape[1]).unsqueeze(-1)), dim=-1)
    prob = logits.softmax(-1)[..., :-1]
    return torch.einsum("thk,tkd->thd", prob, rows).to(q.dtype)


def test_matches_reference_gather():
    torch.manual_seed(0)
    q = torch.randn(5, 8, 64, dtype=torch.bfloat16)
    window_kv = torch.randn(128, 64, dtype=torch.bfloat16)
    draft_kv = torch.randn(5, 64, dtype=torch.bfloat16)
    sink = torch.randn(8)
    got = draft_attend(q, window_kv, draft_kv, sink, scale=64 ** -0.5)
    want = reference(q, window_kv, draft_kv, sink, 64 ** -0.5, start=5000)
    assert torch.equal(got, want)


def test_short_history_matches_reference():
    torch.manual_seed(1)
    start = 40
    q = torch.randn(5, 8, 64, dtype=torch.bfloat16)
    window_kv = torch.randn(start + 1, 64, dtype=torch.bfloat16)
    draft_kv = torch.randn(5, 64, dtype=torch.bfloat16)
    sink = torch.randn(8)
    got = draft_attend(q, window_kv, draft_kv, sink, scale=64 ** -0.5)
    want = reference(q, window_kv, draft_kv, sink, 64 ** -0.5, start=start)
    assert torch.equal(got, want)


def test_every_draft_row_sees_every_other():
    """Semi-autoregressive: masking a later draft row must change row 0."""
    torch.manual_seed(2)
    q = torch.randn(5, 4, 64)
    window_kv = torch.randn(8, 64)
    draft_kv = torch.randn(5, 64)
    sink = torch.zeros(4)
    full = draft_attend(q, window_kv, draft_kv, sink, scale=64 ** -0.5)
    dropped = draft_kv.clone()
    dropped[4] = 0.0
    other = draft_attend(q, window_kv, dropped, sink, scale=64 ** -0.5)
    assert not torch.allclose(full[0], other[0])


def test_rejects_mismatched_draft_rows():
    q = torch.randn(5, 4, 64)
    try:
        draft_attend(q, torch.randn(8, 64), torch.randn(4, 64), torch.zeros(4), scale=1.0)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a short draft KV block")
