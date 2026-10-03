"""Speculative decode over an already prefilled slot.

forward() runs a whole draft window through every layer and publishes nothing:
each source layer parks projected and compressed rows in scratch. commit() then
installs exactly the accepted prefix in the canonical residual and paged pools,
so a rejected draft leaves committed state unchanged. Logits cover
every row of the window because verification compares them all.

main_hidden carries the same target-layer features prefill hands the drafter,
so the MTP head sees one representation, whichever pass produced it.
"""
from dataclasses import dataclass
import torch

from ops.prefill import residual as r
from model.decode_attention import DecodeScratch
from model.prefill import PrefillModel
from model.prefill_config import validate_config


@dataclass
class DecodeOutput:
    """logits [Q, vocab]; main_hidden [Q, dim*len(target_layers)] or None."""
    logits: torch.Tensor
    main_hidden: torch.Tensor


def _requests(slot):
    """Slots in batch order; a single slot is a batch of one."""
    return (slot,) if isinstance(slot, int) else tuple(slot)


def _cursors(past, slot):
    """Committed cursor per request; each request advances on its own."""
    return (past.pos[slot] if isinstance(slot, int)
            else tuple(past.pos[s] for s in slot))


def _per_request(slot, value):
    """Spread a per-request argument out in slot order, batched or not."""
    return (value,) if isinstance(slot, int) else tuple(value)


def _histories(slot, history_tokens):
    """Raw pre-window history per request; Engram hashes each against its own."""
    if isinstance(slot, int):
        return (history_tokens,)
    return tuple(history_tokens) if history_tokens else ((),) * len(slot)


def _joined(slot, tokens, windows):
    """The batch's windows as one flat token run, in whatever form they came.

    Captured steps hand over a device tensor, so the windows are concatenated
    on the device; nothing is pulled back to the host on the decode path.
    """
    if isinstance(slot, int):
        return tokens
    if torch.is_tensor(windows[0]):
        return torch.cat(windows)
    return [t for window in windows for t in window]


class DecodeModel:
    _check_past = PrefillModel._check_past
    _embed = PrefillModel._embed

    def __init__(self, config, blocks, embed, head, norm, *, target_layers=()):
        validate_config(config)
        self.c, self.blocks = dict(config), tuple(blocks)
        self.embed, self.head, self.norm = embed, head, norm
        self.targets = frozenset(target_layers)
        if [b.layer for b in self.blocks] != list(range(config['n_layers'])):
            raise ValueError('decode requires every backbone layer in order')
        for b in self.blocks:
            if (b.layer in config['engram_layer_ids']) != (b.engram is not None):
                raise ValueError('missing Engram row provider')
        if not self.targets <= {b.layer for b in self.blocks}:
            raise ValueError('drafter target layer outside the backbone')

    @torch.inference_mode()
    def release_engram_gates(self):
        """Unblock a launched replay whose staging failed part way through."""
        for block in self.blocks:
            if block.engram is not None:
                block.engram.release_gate()

    def stage_engram(self, tokens, *, past, slot, history_tokens=(), gated=False):
        """Do the Engram host work for the next window before a graph replay.

        Hashing, the table gather and the H2D of the rows cannot live inside a
        captured graph, so the caller runs this eagerly once per round and the
        replayed body reads the device buffer it fills.
        """
        starts = _cursors(past, slot)
        staged = [b for b in self.blocks if b.engram is not None]
        if not staged:
            return
        hist = history_tokens or None
        # Warm the shared hash memo on this thread so the pool workers only
        # do their own table gather, then overlap those gathers: the host
        # table lookup is the single largest non-replay cost per round. A
        # batch warms every request's window in one call.
        # The memo keys on the identity of the token buffer, but a captured
        # batch reuses one staging buffer for the process lifetime: same
        # object every round, new tokens inside it. Drop the memo per pass
        # or the rows keep describing the window this batch has left.
        staged[0].engram.rows.hasher._memos = ()
        staged[0].engram.rows.ids(starts, tokens, history_tokens)
        for block in staged:
            block.engram.prefetch(slot=slot, start=starts, tokens=tokens,
                                  history_tokens=hist)
        for block in staged:
            block.engram.stage(slot=slot, start=starts, tokens=tokens,
                               history_tokens=hist, gated=gated)

    @torch.inference_mode()
    def forward(self, tokens, *, past, slot, scratch=None, history_tokens=()):
        """Score a draft window; the pools keep the state they had on entry."""
        self._check_past(past)
        slots, starts = _requests(slot), _cursors(past, slot)
        windows = _per_request(slot, tokens)
        histories = _histories(slot, history_tokens)
        ngram = self.c['engram_max_ngram_size'] - 1
        for s, begin, window, history in zip(slots, _per_request(slot, starts),
                                             windows, histories):
            if s in past.free_slots or s in past.replay_pending:
                raise ValueError('decode requires a positioned hot slot')
            if not len(window):
                raise ValueError('empty draft window')
            if len(history) != min(begin, ngram):
                raise ValueError('missing Engram history before draft window')
            past.ensure(s, begin + len(window))
        scratch = DecodeScratch() if scratch is None else scratch
        # Rotation rows are selected once per pass and then reused by every
        # layer; a reused scratch must not carry another pass's rows over.
        scratch.rope_rows.clear()
        h, pre = self._embed(_joined(slot, tokens, windows))
        for block in self.blocks:
            # A staged block already holds this window's rows in its device
            # buffer: prefetching again would gather them a second time.
            staged = None if block.engram is None else block.engram._rows_buf
            if block.engram is not None and (staged is None
                                             or staged.shape[0] != h.shape[0]):
                block.engram.prefetch(slot=slot, start=starts, tokens=tokens,
                                      history_tokens=history_tokens)
        targets = []
        for block in self.blocks:
            h = block.prepare(h, slot=slot, start=starts, tokens=tokens,
                              history_tokens=history_tokens)
            h, pre = block(h, pre, past=past, slot=slot, start=starts, scratch=scratch)
            if block.layer in self.targets:
                targets.append(h.mean(dim=-2))
        hidden = r.collapse_norm(h, pre, self.norm, self.c['norm_eps'])
        return DecodeOutput(self.head(hidden),
                            torch.cat(targets, -1) if targets else None)

    @torch.inference_mode()
    def commit(self, accepted, *, past, slot, scratch):
        """Publish the accepted prefix of every scored window and advance."""
        slots = _requests(slot)
        starts = _per_request(slot, _cursors(past, slot))
        counts = _per_request(slot, accepted)
        width = len(next(iter(scratch.rows.values()))) // len(slots)
        for block in self.blocks:
            block.attention.commit(past=past, slots=slots, starts=starts,
                                   accepted=counts, scratch=scratch,
                                   width=width)
        for one, start, count in zip(slots, starts, counts):
            past.set_pos(one, start + count)
