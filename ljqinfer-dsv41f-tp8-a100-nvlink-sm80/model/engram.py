"""Released Engram hash semantics; metadata setup adapted from official engram.py."""
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from torch import nn


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram), each split
    over `n_heads` heads. Every (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


def _flat(values):
    """Token sequence as a plain tuple of ints.

    A torch tensor iterates into 0-dim tensors, and a CUDA one turns every
    element into its own synchronising D2H, so convert in one step.
    """
    if isinstance(values, torch.Tensor):
        return tuple(values.to('cpu', torch.int64).tolist())
    return tuple(values)


class EngramHash:
    """CPU hash metadata only. Request owns raw history; no slot/token cache."""
    def __init__(self, config, tokenizer):
        from types import SimpleNamespace
        args = SimpleNamespace(**config)
        self.layout = EngramLayout.from_args(args)
        if self.layout is None:
            raise ValueError('Engram configuration required')
        token_map, size = build_compressed_token_map(tokenizer)
        if size != args.engram_compressed_vocab_size:
            raise ValueError(f'compressed vocabulary mismatch: {size}')
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[args.engram_pad_id]
        self.primes = torch.tensor(self.layout.primes, dtype=torch.int64)
        flat = self.primes.flatten(1)
        self.offsets = flat.cumsum(-1) - flat
        if tuple(flat.sum(-1).tolist()) != self.layout.num_embeddings:
            raise ValueError('Engram table sizes do not match hash buckets')
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids,
            self.layout.max_ngram_size, size)

    # One entry per request in flight: a batched round hashes a different
    # window per request, and every engram layer then asks for all of them
    # again, so a single-slot memo would thrash.
    _MEMO_KEEP = 8
    _memos = ()

    def __call__(self, tokens, *, start, history_tokens=(), token_mask=None,
                 history_mask=None):
        """Hash once per chunk: every engram layer slices its own column out.

        The result is shared, so callers must treat it as read-only.
        """
        plain = token_mask is None and history_mask is None
        if plain:
            for memo in self._memos:
                if (memo[0] is tokens and memo[1] is history_tokens
                        and memo[2] == start):
                    return memo[3]
        if plain:
            # Decode hot path: a handful of tokens per round, where ~25 tiny
            # torch CPU ops cost more than the arithmetic. Same math in numpy.
            ids = self._hash_np(tokens, start=start, history_tokens=history_tokens)
        else:
            ids = self._hash(tokens, start=start, history_tokens=history_tokens,
                             token_mask=token_mask, history_mask=history_mask)
        if plain:
            self._memos = (self._memos +
                           ((tokens, history_tokens, start, ids),))[-self._MEMO_KEEP:]
        return ids

    _np = None

    def _hash_np(self, tokens, *, start, history_tokens=()):
        """Unmasked `_hash` on numpy; bit-identical, one allocation per op."""
        n = self.layout.max_ngram_size
        if start < 0 or len(history_tokens) != min(start, n-1):
            raise ValueError('exact raw token history required before chunk')
        if self._np is None:
            self._np = (self.token_map.numpy(), self.multipliers.numpy(),
                        self.primes.numpy(), self.offsets.numpy())
        token_map, multipliers, primes, offsets = self._np
        raw = np.fromiter(_flat(history_tokens)+_flat(tokens), dtype=np.int64)
        if raw.size and (raw.min() < 0 or raw.max() >= len(token_map)):
            raise ValueError('token outside vocabulary')
        source, h, q = token_map[raw], len(history_tokens), len(tokens)
        pos = np.arange(q, dtype=np.int64) + h
        lookback = np.empty((q, n), dtype=np.int64)
        for shift in range(n):
            lookback[:, shift] = np.where(pos < shift, self.pad_id,
                                          source[np.maximum(pos-shift, 0)])
        products = lookback[:, None, :] * multipliers          # [q, L, n]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, n):
            rolling = rolling ^ products[..., i]
            hashes.append(rolling[..., None] % primes[:, i-1])   # [q, L, heads]
        out = np.concatenate(hashes, -1) + offsets
        return torch.from_numpy(out)

    def _hash(self, tokens, *, start, history_tokens=(), token_mask=None,
              history_mask=None):
        n = self.layout.max_ngram_size
        if start < 0 or len(history_tokens) != min(start, n-1):
            raise ValueError('exact raw token history required before chunk')
        raw = torch.tensor(tuple(history_tokens)+tuple(tokens), dtype=torch.int64)
        if raw.numel() and (raw.min() < 0 or raw.max() >= len(self.token_map)):
            raise ValueError('token outside vocabulary')
        source = self.token_map[raw]
        def mask(values, count):
            result = torch.ones(count, dtype=torch.bool) if values is None else torch.as_tensor(values, dtype=torch.bool, device='cpu')
            if result.shape != (count,):
                raise ValueError('mask shape mismatch')
            return result
        live = torch.cat((mask(history_mask,len(history_tokens)), mask(token_mask,len(tokens))))
        source = torch.where(live, source, -1)
        pos = torch.arange(len(tokens)) + len(history_tokens)
        blocked = torch.zeros(len(tokens), dtype=torch.bool)
        lookback = []
        for shift in range(n):
            value = source[(pos-shift).clamp_min(0)]
            blocked |= (pos < shift) | (value == -1)
            lookback.append(torch.where(blocked, self.pad_id, value))
        products = torch.stack(lookback,-1)[:,None,:] * self.multipliers
        rolling, hashes = products[...,0], []
        for i in range(1,n):
            rolling = torch.bitwise_xor(rolling,products[...,i])
            hashes.append(rolling[...,None] % self.primes[:,i-1])
        return torch.cat(hashes,-1) + self.offsets


class EngramRows:
    """Bind one layer's host table to hash->gather->dequantize; no hidden state."""
    def __init__(self, hasher, layer, table, *, rank=None, workspace=None):
        self.hasher, self.table, self.rank = hasher, table, rank
        self.workspace = workspace
        self.layer = hasher.layout.layer_ids.index(layer)

    def mask_key(self, slot):
        return tuple((offset, len(x)) for offset, x in
                     getattr(self.hasher, 'image_spans', {}).get(slot, ()))

    def _masked_ids(self, slot, start, tokens, history):
        spans = self.mask_key(slot)
        if not spans:
            return self.hasher(tokens, start=start, history_tokens=history)[:, self.layer]
        def live(pos):
            return not any(a <= pos < a+n for a, n in spans)
        return self.hasher(tokens, start=start, history_tokens=history,
                           token_mask=tuple(live(start+i) for i in range(len(tokens))),
                           history_mask=tuple(live(start-len(history)+i)
                                              for i in range(len(history))))[:, self.layer]

    def ids(self, start, tokens, history_tokens=(), *, slot=None):
        """Hash this layer's column for one request or for a whole batch.

        A batch arrives as one cursor, token window and history per request:
        each request hashes against its own cursor, and the columns are laid
        out back to back so the gather below stays a single host pass.
        """
        if isinstance(start, int):
            return self._masked_ids(slot, start, tokens, history_tokens)
        history = history_tokens or ((),) * len(start)
        slots = (None,) * len(start) if slot is None else slot
        return torch.cat([self._masked_ids(sl, s, t, h)
                          for sl, s, t, h in zip(slots, start, tokens, history)])

    def gather_host(self, slot, start, tokens, history_tokens=()):
        """Token-only work (hash + host table gather); safe off the main thread."""
        ids = self.ids(start, tokens, history_tokens, slot=slot)
        return self.table.gather(ids,rank=self.rank)

    def rows_host(self, slot, start, tokens, history_tokens=(), out=None):
        """Fused hash -> gather -> dequantize; host bf16 rows, no intermediates."""
        ids = self.ids(start, tokens, history_tokens, slot=slot)
        fused = getattr(self.table, 'rows_bf16', None)
        if fused is None:  # tables that only promise gather() keep the old path
            return self.dequantize(self.table.gather(ids, rank=self.rank))
        return fused(ids, rank=self.rank, out=out)

    def dequantize(self, gathered):
        values, scales = gathered
        if self.workspace is not None:
            return self.workspace(values.view(torch.uint8), scales)
        return (values.float().unflatten(-1,(-1,32)) *
                (scales.float()-127).exp2()[...,None]).flatten(-2).to(torch.bfloat16)

    def __call__(self, slot, start, tokens, history_tokens=()):
        return self.dequantize(self.gather_host(slot,start,tokens,history_tokens))
